import asyncio
import json
import os
import warnings
from argparse import ArgumentParser
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

from src.data import HFDepthQALoader, filter_data_dict, slice_data_dict
from src.model import OpenAILLM
from src.utils import (
    SAMPLING_PARAMS_OPENAI,
    SYSTEM_PROMPT_ZERO_SHOT,
    USER_PROMPT_TEMPLATE_ZERO_SHOT,
    USER_PROMPT_TEMPLATE_LAST_TURN,
    get_output_path
)

DEBUG = False


async def multiturn_completions(
    model,
    inputs,
    total_len,
    max_retries=5
):
    system_prompt = SYSTEM_PROMPT_ZERO_SHOT
    chat_dict = defaultdict(str)

    # Adjust batch size to fit the number of inputs
    # VLLM supports adaptive batch size already
    total_len = len(inputs)
    batched_outputs = []

    # Process initial batches with progress bar
    print("Processing initial chat...")
    for i in tqdm(
        range(0, len(inputs)), total=total_len, desc="Initial Chat"
    ):
        chat_input = inputs[i]

        for j, inst in enumerate(chat_input):
            if j == 0:
                input_ = apply_template_chat(system_prompt, inst["input"])
            else:
                input_.append({"role": "user", "content": inst["input"]})

            if inst["id"] in chat_dict:
                input_.append({"role": "assistant", "content": chat_dict[inst["id"]]})
                continue
            
            
            output = await model.completions([input_], **SAMPLING_PARAMS_OPENAI)
            chat_dict[inst["id"]] = output[0]
            input_.append({"role": "assistant", "content": output[0]})

        batched_outputs.append(input_)

    # Identify failed instances and prepare for retries

    to_retry_inputs = []
    to_retry_indices = []

    for i, output in enumerate(batched_outputs):
        for chat in output:
            if "assistant" in chat:
                if chat["content"] is None:  # Parsing failed
                    to_retry_inputs.append(inputs[i])
                    to_retry_indices.append(i)

    # Retry logic with progress bar
    retries = 0
    while to_retry_inputs and retries < max_retries:
        retries += 1
        print(f"Retrying failed batches: Attempt {retries}/{max_retries}")
        retry_outputs = []
        for i in tqdm(
            range(0, len(to_retry_inputs)), desc=f"Retry Attempt {retries}"
        ):
            chat_input = inputs[i]
            for j, inst in enumerate(chat_input):
                if j == 0:
                    input_ = apply_template_chat(system_prompt, inst["input"])
                else:
                    input_.append({"role": "user", "content": inst["input"]})

                if inst["id"] in chat_dict:
                    input_.append({"role": "assistant", "content": chat_dict[inst["id"]]})
                    continue
                output = await model.completions([input_], **SAMPLING_PARAMS_OPENAI)
                chat_dict[inst["id"]] = output[0]
                input_.append({"role": "assistant", "content": output[0]})

            batched_outputs.append(input_)

        new_to_retry_inputs = []
        new_to_retry_indices = []
        for idx, (retry_idx, output) in enumerate(zip(to_retry_indices, retry_outputs)):
            if output is None:  # Still failing
                new_to_retry_inputs.append(to_retry_inputs[idx])
                new_to_retry_indices.append(to_retry_indices[idx])
            else:
                batched_outputs[retry_idx] = output  # Update with successful retry

        to_retry_inputs = new_to_retry_inputs
        to_retry_indices = new_to_retry_indices

    # Final aggregation and printing
    outputs_len = len(chat_dict)
    print(f"Processed {outputs_len} / {total_len} instances.")
    '''
    if outputs_len < total_len:
        warnings.warn("Some instances failed.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )
    '''
    return chat_dict


def apply_template_chat(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def prepare_inputs_multiturn(
    questions: Dict[str, Dict],
    nodes: Dict[str, Dict],
    node_to_q: Dict[str, str]
) -> List[str]:

    chat_inputs = []
   
    for nodeid, node in nodes.items():
        chat = []
        target_question_d = questions[node_to_q[nodeid]]
        target_question = target_question_d["question"]

        for predec_nodeid in node["direct_predecessors"]:
            predec_question_d = questions[node_to_q[predec_nodeid]]
            predec_question = predec_question_d["question"]
            user_prompt = USER_PROMPT_TEMPLATE_ZERO_SHOT.format(question=predec_question)
            chat.append({"id":predec_nodeid, "input": user_prompt})
        last_user_prompt = USER_PROMPT_TEMPLATE_LAST_TURN.format(question=target_question)
        chat.append({"id": nodeid, "input": last_user_prompt})

        chat_inputs.append(chat)

    return chat_inputs


async def main(args):
    global DEBUG
    DEBUG = args.debug

    # Load data
    dataloader = HFDepthQALoader()
    questions, nodes, node_to_q = dataloader.load_data()
    print(f"Loaded {len(questions)} questions and {len(nodes)} nodes.")

    # Load model
    model = OpenAILLM(args.model_name)
    nodes = filter_data_dict(nodes, lambda node: node["depth"] > 1)

    if DEBUG:
        nodes = slice_data_dict(nodes, start=0, end=5)

    inputs = prepare_inputs_multiturn(
        questions,
        nodes,
        node_to_q
    )

    if DEBUG:
        inputs = inputs[:5]

    # Inference
    predictions = await multiturn_completions(model, inputs, len(node_to_q))

    # Save results
    results = {}
    for idx, nodeid in enumerate(predictions.keys()):
        results[nodeid] = questions[node_to_q[nodeid]]
        results[nodeid].update({"predicted_answer": predictions[nodeid]})

    output_path = get_output_path(
        output_file=args.output_file, default_output_dir="../../outputs/inference"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    # I/O arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of model hosted in Hugging Face under AutoModelForCausalLM",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="kaist-ai/DepthQA",
        help="Dataset name in Hugging Face (for zero-shot) or local zero-shot JSON output file (for prompt-*)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file name. Unless the parent directory is specified, will be saved under outputs/inference by default.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if output file exists.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    assert args.output_file.endswith(".json"), "Output file must be a JSON file."

    output_path = get_output_path(
        output_file=args.output_file, default_output_dir="../../outputs/inference"
    )
    assert not (
        os.path.exists(output_path) and not args.force_rerun
    ), f"Output file {output_path} already exists. Skipping inference."

    asyncio.run(main(args))

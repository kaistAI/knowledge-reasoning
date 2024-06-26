import asyncio
import json
import os
import warnings
from argparse import ArgumentParser
from typing import Dict, List

from tqdm import tqdm

from src.data import HFDepthQALoader, filter_data_dict, slice_data_dict
from src.model import OpenAILLM
from src.utils import (
    SAMPLING_PARAMS_OPENAI,
    SYSTEM_PROMPT_CTX,
    SYSTEM_PROMPT_ZERO_SHOT,
    USER_PROMPT_TEMPLATE_CTX,
    USER_PROMPT_TEMPLATE_ZERO_SHOT,
    format_messages,
    get_output_path,
    show_random_inputs,
)

DEBUG = False


# Model inference (Use offline batching)
async def batch_completions_with_retries(
    model,
    inputs,
    batch_size,
    max_retries=5,
):

    batched_outputs = []

    total_batches = len(inputs) // batch_size + (
        1 if len(inputs) % batch_size > 0 else 0
    )
    total_len = len(inputs)

    # Process initial batches with progress bar
    print("Processing initial batches...")
    for i in tqdm(
        range(0, len(inputs), batch_size), total=total_batches, desc="Initial Batches"
    ):
        batch_inputs = inputs[i : i + batch_size]
        batch_outputs = await model.completions(batch_inputs, **SAMPLING_PARAMS_OPENAI)
        batched_outputs.extend(batch_outputs)

    # Identify failed instances and prepare for retries
    to_retry_inputs = []
    to_retry_indices = []
    for i, output in enumerate(batched_outputs):

        if output is None:  # Parsing failed
            to_retry_inputs.append(inputs[i])
            to_retry_indices.append(i)

    # Retry logic with progress bar
    retries = 0
    while to_retry_inputs and retries < max_retries:
        retries += 1
        print(f"Retrying failed batches: Attempt {retries}/{max_retries}")
        retry_outputs = []
        for i in tqdm(
            range(0, len(to_retry_inputs), batch_size), desc=f"Retry Attempt {retries}"
        ):
            batch_inputs = to_retry_inputs[i : i + batch_size]
            batch_outputs = await model.completions(
                batch_inputs, **SAMPLING_PARAMS_OPENAI
            )

            assert len(batch_outputs) == len(batch_inputs)
            retry_outputs.extend(batch_outputs)

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
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed to generate.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(f"Failed to generate for {total_len - outputs_len} instances.")

    predictions = []

    for output in tqdm(batched_outputs, desc="Finalizing"):
        if output is not None:
            predictions.append(output)
        else:
            predictions.append(None)
    if DEBUG:
        print("Checking the results")
        for prediction in predictions[:5]:
            print(prediction)

    return predictions


def prepare_inputs_zero_shot(questions: Dict[str, Dict]) -> List[str]:
    inputs = []
    system_prompt = SYSTEM_PROMPT_ZERO_SHOT
    for question_d in questions.values():
        target_question = question_d["question"]
        user_prompt = USER_PROMPT_TEMPLATE_ZERO_SHOT.format(question=target_question)
        messages = format_messages(system_prompt, user_prompt)
        inputs.append(messages)

    return inputs


def prepare_inputs_ctx(
    questions: Dict[str, Dict],
    nodes: Dict[str, Dict],
    node_to_q: Dict[str, str],
    use_gold: bool,
) -> List[str]:
    inputs = []
    system_prompt = SYSTEM_PROMPT_CTX
    for nodeid, node in nodes.items():
        target_question_d = questions[node_to_q[nodeid]]
        target_question = target_question_d["question"]
        predecessor_pairs = ""
        for predec_nodeid in node["direct_predecessors"]:
            predec_question_d = questions[node_to_q[predec_nodeid]]
            predec_question = predec_question_d["question"]
            if use_gold:
                predec_answer = predec_question_d["answer"]
            else:
                predec_answer = predec_question_d["predicted_answer"]
            pair = f"Q: {predec_question}\nA: {predec_answer}\n"
            predecessor_pairs += pair
        user_prompt = USER_PROMPT_TEMPLATE_CTX.format(
            qa_pairs=predecessor_pairs, question=target_question
        )

        messages = format_messages(system_prompt, user_prompt)
        inputs.append(messages)

    return inputs


async def main(args):
    global DEBUG
    DEBUG = args.debug

    # Load data
    dataloader = HFDepthQALoader()
    if args.task_type == "prompt-pred":
        with open(args.input) as f:
            questions = json.load(f)
        _, nodes, node_to_q = dataloader.load_data(except_questions=True)
    else:
        questions, nodes, node_to_q = dataloader.load_data()
    print(f"Loaded {len(questions)} questions and {len(nodes)} nodes.")

    # Load model
    model = OpenAILLM(args.model_name)

    # Prepare inputs
    if args.task_type == "zero-shot":
        if DEBUG:
            questions = slice_data_dict(questions, start=0, end=5)
        inputs = prepare_inputs_zero_shot(questions)
    else:
        nodes = filter_data_dict(nodes, lambda node: node["depth"] > 1)
        if DEBUG:
            nodes = slice_data_dict(nodes, start=0, end=5)
        inputs = prepare_inputs_ctx(
            questions,
            nodes,
            node_to_q,
            use_gold=args.task_type == "prompt-gold",
        )
    show_random_inputs(inputs)

    if DEBUG:
        inputs = inputs[:5]

    # Inference
    predictions = await batch_completions_with_retries(model, inputs, args.batch_size)

    # Save results
    results = {}
    if args.task_type == "zero-shot":
        for idx, (qid, question_d) in enumerate(questions.items()):
            results[qid] = question_d
            results[qid].update({"predicted_answer": predictions[idx]})
    else:
        for idx, nodeid in enumerate(nodes.keys()):
            results[nodeid] = questions[node_to_q[nodeid]]
            results[nodeid].update({"predicted_answer": predictions[idx]})

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
        help="Name of model hosted in OpenAI",
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
        help="Output JSON file name. Will be saved under outputs/inference by default.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if output file exists.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    # Compute arguments
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    # Prompt arguments
    parser.add_argument(
        "--task_type",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "prompt-gold", "prompt-pred"],
        help="Task type for the model, which determines the input text.",
    )
    args = parser.parse_args()

    assert not (
        args.task_type == "prompt-pred" and not args.input.endswith(".json")
    ), "Input file for prompt-pred task should be a JSON file that contains zero-shot predictions."

    assert args.output_file.endswith(".json"), "Output file must be a JSON file."

    output_path = get_output_path(
        output_file=args.output_file, default_output_dir="../../outputs/inference"
    )
    assert not (
        os.path.exists(output_path) and not args.force_rerun
    ), f"Output file {output_path} already exists. Skipping inference."

    asyncio.run(main(args))

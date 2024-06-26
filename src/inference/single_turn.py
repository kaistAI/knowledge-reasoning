import json
import os
import warnings
from argparse import ArgumentParser
from typing import Dict, List

from tqdm import tqdm

from src.data import HFDepthQALoader, filter_data_dict, slice_data_dict
from src.model import VLLM
from src.utils import (
    SAMPLING_PARAMS,
    SYSTEM_PROMPT_CTX,
    SYSTEM_PROMPT_ZERO_SHOT,
    USER_PROMPT_TEMPLATE_CTX,
    USER_PROMPT_TEMPLATE_ZERO_SHOT,
    get_output_path,
    show_random_inputs,
)

DEBUG = False


# Model inference (Use offline batching)
def batch_completions(
    model,
    inputs: List[str],
    batch_size,
):
    batched_outputs = []

    # Adjust batch size to fit the number of inputs
    # VLLM supports adaptive batch size already
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
        batch_outputs = model.completions(
            batch_inputs, **SAMPLING_PARAMS, use_tqdm=True
        )
        batched_outputs.extend(batch_outputs)

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )

    for i, output in enumerate(batched_outputs):
        if output == "":
            print("Empty output")
            batched_outputs[i] = None

    if DEBUG:
        print("Checking the results")
        for output in batched_outputs[:5]:
            print(output)

    return batched_outputs


def apply_template_chat(system_prompt, user_prompt, tokenizer):
    if tokenizer.chat_template and "system" not in tokenizer.chat_template:
        messages = [
            {"role": "user", "content": system_prompt + "\n" + user_prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return (
        tokenizer.apply_chat_template(  # automatically format to default chat template
            messages, tokenize=False, add_generation_prompt=True
        )
    )


def prepare_inputs_zero_shot(questions: Dict[str, Dict], tokenizer) -> List[str]:
    inputs = []
    system_prompt = SYSTEM_PROMPT_ZERO_SHOT
    for question_d in questions.values():
        target_question = question_d["question"]
        user_prompt = USER_PROMPT_TEMPLATE_ZERO_SHOT.format(question=target_question)

        input_str = apply_template_chat(system_prompt, user_prompt, tokenizer)
        inputs.append(input_str)

    return inputs


def prepare_inputs_ctx(
    questions: Dict[str, Dict],
    nodes: Dict[str, Dict],
    node_to_q: Dict[str, str],
    tokenizer,
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

        input_str = apply_template_chat(system_prompt, user_prompt, tokenizer)
        inputs.append(input_str)

    return inputs


def main(args):
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
    model = VLLM(args.model_name, num_gpus=args.num_gpus)
    tokenizer = model.get_tokenizer()

    # Prepare inputs
    if args.task_type == "zero-shot":
        if DEBUG:
            questions = slice_data_dict(questions, start=0, end=5)
        inputs = prepare_inputs_zero_shot(questions, tokenizer)
    else:
        nodes = filter_data_dict(nodes, lambda node: node["depth"] > 1)
        if DEBUG:
            nodes = slice_data_dict(nodes, start=0, end=5)
        inputs = prepare_inputs_ctx(
            questions,
            nodes,
            node_to_q,
            tokenizer,
            use_gold=args.task_type == "prompt-gold",
        )
    show_random_inputs(inputs)

    if DEBUG:
        inputs = inputs[:5]

    # Inference
    predictions = batch_completions(model, inputs, args.batch_size)

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
    # Compute arguments
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use for inference. Note that we use bfloat16 if available and float16 otherwise.",
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

    main(args)

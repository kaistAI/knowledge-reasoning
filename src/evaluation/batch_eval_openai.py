import json
import os
import time
import warnings
from argparse import ArgumentParser
from typing import Dict, List

from src.model import OpenAIBatchClient
from src.utils import (
    SAMPLING_PARAMS_OPENAI,
    SYSTEM_PROMPT_EVAL,
    USER_PROMPT_TEMPLATE_EVAL,
    calculate_cost,
    format_messages,
    get_output_path,
    show_random_inputs,
)

from .output_parser import parse_judgment

DEBUG = False


def prepare_inputs(predictions: Dict[str, Dict], eval_model_name: str) -> List[Dict]:
    inputs = []
    system_prompt = SYSTEM_PROMPT_EVAL
    for id, record in predictions.items():  # id can be qid or nodeid
        instruction = record["question"].strip()
        reference_answer = record["answer"].strip()
        response = record["predicted_answer"].strip()
        user_prompt = USER_PROMPT_TEMPLATE_EVAL.format(
            instruction=instruction,
            reference_answer=reference_answer,
            response=response,
        )
        messages = format_messages(system_prompt, user_prompt)
        inputs.append(
            {
                "custom_id": id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": eval_model_name,
                    "messages": messages,
                    **SAMPLING_PARAMS_OPENAI,
                },
            }
        )
    return inputs


def prepare_input_file(prediction_file, eval_model_name):
    with open(prediction_file) as f:
        predictions = json.load(f)

    inputs = prepare_inputs(predictions, eval_model_name)
    show_random_inputs(inputs)
    if DEBUG:
        inputs = inputs[:5]

    batch_eval_input_file = prediction_file.replace(".json", "_batch_eval_input.jsonl")
    with open(batch_eval_input_file, "w") as f:
        for input in inputs:
            f.write(json.dumps(input) + "\n")
    return batch_eval_input_file


def prepare_output_file(outputs, prediction_file, output_file):
    with open(prediction_file) as f:
        predictions = json.load(f)

    input_token_cnt = 0
    output_token_cnt = 0

    batch_output_file = args.output_file.replace(".json", "_batch_eval_output.jsonl")
    batch_output_writer = open(batch_output_file, "w")
    for output in outputs.iter_lines():
        batch_output_writer.write(output + "\n")

        output = json.loads(output)
        custom_id = output["custom_id"]
        judgment = output["response"]["body"]["choices"][0]["message"]["content"]
        feedback, score = parse_judgment(judgment)
        predictions[custom_id].update({"feedback": feedback, "score": score})

        input_token_cnt += output["response"]["body"]["usage"]["prompt_tokens"]
        output_token_cnt += output["response"]["body"]["usage"]["completion_tokens"]
    batch_output_writer.close()

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    return input_token_cnt, output_token_cnt


def main_individual(args):
    global DEBUG
    DEBUG = args.debug

    client = OpenAIBatchClient()
    if args.mode == "create":
        batch_eval_input_file = prepare_input_file(
            args.prediction_file, args.eval_model_name
        )
        batch = client.create_batch(batch_eval_input_file, args.description)
        print(f"Batch created: {batch}")
        return batch.id

    elif args.mode == "check":
        status, batch_output_file_id = client.check_batch(args.batch_id)
        print(f"{args.batch_id} status: {status}")
        return status, batch_output_file_id

    elif args.mode == "cancel":
        client.cancel_batch(args.batch_id)

    elif args.mode == "list":
        client.list_batches()

    elif args.mode == "retrieve":
        outputs = client.retrieve_batch(args.batch_output_file_id)

        output_path = get_output_path(
            output_file=args.output_file, default_output_dir="../../outputs/evaluation"
        )
        args.output_file = output_path

        input_token_cnt, output_token_cnt = prepare_output_file(
            outputs, args.prediction_file, args.output_file
        )

        cost = calculate_cost(args.eval_model_name, input_token_cnt, output_token_cnt)
        print(f"Cost: {cost:.2f} USD")

    else:
        raise ValueError("Invalid mode")


def main_auto(args):
    client = OpenAIBatchClient()

    # Step 1: Create batch request
    batch_eval_input_file = prepare_input_file(
        args.prediction_file, args.eval_model_name
    )
    batch = client.create_batch(batch_eval_input_file, args.description)
    print(f"Batch created: {batch}")
    batch_id = batch.id

    # Step 2: Check status periodically
    while True:
        status, batch_output_file_id = client.check_batch(batch_id)
        print(f"Current status: {status}")

        if status == "completed":
            break
        elif status in ["failed", "cancelling", "cancelled", "expired"]:
            raise Exception(f"Batch failed with status: {status}")

        time.sleep(30)  # Wait for 30 seconds before checking again
    print(f"Batch completed. Output file ID: {batch_output_file_id}")

    # Step 3: Retrieve results
    outputs = client.retrieve_batch(batch_output_file_id)
    output_path = get_output_path(
        output_file=args.output_file, default_output_dir="../../outputs/evaluation"
    )
    args.output_file = output_path

    print(f"Retrieved results saved to {output_path}")
    input_token_cnt, output_token_cnt = prepare_output_file(
        outputs, args.prediction_file, args.output_file
    )

    cost = calculate_cost(args.eval_model_name, input_token_cnt, output_token_cnt)
    print(f"Cost: {cost:.2f} USD")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "create", "check", "cancel", "list", "retrieve"],
    )
    # Arguments for creating or retrieving a batch job
    parser.add_argument(
        "--eval_model_name",
        type=str,
        default="gpt-4-0125-preview",
        help="OpenAI evaluator model name",
    )
    # parser.add_argument("--dataset", type=str, default="kaist-ai/DepthQA", help="Dataset name in Hugging Face")  # TODO: Allow retrieving metadata from dataset in case the dataset changes
    parser.add_argument(
        "--prediction_file", type=str, help="Output file containing model predictions"
    )
    parser.add_argument("--description", type=str, help="Description of the batch job")
    parser.add_argument(
        "--batch_output_file_id", type=str, help="Output file ID of the batch job"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output JSON file to write the results. Unless the parent directory is specified, will be saved under outputs/inference by default. The same file but with a .jsonl extension will be created intermediately to store the batch job output.",
    )
    # Argument for checking or canceling a batch job
    parser.add_argument("--batch_id", type=str, help="Batch ID of submitted job")
    # Argument for debugging
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.mode == "auto":
        warnings.warn(
            "Creating a batch job in auto mode will overwrite the output file. Make sure you want to proceed."
        )
        assert args.prediction_file and args.prediction_file.endswith(
            ".json"
        ), "A valid JSON prediction file is required for creating a batch job"
        assert args.description, "Description is required for creating a batch job"
        assert args.output_file and args.output_file.endswith(
            ".json"
        ), "A valid JSON output file is required for retrieving a batch job"
        main_auto(args)
    else:
        if args.mode == "create":
            assert args.prediction_file and args.prediction_file.endswith(
                ".json"
            ), "A valid JSON prediction file is required for creating a batch job"
            assert args.description, "Description is required for creating a batch job"

        elif args.mode == "check" or args.mode == "cancel":
            assert args.batch_id and args.batch_id.startswith(
                "batch_"
            ), "A valid batch ID is required for checking a batch job"

        elif args.mode == "retrieve":
            assert args.batch_output_file_id and args.batch_output_file_id.startswith(
                "file-"
            ), "Output file ID is required for retrieving a batch job"
            assert args.output_file and args.output_file.endswith(
                ".json"
            ), "A valid JSON output file is required for retrieving a batch job"

        main_individual(args)

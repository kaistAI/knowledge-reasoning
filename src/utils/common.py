import os
import random

import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def show_random_inputs(inputs):
    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)
    return inputs


def get_output_path(output_file: str, default_output_dir: str):
    if os.path.dirname(output_file):
        return output_file
    else:
        return os.path.join(default_output_dir, output_file)


def format_messages(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def calculate_cost(model_name, input_token_cnt, output_token_cnt, batch_eval=True):
    def pricing_info(model):
        if model.startswith("gpt-4") and model.endswith("preview"):
            input_rate = 0.01
            output_rate = 0.03
        elif model == "gpt-4":
            input_rate = 0.03
            output_rate = 0.06
        elif model == "gpt-4-32k":
            input_rate = 0.06
            output_rate = 0.12
        elif model == "gpt-3.5-turbo-0125":
            input_rate = 0.0005
            output_rate = 0.0015
        elif model == "gpt-3.5-turbo-instruct":
            input_rate = 0.0015
            output_rate = 0.0020
        else:
            raise ValueError(f"Model {model} not supported.")
        return input_rate, output_rate

    input_rate, output_rate = pricing_info(model_name)
    if batch_eval:
        input_rate /= 2
        output_rate /= 2
    cost = input_rate * input_token_cnt / 1000 + output_rate * output_token_cnt / 1000
    return cost

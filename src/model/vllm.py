import os
from typing import List

import torch
from vllm import LLM, SamplingParams


class VLLM:
    def __init__(self, name, tokenizer_name=None, num_gpus=1):
        dtype = "float16"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"

        self.name = name

        max_model_len = None

        print(f"Loading {name}...")
        self.model = LLM(
            model=self.name,
            tokenizer=tokenizer_name,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            download_dir=os.getenv("HF_HOME"),
        )

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def completions(
        self,
        prompts: List[str],
        use_tqdm=False,
        **kwargs,
    ):
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)
        outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text.strip() for output in outputs]
        return outputs

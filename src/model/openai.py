import os

import openai
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm


class OpenAILLM:
    def __init__(
        self,
        name,
    ):
        self.name = name
        self.batch_size = 100
        self.requests_per_minute = 100
        self.limiter = AsyncLimiter(self.requests_per_minute, 60)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    async def get_completion_text_async(self, messages, **kwargs):
        async with self.limiter:
            try:
                # Assuming you have a session and client setup for OpenAI
                completion = await self.client.chat.completions.create(
                    model=self.name, messages=messages, **kwargs
                )
                content = completion.choices[0].message.content.strip()
                return content
            except openai.APIConnectionError as e:
                print("APIConnectionError: The server could not be reached")
                print(
                    e.__cause__
                )  # an underlying Exception, likely raised within httpx.
            except openai.RateLimitError as e:
                print(
                    "RateLimitError: A 429 status code was received; we should back off a bit."
                )
            except openai.APIStatusError as e:
                print("APIStatusError: Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                return ""  # , {}

    async def completions(
        self,
        messages,
        **kwargs,
    ):
        assert isinstance(messages, list)
        assert list(messages[0][0].keys()) == ["role", "content"]

        result_responses = []

        for start_idx in tqdm(
            range(0, len(messages), self.batch_size), desc="Processing batches"
        ):
            end_idx = start_idx + self.batch_size
            batch_prompts = messages[start_idx:end_idx]
            batch_responses = await tqdm_asyncio.gather(
                *[
                    self.get_completion_text_async(prompt, **kwargs)
                    for prompt in batch_prompts
                ]
            )
            result_responses.extend(batch_responses)

        return result_responses


class OpenAIBatchClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def create_batch(self, input_file, description=None):
        batch_input_file = self.client.files.create(
            file=open(input_file, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        return batch

    def cancel_batch(self, batch_id):
        self.client.batches.cancel(batch_id)

    def check_batch(self, batch_id):
        batch = self.client.batches.retrieve(batch_id)
        status = batch.status
        batch_output_file_id = batch.output_file_id
        return status, batch_output_file_id

    def list_batches(self):
        batches = self.client.batches.list()
        batches = sorted(batches, key=lambda x: x.created_at)
        for batch in batches:
            desc = batch.metadata.get("description", "") if batch.metadata else ""
            batch_id = batch.id
            status = batch.status
            if "cancel" in status:
                continue
            output_file_id = batch.output_file_id
            request_counts = batch.request_counts
            print("-" * 20)
            print(desc)
            print(f"\tBatch ID: {batch_id}")
            print(f"\tStatus: {status}")
            print(f"\tOutput file ID: {output_file_id}")
            print(f"\t{request_counts}")

    def retrieve_batch(self, batch_output_file_id):
        content = self.client.files.content(batch_output_file_id)
        return content


if __name__ == "__main__":
    print("Hello, World!")

    model = OpenAILLM("gpt-3.5-turbo")

    responses = model.completions(
        model="gpt-3.5-turbo",
        messages=[
            [{"role": "user", "content": "good morning? "}],
            [{"role": "user", "content": "what's the time? "}],
        ],
    )
    import pdb

    pdb.set_trace()

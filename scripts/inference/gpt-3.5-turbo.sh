export OPENAI_API_KEY=<YOUR_API_KEY>


python -m src.inference.single_turn_openai \
--model_name gpt-3.5-turbo-0125 \
--input kaist-ai/DepthQA \
--output_file outputs/inference/gpt-3.5-turbo_zero-shot.json \
--task_type zero-shot \

python -m src.inference.single_turn_openai \
--model_name gpt-3.5-turbo-0125 \
--input kaist-ai/DepthQA \
--output_file outputs/inference/gpt-3.5-turbo_prompt-gold.json \
--task_type prompt-gold \

python -m src.inference.single_turn_openai \
--model_name gpt-3.5-turbo-0125 \
--input outputs/inference/gpt-3.5-turbo_zero-shot.json \
--output_file outputs/inference/gpt-3.5-turbo_prompt-pred.json \
--task_type prompt-pred \

python -m src.inference.multi_turn_openai \
--model_name gpt-3.5-turbo-0125 \
--input kaist-ai/DepthQA \
--output_file outputs/inference/gpt-3.5-turbo_multi-turn.json 
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=<YOUR_HF_HOME>
NUM_GPUS=1

python -m src.inference.single_turn \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--input kaist-ai/DepthQA \
--output_file outputs/inference/llama3_8b_zero-shot.json \
--num_gpus $NUM_GPUS \
--task_type zero-shot \

python -m src.inference.single_turn \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--input kaist-ai/DepthQA \
--output_file outputs/inference/llama3_8b_prompt-gold.json \
--num_gpus $NUM_GPUS \
--task_type prompt-gold \

python -m src.inference.single_turn \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--input outputs/inference/llama3_8b_zero-shot.json \
--output_file outputs/inference/llama3_8b_prompt-pred.json \
--num_gpus $NUM_GPUS \
--task_type prompt-pred \

python -m src.inference.multi_turn \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--input kaist-ai/DepthQA \
--output_file outputs/inference/llama3_8b_multi-turn.json \
--num_gpus $NUM_GPUS 
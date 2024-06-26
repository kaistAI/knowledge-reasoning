export OPENAI_API_KEY=<YOUR_API_KEY>

# ------------------------- 1~3. Create, check status, retrieve results of batch request ------------------------ #
python -m src.evaluation.batch_eval_openai \
--mode auto \
--eval_model_name gpt-4-0125-preview \
--prediction_file outputs/inference/llama3_8b_prompt-gold.json \
--description llama3_8b_prompt-gold \
--output_file outputs/evaluation/llama3_8b_prompt-gold.json

# ---------------------- 4. Calculate evaluation metrics --------------------- #
# TODO: @miyoungko
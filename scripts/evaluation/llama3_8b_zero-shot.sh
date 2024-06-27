export OPENAI_API_KEY=<YOUR_API_KEY>

# ------------------------- 1. Create a batch request ------------------------ #
python -m src.evaluation.batch_eval_openai \
--mode create \
--eval_model_name gpt-4-0125-preview \
--prediction_file outputs/inference/llama3_8b_zero-shot.json \
--description llama3_8b_zero-shot
# Printed output:
# Batch(id='batch_ckmtDkk2bbpdEXp1KmM0vqk0', completion_window='24h', created_at=1719384093, endpoint='/v1/chat/completions', input_file_id='file-hbN7K9bwo42Gs6daIE2ivjQJ', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1719470493, failed_at=None, finalizing_at=None, in_progress_at=None, metadata={'description': 'llama3_8b_zero-shot'}, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))

# ------------------------- 2. Check the status of a batch request ------------------------ #
python -m src.evaluation.batch_eval_openai \
--mode check \
--batch_id batch_ckmtDkk2bbpdEXp1KmM0vqk0
# Printed output (in progress):
# Batch(id='batch_ckmtDkk2bbpdEXp1KmM0vqk0', completion_window='24h', created_at=1719384093, endpoint='/v1/chat/completions', input_file_id='file-hbN7K9bwo42Gs6daIE2ivjQJ', object='batch', status='in_progress', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1719470493, failed_at=None, finalizing_at=None, in_progress_at=1719384093, metadata={'description': 'llama3_8b_zero-shot'}, output_file_id=None, request_counts=BatchRequestCounts(completed=549, failed=0, total=1571))

python -m src.evaluation.batch_eval_openai \
--mode list
# Printed output (completed):
# llama3_8b_zero-shot
#         Batch ID: batch_ckmtDkk2bbpdEXp1KmM0vqk0
#         Status: completed
#         Output file ID: file-ffvIyZGGyWMuQoTSLRXgNb8X
#         BatchRequestCounts(completed=1571, failed=0, total=1571)

# ------------------------- 3. Retrieve the results of a batch request ------------------------ #
python -m src.evaluation.batch_eval_openai \
--mode retrieve \
--prediction_file outputs/inference/llama3_8b_zero-shot.json \
--batch_output_file_id file-ffvIyZGGyWMuQoTSLRXgNb8X \
--output_file outputs/evaluation/llama3_8b_zero-shot.json

# ---------------------- 4. Calculate evaluation metrics --------------------- #
python -m src.evaluation.metric_calculator \
--input outputs/evaluation/llama3_8b_zero-shot.json \
--output_file outputs/evaluation/llama3_8b_zero-shot_metric.json
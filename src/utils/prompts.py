from typing import Tuple

# Zero-shot inference
SYSTEM_PROMPT_ZERO_SHOT = (
    "You are a helpful, respectful and honest assistant. Answer the question."
)

USER_PROMPT_TEMPLATE_ZERO_SHOT = """
###Question: 
{question}

###Answer: """


# Prompt (Gold.) or Prompt (Pred.) inference
SYSTEM_PROMPT_CTX = "You are a helpful, respectful and honest assistant. Answer the question using the knowledge in given QA pairs."

USER_PROMPT_TEMPLATE_CTX = """
###QA pairs:
{qa_pairs}
###Question: 
{question}

###Answer: """


USER_PROMPT_TEMPLATE_LAST_TURN = """
Based on previous questions and responses, answer the given question:

###Question: 
{question}

###Answer: """


# Evaluation
SYSTEM_PROMPT_EVAL = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

USER_PROMPT_TEMPLATE_EVAL = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual?]
Score 1: The response is largely incorrect, inaccurate, and not factual. It demonstrates a fundamental misunderstanding of the query or topic, leading to irrelevant or completely erroneous information.
Score 2: The response is partially correct but contains significant inaccuracies or factual errors. It shows some understanding of the query or topic but fails to provide a fully accurate or reliable answer.
Score 3: The response is generally correct and factual but may include minor inaccuracies or lack of detail. It shows a good understanding of the query or topic but may miss some nuances or specific information.
Score 4: The response is mostly correct, accurate, and factual. It demonstrates a strong understanding of the query or topic, with only minimal inaccuracies or omissions that do not significantly detract from the overall quality of the response.
Score 5: The response is consistently correct, accurate, and entirely factual. It reflects a comprehensive understanding of the query or topic, providing detailed, precise, and fully reliable information without any inaccuracies or omissions.

###Feedback: """

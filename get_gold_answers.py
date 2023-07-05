# get_gold_answers.py
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-finetuned-squadv1",
    tokenizer="mrm8488/spanbert-finetuned-squadv1"
)


def get_gold_answer(qa_dict):
    for qa in qa_dict:
        for question in qa['questions']:
            qa['answers'].append(extract_gold_ans(qa['sentence'], question))


def extract_gold_ans(context, question):
    response = qa_pipeline({'context': context, 'question': question})
    return response

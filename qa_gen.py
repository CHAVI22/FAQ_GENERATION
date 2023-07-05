import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained(
    'ramsrigouthamg/t5_squad_v1')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
device = torch.device("cpu")
model = model.to(device)


def get_questions(qa_dict):
    for qa in qa_dict:
        for span in qa['spans']:
            text = "context: {} answer: {}".format(qa['sentence'], span)
            encoding = tokenizer.encode_plus(
                text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt")
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
            outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)

            dec = [tokenizer.decode(ids, skip_special_tokens=True)
                   for ids in outs]
            Question = dec[0].replace("question:", "")
            Question = Question.strip()
            qa['questions'].append(Question)

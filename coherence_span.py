# cohernece_span
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


def calculate_coherence_matrix(sents):
    if not isinstance(sents, list):
        raise ValueError("Input must be a list of sentences")
    v = sbert_model.encode(sents)
    sim_matrix = cosine_similarity(v)
    sim_matrix[np.arange(len(sents)), np.arange(
        len(sents))] = 1  # set diagonal to 1
    return sim_matrix.tolist()


def get_sentence_combination(sents, matrix):
    combs = []
    for i in range(1, len(matrix) - 1):
        before = sents[max(range(i), key=lambda k: matrix[i][k])]
        after = sents[max(range(i + 1, len(matrix)),
                          key=lambda k: matrix[i][k])]
        ans = before + sents[i] + after
        combs.append(ans)
    return combs


def get_summaries(combinations):
    return [tokenizer.decode(
        model.generate(tokenizer.encode("summarize: " + c, return_tensors="pt"),
                       num_beams=4, length_penalty=-1.0,
                       min_length=10, max_length=50, temperature=0.7, early_stopping=True)[0],
        skip_special_tokens=True) for c in combinations]


def get_coherent_sentences(x):
    m = calculate_coherence_matrix(x)
    c = get_sentence_combination(x, m)
    summs = [tokenizer.decode(model.generate(tokenizer.encode("summarize: " + c, return_tensors="pt"),
                                             num_beams=4, length_penalty=-1.0,
                                             min_length=10, max_length=50, temperature=0.7, early_stopping=True)[0],
                              skip_special_tokens=True) for c in c]
    return summs

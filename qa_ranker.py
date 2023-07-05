
# question_answer_ranker.py
from transformers import pipeline
from transformers.data.processors.squad import SquadExample
from sklearn.preprocessing import MinMaxScaler
qapair_rank_pipeline = pipeline(
    "question-answering",
    model="iarfmoose/bert-base-cased-qa-evaluator",
    tokenizer="mrm8488/spanbert-finetuned-squadv1"
)
# relevance_pipeline = pipeline("text2text-generation", model="t5-small")


def generate_qa_pairs(list_of_dictionaries, num_top_pairs=5, score_threshold=0.8):
    unsorted_list = []
    for item in list_of_dictionaries:
        # print(item)
        # print(len(item["questions"]))
        for i in range(len(item["questions"])):
            d = {}
            d["question"] = item["questions"][i]
            d["answer"] = item["answers"][i]["answer"]
            d["score"] = item["answers"][i]["score"]
            d["context"] = item["context"]
            # d["score"] = item["answers"][i]["score"]
            unsorted_list.append(d)
            # print(d)
    ranked_list = rank_qa_pairs(unsorted_list)
    unique_list = []
    qns = []
    for i in ranked_list:
        if i["question"] not in qns:
            unique_list.append(i)
            qns.append(i["question"])
    # filtered_list = [d for d in ranked_list if d["score"] >= score_threshold]
    sorted_list = sorted(unique_list, key=lambda k: k["score"], reverse=True)

    return sorted_list[:num_top_pairs]


def rank_qa_pairs(list_of_dictionaries):
    ranked_pairs = []
    scores = []
    for qa in list_of_dictionaries:
        question = qa["question"].replace("[CLS]", "").replace("[SEP]", "")
        context = qa["context"].replace("[CLS]", "").replace("[SEP]", "")
        example = {"context": context,

                   "question": question}
        # print(qapair_rank_pipeline(example))
        relevance_score = qapair_rank_pipeline(example)["score"]
        # print(relevance_score)
        scores.append(relevance_score)
    # Normalize the relevance scores between 0 and 1
    scaler = MinMaxScaler()
    relevance_scores = scaler.fit_transform(
        [[score] for score in scores]).flatten()
    for i, qa in enumerate(list_of_dictionaries):
        # print(i,qa)
        answer_score = qa["score"]
        final_score = 0.15 * relevance_scores[i] + 0.85 * answer_score

        qa_new = qa.copy()
        # print(qa_new)
        qa_new["score"] = final_score
        # print(qa_new)
        ranked_pairs.append(qa_new)
    # print(ranked_pairs)
    return ranked_pairs

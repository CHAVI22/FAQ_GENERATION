from flask import Flask, render_template, request, redirect
from get_span import extract_spans
from qa_gen import get_questions
from get_gold_answers import get_gold_answer
from qa_ranker import generate_qa_pairs

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def get_passage():
    if request.method == 'POST':
        context = request.form['context'].replace("\n", "")
        limit = int(request.form['limit'])
        qa_dict = extract_spans(context)
        if len(qa_dict) == 0:
            QAs = [{
                "question": "Invalid input",
                "answer": "Please enter a passage"
            }]
        else:

            get_questions(qa_dict)
            get_gold_answer(qa_dict)
            QAs = generate_qa_pairs(qa_dict, limit)
        # final_QA =[]
        # for qa in QAs:

        return render_template("result.html", QAs=QAs)
    return render_template("index.html")


@app.route('/result')
def display_result():
    return render_template("result.html",)


if __name__ == "__main__":
    app.run(port=4996)

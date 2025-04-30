
# from flask import Flask

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         user_query = request.form["query"]
#         result = ask_query(user_query)
#         return render_template("index.html", result=result, query=user_query)
#     return render_template("index.html", result=None)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000,debug=True)
from flask import Flask, render_template, request, jsonify
from iris_db import extract_query_entities, global_query, ask_query_rag,ask_query_graphrag, llm_answer_summarize, llm_answer_for_batch_graphrag,llm_answer_for_batch_rag
# from sentence_transformers import SentenceTransformer


# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     question = ""  # Default empty question
#     answer = None
    
#     if request.method == "POST":
#         question = request.form.get("question")  # Get input from form
#         answer = ask_query(question)  # Call the function from iris_db
    
#     return render_template("index.html", question=question, answer=answer)  # Pass question & answer

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    question = ""  # Default empty question
    answer1 = None
    answer2 = None

    if request.method == "POST":
        question = request.form.get("question")
        action = request.form.get("action")  # Which button was clicked

       
        answer1 = ask_query_graphrag(question, graphitems=100, vectoritems=0)
     
        answer2 = ask_query_rag(question, graphitems=0, vectoritems=100)

    return render_template("index1.html", question=question, answer1=answer1, answer2=answer2)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
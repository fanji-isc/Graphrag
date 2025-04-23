import os
import warnings
import ast
import sys
# from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import iris
from flask import Flask

# Suppress warnings
warnings.simplefilter("ignore")

# Load environment variables (uncomment this line if you have a .env file)
# load_dotenv()

# Set langchain variables
set_debug(False)
set_verbose(False)

# Configuration variables
max_papers = 30
data_path = "/home/jovyan/workspace/data/"

# Set the OpenAI API key (replace this with a secure method if needed)
os.environ["OPENAI_API_KEY"] = "sk-N84wsFAuOZNbYfq853q6D1XJCibzRfnjYk5txC700vT3BlbkFJRktCqqeKcDyKYvoxr07rEbcZ4D_o9IqkBHZ9ECRm8A"

# Model settings
gpt4omini = "gpt-4o-mini"
model = gpt4omini

# Iris connection setup
hostname = "iris"
port = 1972
namespace = "IRISAPP"
username = "SuperUser"
password = "SYS"

# Connect to Iris
connection = iris.connect("{:}:{:}/{:}".format(hostname, port, namespace), username, password)
irispy = iris.createIRIS(connection)

# File paths
docsfile = '/home/jovyan/workspace/CSV/Documents.csv'
relationsfile = '/home/jovyan/workspace/CSV/Relations2.csv'
entitiesfile = '/home/jovyan/workspace/CSV/Entities.csv'

# Hidden print suppressor
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Langchain-related query functions
def extract_query_entities(query):
    prompt_text = '''Based on the following example, extract entities from the user provided queries.
                    Below are a number of example queries and their extracted entities. Provide only the entities.
                    'How many wars was George Washington involved in' -> ['War', 'George Washington'].\n
                    'What are the relationships between the employees' -> ['relationships','employees].\n

                    For the following query, extract entities as in the above example.\n query: {content}'''

    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"content": query})
    return ast.literal_eval(response)

def global_query(query, items=50, vector_search=10, batch_size=10):
    with HiddenPrints():
        docs = irispy.classMethodValue("GraphKB.Query", "Search", query, items/2, items/2)
        docs = docs.split('\n\r\n')

    answers = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        response = llm_answer_for_batch(batch, query)
        answers.append(response)

    return llm_answer_summarize(query, answers)

def ask_query(query, items=10, method='local'):
    with HiddenPrints():
        docs = [irispy.classMethodValue("GraphKB.Query", "Search", query, items/2, items/2)]

    response = llm_answer_for_batch(docs, query, False)
    return response

def llm_answer_summarize(query, answers):
    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt_text = """You are an assistant for question-answering tasks. 
    Use the following answers to a query derived from analyzing batches of documents. Please compile these answers into one overall answer. If you don't know the answer, just say that you don't know. 
    Question: {question}  
    Previous Answers: {answers}
    Answer: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, 'answers': answers})
    return response

def llm_answer_for_batch(batch, query, cutoff=True):
    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt_text = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context from a graph database to answer the question. If you don't know the answer, just say that you don't know. 
    """ + (("Use three sentences maximum and keep the answer concise:") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {graph_context}
    Answer: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, 'graph_context': batch})
    return response

# Load data into Iris database
irispy.classMethodValue("GraphKB.Documents", "LoadData", docsfile)
irispy.classMethodValue("GraphKB.Entity", "LoadData", entitiesfile)
irispy.classMethodValue("GraphKB.Relations", "LoadData", relationsfile)

# Create embeddings (optional, depending on your needs)
irispy.classMethodValue("GraphKB.EntityEmbeddings", "DataToEmbeddings")

# Test query (replace with any query you need)
print(ask_query("Summarize the most significant papers in immunology", items=10))

# Create Flask app
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return f"Max Papers: {max_papers}, Data Path: {data_path}"

# if __name__ == '__main__':
#     app.run(debug=True)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form["query"]
        result = ask_query(user_query)
        return render_template("index.html", result=result, query=user_query)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
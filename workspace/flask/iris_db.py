


import os
import warnings
import ast
import sys
# from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

import iris  # Make sure you have this package installed in your container

# IRIS Database Credentials
hostname = "iris"  # Use the service name from docker-compose
port = 1972
namespace = "IRISAPP"
username = "_SYSTEM"
password = "SYS"

try:
    conn = iris.connect(f"{hostname}:{port}/{namespace}", username, password, sharedmemory=False)
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")

if conn:
    os.environ["OPENAI_API_KEY"] = "YOUR--KEY"

irispy = iris.createIRIS(conn)

gpt4omini = "gpt-4o-mini"

model = gpt4omini

# docsfile = '/app/CSV/papers100.csv'
# relationsfile = '/app/CSV/relations100.csv'
# entitiesfile = '/app/CSV/entities100.csv'

docsfile = '/home/irisowner/dev/CSV/papers100.csv'
relationsfile = '/home/irisowner/dev/CSV/relations100.csv'
entitiesfile = '/home/irisowner/dev/CSV/entities100.csv'

# Load data
irispy.classMethodValue("GraphKB.Documents","LoadData",docsfile)
irispy.classMethodValue("GraphKB.Entity","LoadData",entitiesfile)
irispy.classMethodValue("GraphKB.Relations","LoadData",relationsfile)


entitiesembeddingsfile = '/home/irisowner/dev/CSV/entities_embeddings.csv'
papersembeddingsfile = '/home/irisowner/dev/CSV/papers_embeddings.csv'

irispy.classMethodValue("GraphKB.DocumentsEmbeddings","LoadData",papersembeddingsfile)
irispy.classMethodValue("GraphKB.EntityEmbeddings","LoadData",entitiesembeddingsfile)

# entitiesembeddingsfile = '/home/irisowner/dev/CSV/entities_embeddings.csv'
# papersembeddingsfile = '/home/irisowner/dev/workspace/CSV/papers_embeddings.csv'
# docsfile = '/home/irisowner/dev/CSV/papers100.csv'
# relationsfile = '/home/irisowner/dev/CSV/relations100.csv'
# entitiesfile = '/home/irisowner/dev/CSV/entities100.csv'


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_embeddings(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([text])[0]
    rounded = np.round(embeddings, 7).tolist()
    return str(rounded)

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

# def ask_query(query, items=10, method='local'):
#     with HiddenPrints():
#         docs = [irispy.classMethodValue("GraphKB.Query", "Search", query, items/2, items/2)]

#     response = llm_answer_for_batch(docs, query, False)
#     return response
def ask_query(query, graphitems=100,vectoritems=0, method='local'):
    
    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)
    with HiddenPrints():
      docs = [irispy.classMethodValue("GraphKB.Query","Search",user_query_entity,user_query_embeddings,graphitems,vectoritems)]
        
    response = llm_answer_for_batch_graphrag(docs, query, False)
    return response

def ask_query(query, graphitems=0,vectoritems=100, method='local'):
    
    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)
    with HiddenPrints():
      docs = [irispy.classMethodValue("GraphKB.Query","Search",user_query_entity,user_query_embeddings,graphitems,vectoritems)]
        
    response = llm_answer_for_batch_rag(docs, query, False)
    return response

def llm_answer_summarize(query, answers):
    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt_text = """You are an assistant for question-answering tasks. 
    Use the following answers to a query derived from analyzing batches of documents. Please compile these answers into one overall answer. 
    Question: {question}  
    Previous Answers: {answers}
    Answer: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, 'answers': answers})
    return response

def llm_answer_for_batch_graphrag(batch, query, cutoff=True):
    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt_text = """You are an expert assistant for graph-based academic search. 
    You are given a graph context of academic papers, authors, abstract, and related information.
    Use the following pieces of retrieved context from a graph database to answer the question. 
    """ + (("keep the answer detailed, complete and with supporting references,list as much info as you can:") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {graph_context}
    Answer: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, 'graph_context': batch})
    # return response
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]

    return answer_lines

def llm_answer_for_batch_rag(batch, query, cutoff=True):
    llm = ChatOpenAI(temperature=0, model_name=model)
    prompt_text = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    """ + (("Use three sentences maximum and keep the answer concise. If you don't know, just say I don't know:") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {graph_context}
    Answer: 
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, 'graph_context': batch})
    # return response
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]

    return answer_lines

# irispy.classMethodValue("GraphKB.Documents", "LoadData", docsfile)
# irispy.classMethodValue("GraphKB.Entity", "LoadData", entitiesfile)
# irispy.classMethodValue("GraphKB.Relations", "LoadData", relationsfile)


# irispy.classMethodValue("GraphKB.DocumentsEmbeddings","LoadData",papersembeddingsfile)
# irispy.classMethodValue("GraphKB.EntityEmbeddings","LoadData",entitiesembeddingsfile)

# irispy.classMethodValue("GraphKB.EntityEmbeddings", "DataToEmbeddings")


# print(ask_query("Summarize the most significant papers in immunology", items=10))
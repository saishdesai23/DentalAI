"""
File: main.py
Author: Saish Desai
Date: 2023-07-19

Description: RAG pipeline for developing a chatbot solving quesries pertaining to dental problems
"""

# import libraries
import getpass
import os
import openai
import pprint
import streamlit as st
from typing import Annotated, Dict, TypedDict
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  
from langgraph.graph import END, StateGraph

# import custome modules
from loading import doc_load, doc_merging
from indexing import text_split, doc_indexing
from corrective_rag import *
from corrective_rag_with_hist import *
from rag_with_hist import *

# api keys
openai.api_key = os.environ["OPENAI_API_KEY"]   # https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

# langchainkey for langsmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"]= getpass.getpass()

# langchainkey for langsmith
LANGCHAIN_TRACING_V2= "true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT="dental_v1"

# input variables
run_local = "Yes"


# Document Loading and Merging
pdf_list = [os.path.join("data", pdf) for pdf in os.listdir("data")]
url_list = ["https://my.clevelandclinic.org/health/diseases/10946-cavities"]
data_path = pdf_list + url_list

doc_l = []
for path in data_path:
  if path.startswith("https"):
    doc_l.append(doc_load("webpage", path))
  elif path.endswith("pdf"):
    doc_l.append(doc_load("pdf", path))

merged_docs = doc_merging(doc_l)


# Document Splitting and indexing
doc_splits = text_split(merged_docs)
persist_directory = "vectorstore/"
doc_indexing('Chroma', doc_splits, persist_directory)

# Retriever
vectorstore = Chroma(persist_directory=persist_directory,
                                       embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Selecting RAG Option
rag_option_list = ['corrective_rag', 'rag_with_hist', 'self_rag'] # not added self rag as of
rag_option = 'corrective_rag'

if rag_option == 'corrective_rag':

    # Corrective Rag using Langgraph
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            keys: A dictionary where each key is a string.
        """
        keys: Dict[str, any]

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()

elif rag_option == 'rag_with_hist':
   rag_chain = rag_hist()


# Set the title using StreamLit
# https://www.geeksforgeeks.org/build-chatbot-webapp-with-langchain/
st.title(' Dental AI ')
input_text = "Hi, how may I help you today?"
chat_history = []
if input_text:
    question = st.text_input(input_text)
    print(question)
    if question:
        if rag_option == 'corrective_rag':
            # Run
            inputs = {
                "keys": {
                    "question": question,
                    "local": run_local,
                }
            }
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node
                    pprint.pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")

            # Final generation
            pprint.pprint(value["keys"]["generation"])
            ans = value["keys"]["generation"]
        elif rag_option == 'rag_with_hist':
            ans = rag_chain.invoke({"question": question, "chat_history": chat_history})
            print(ans)
            chat_history.extend([HumanMessage(content=question), ans])
        st.write(ans)


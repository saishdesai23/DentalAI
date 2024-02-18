"""
File: main.py
Author: Saish Desai
Date: 2023-07-19

Description: RAG pipeline for developing a chatbot
"""

import getpass
import os
import openai
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import warnings
warnings.filterwarnings("ignore")

from loading import doc_load, doc_merging
from indexing import text_split, doc_indexing
# openai key
openai.api_key = os.environ["OPENAI_API_KEY"]   # https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
# langchainkey for langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

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

# QA chain
qa_system_prompt = """You are a Dental assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question associated
with a dental problem. Start you frist answer by saying "DentalAI at
your service😊!If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}

Question: {question}

Helpful Answer:
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}")
    ]
)


# Initialize a RetrievalQA chain with the language model and vector database retriever
llm = ChatOpenAI(model_name= 'gpt-3.5-turbo', temperature=0)
rag_chain = (
              {"context": retriever , "question": RunnablePassthrough()}
               |qa_prompt
               |llm)

first_question = input("Ask me a question related to dental hygiene? :\n")
ai_msg = rag_chain.invoke(first_question)
print(ai_msg)
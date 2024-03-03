from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.merge import MergedDataLoader


def doc_load(file_type, file_path):
  """
  Function to load the documents needed to assist the Language Model in Answering Questions related to the topic
  """
  if file_type == 'webpage':
    doc_loader = WebBaseLoader(file_path)
  elif file_type == 'pdf':
    doc_loader = PyPDFLoader(file_path)
  return doc_loader

def doc_merging(doc_loader_list):
  """
  Merging documents from all sources
  """
  loader_all = MergedDataLoader(loaders=doc_loader_list)
  docs = loader_all.load()
  return docs
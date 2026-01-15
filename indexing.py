from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch


def text_split(merged_docs):
    """
    Splitting the merged documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=1000,
                                                chunk_overlap=200,
                                                add_start_index=True)
    all_splits = text_splitter.split_documents(merged_docs)
    return all_splits
    



def doc_indexing(vectorstore_type, doc_splits, persist_directory):
  """
  Choosing a vecstore for storing and indexing
  """

  if vectorstore_type == 'Chroma':
    # Chroma Vectorstore
    vectorstore = Chroma.from_documents(documents=doc_splits,
                                        embedding=OpenAIEmbeddings(),
                                        persist_directory = persist_directory)
  if vectorstore_type == 'Docarray':
    # Doc Array Vectorstore
    vectorstore = DocArrayInMemorySearch.from_documents(documents=doc_splits,
                                        embedding=OpenAIEmbeddings(),
                                        persist_directory = persist_directory)


    # vectorstore.persist()
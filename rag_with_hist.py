from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings 
from langchain_core.messages import AIMessage, HumanMessage

# Retriever
persist_directory = "vectorstore/"
vectorstore = Chroma(persist_directory=persist_directory,
                                       embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def rag_hist():

    # Step1: Initialize a language model with ChatOpenAI
    llm = ChatOpenAI(model_name= 'gpt-3.5-turbo', temperature=0)

    # Step 2: Create a prompt to support chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Print out the reformulated question.
    Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    # Step 3: Create a chain for contextualizing the question
    qa_system_prompt = """You are a Dental assistant for question-answering tasks. \
    Use the following pieces of retrieved context and chat history to answer the question. 
    Start you frist answer by saying "DentalAI at
    your service😊! If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    # Initialize a RetrievalQA chain with the language model and vector database retriever
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm
    )
    return rag_chain
if __name__ == "__main__":
        rag_chain = rag_hist()
        chat_history = []

        # asking the first question and saving it in the chat history
        first_question = "What is a cavity?"
        ai_msg = rag_chain.invoke({"question": first_question, "chat_history": chat_history})
        print(ai_msg)
        chat_history.extend([HumanMessage(content=first_question), ai_msg])

        print("***********************************************************************")
        # asking the second question and saving it in the chat history
        second_question = "What are its different types?"
        ai_msg = rag_chain.invoke({"question": second_question, "chat_history": chat_history})
        print(ai_msg)
        chat_history.extend([HumanMessage(content=second_question), ai_msg])

        print("***********************************************************************")
        # asking the third question based on the retrieved docutment and chat history
        third_question = "Explain one of the types in detail"
        final_ans = rag_chain.invoke({"question": third_question, "chat_history": chat_history})
        print(final_ans)
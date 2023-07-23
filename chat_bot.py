from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import textwrap
import pickle
from dotenv import find_dotenv, load_dotenv
import streamlit as st

# load environment variables
load_dotenv(find_dotenv())


def conversation_chain_retrieval(vectorstore):
    openai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    # llama2_7b = HuggingFaceHub(
    #     repo_id="meta-llama/Llama-2-7b-chat-hf",
    #     model_kwargs={"temperature": 0.2, "max_length": 512},
    # )

    # memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=openai, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def get_response_from_query(db, query, k=8):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 500 and k to 8 maximizes
    the number of tokens to analyze.
    """

    # find the most relevant documents
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # prompt templates
    template = """
        You are a helpful assistant that can answer questions about the documents: {docs}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    ## human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # llms
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    # llm = HuggingFaceHub(
    #     repo_id="meta-llama/Llama-2-13b-chat-hf",
    #     model_kwargs={"temperature": 0.2, "max_length": 512},
    #     verbose=True,
    # )
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


if __name__ == "__main__":
    # load data
    data_save_time = "2023-07-22_12-09-08"
    db = pickle.load(open(f"cache_data/db_{data_save_time}.pkl", "rb"))
    query = "What Weishan did in 2023?"
    response, docs = get_response_from_query(db, query)
    print(response)

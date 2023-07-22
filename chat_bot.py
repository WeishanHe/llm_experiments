from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import textwrap
from dotenv import find_dotenv, load_dotenv
import streamlit as st

# load environment variables
load_dotenv(find_dotenv())


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
        You are a helpful assistant that that can answer questions about the documents 
        based on the information in the documents: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    ## human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # memory
    query_memory = ConversationBufferMemory(
        input_key="query", memory_key="chat_history"
    )

    # llms
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, memory=query_memory)
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs, query_memory.buffer

import streamlit as st
import pickle
import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain.memory import ConversationBufferMemory

from chat_bot import get_response_from_query

# load environment variables
load_dotenv(find_dotenv())

# App framework
st.title("👾 Ask me Anything")
query = st.text_input("Type your question here")

# read data
data_save_time = "2023-07-22_12-09-08"
db = pickle.load(open(f"cache_data/db_{data_save_time}.pkl", "rb"))

# run model and show response to the screen
if query:
    response, docs, query_memory = get_response_from_query(db, query)
    st.write(response)

    # st.write('**Documents used:**')
    # for doc in docs:
    #     st.write(f'**{doc.title}**')
    #     st.write(doc.page_content)
    #     st.write('---')
    with st.expander("Chat History"):
        st.info(query_memory)

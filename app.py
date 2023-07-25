import streamlit as st
import pickle
from dotenv import find_dotenv, load_dotenv
import os

from chat_bot import get_response_from_query
from settings import data_save_time

# load environment variables
load_dotenv(find_dotenv())

st.set_page_config(page_title="Ask Weishan Anything", page_icon="🦥")
st.header("🦥 Ask me Anything about My Professional Life")
query = st.text_input("Type your question here, e.g. 'How is your Python skill?'")

# load database
db = pickle.load(open(f"cache_data/db_{data_save_time}.pkl", "rb"))


# run model and show response to the screen
if query:
    response, docs = get_response_from_query(db, query)
    st.write(response)

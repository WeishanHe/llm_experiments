import streamlit as st
from dotenv import load_dotenv
import pickle

from htmlTemplates import css, bot_template, user_template
from chat_bot import conversation_chain_retrieval


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask me Anything", page_icon="👾")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = "Introduce yourself"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask me anything about my professional life")
    user_question = st.text_input("Type your question here")
    if user_question:
        handle_userinput(user_question)
        with st.spinner("Processing"):
            # get vectorstore
            data_save_time = "2023-07-22_12-09-08"
            vectorstore = pickle.load(open(f"cache_data/db_{data_save_time}.pkl", "rb"))

            # create conversation chain
            st.session_state.conversation = conversation_chain_retrieval(vectorstore)


if __name__ == "__main__":
    main()

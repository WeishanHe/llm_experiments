# Ask Weishan Anything (about her professional life)

In this project, I applied LangChain to leverage the power of GPT3.5 to build a chatbot which can answer questions about my profesional life. The front-end interface is powered by Streamlit. You can ask questions like "How is your Python skill?", "Tell me about a time when you use data to generate meaningful insights", etc. The bot will answer your questions based on the materials that I prepared, including resume, project stories, etc.

## Model Building
LangChain is a powerful framework that enables us to develop our LLM-based applications in simple Python code. The whole development process for my chatbot involves the following steps:
1. Build Vectorstore - ingest.py
I first read the files that I want to use to build the database. Then I used LangChain's `RecursiveCharacterTextSplitter`
function to divide the raw text into chuncks. Following, I applied the `FAISS` library to convert the chunks into vectors and store them in a vectorstore. The vectorstore aka database is exported as a pickle file. I employed `OpenAIEmbeddings()` in this project but you can also take advantage of other embedding models available at HuggingFace. 

2. Build the chatbot - chatbot.py
In order to enable the chatbot to answer questions based on my past experience, I applied `similarity_search` between the database and the user input. The `similarity_search` function returns the top 8 most similar chunks. Then I used `GPT3.5` to generate the answer based on the top 8 chunks.

3. Build the app - app.py
I used Streamlit to build the front-end interface.

## Performance
The chatbot is able to answer questions based on my data. However, sometimes it may match the tasks that I did in comapy A to the questions about company B. 

## Next Steps
First, I will explore how to deploy




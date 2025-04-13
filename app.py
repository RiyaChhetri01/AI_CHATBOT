from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Inject CSS for dark theme
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #555;
        }
        .stApp {
            background-color: #000000;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user {
            background-color: #2c2f33;
            color: #00ffcc;
        }
        .bot {
            background-color: #1f1f1f;
            color: #ffff66;
        }
    </style>
""", unsafe_allow_html=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide response to the user."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.markdown("<h1 style='color:white;'>LangChain Demo with LLama3.2 API </h1>", unsafe_allow_html=True)
input_text = st.text_input("Search the topic you want")

# LangChain components
llm = Ollama(model="llama3.2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display response
if input_text:
    response = chain.invoke({'question': input_text})
    st.markdown(f"<div class='chat-bubble user'><b>You:</b> {input_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble bot'><b>Bot:</b> {response}</div>", unsafe_allow_html=True)

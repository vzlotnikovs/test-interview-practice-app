import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    st.title("Interview Practice App")
    st.header("This is a header") 
    st.subheader("This is a subheader")
    st.text_input("Question", key="question")
    OpenAI_api_key = os.getenv('OPENAI_API_KEY')
    if OpenAI_api_key is None:
        raise ValueError('OpenAI API Key environment variable is required.')
    client = OpenAI(api_key=OpenAI_api_key)

if __name__ == "__main__":
    main()

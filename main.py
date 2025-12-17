import streamlit as st
from dotenv import load_dotenv
import os
from interview_practice_app.LLM_manager_class import LLM_Manager

ACCEPTED_FILE_TYPES = ["txt", "pdf", "doc", "docx"]

def main():
    load_dotenv()
    llm_manager = LLM_Manager()
    st.title("Interview Practice App")
    st.header("This is a header") 
    st.subheader("This is a subheader")
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=ACCEPTED_FILE_TYPES)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)
    st.text_input("Question", key="question")
    st.button("Generate", on_click=llm_manager.generate_questions)

if __name__ == "__main__":
    main()

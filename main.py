import streamlit as st
from dotenv import load_dotenv
import chardet
from io import StringIO
from interview_practice_app.LLM_manager_class import LLM_Manager
from interview_practice_app.Interview_manager_class import Interview_Manager

ACCEPTED_FILE_TYPES = ["txt", "pdf", "doc", "docx"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
EXAMPLE_QUESTIONS = ["Which Python data type is mutable and stores key-value pairs?", "Which condition checks equal values in Python?"]

def init_state() -> None:
    """Initializes the Streamlit session state variables."""
    if "job_description" not in st.session_state:
        st.session_state.job_description = None
    if "step2_confirmed" not in st.session_state:
        st.session_state.step2_confirmed = False
    if "questions_generated" not in st.session_state:
        st.session_state.questions_generated = False
    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0

def main() -> None:
    """
    Main function to run the Interview Practice App.
    
    Sets up the Streamlit UI, handles user interactions for job description upload,
    question generation, and interview practice.
    """
    load_dotenv()
    llm_manager = LLM_Manager()
    interview_manager = Interview_Manager()

    init_state()

    st.title("Welcome to Interview Practice App!", width="content", text_alignment="center")
    st.header("Ace your interview practice in 4 easy steps.") 
    st.subheader("Step 1: Upload a job description or copy-paste it into the text box below.")

    uploaded_job_description = st.file_uploader(
        "Option 1: Choose a file to upload",
        accept_multiple_files=False,
        type=ACCEPTED_FILE_TYPES
    )
    copy_pasted_job_description = st.text_area(
        "Option 2: Copy-paste the job description here",
        key="job_description_input"
    )
    
    if uploaded_job_description is not None:
        raw_bytes = uploaded_job_description.getvalue()
        detected = chardet.detect(raw_bytes)
        encoding = detected['encoding'] or 'utf-8'
        stringio = StringIO(raw_bytes.decode(encoding, errors='replace'))
        st.session_state.job_description = stringio.read()
        st.success("Job description saved from file successfully.")
    elif st.button("Save", key="Step 1"):
        st.session_state.job_description = copy_pasted_job_description
        st.success("Job description saved from text box successfully.")

    if st.session_state.job_description:
        st.subheader("Step 2: Define the number, difficulty, and creativity of questions to be generated for your practice interview.")
        
        number_of_questions = st.number_input(
            "Number of questions (up to 5)",
            min_value=1,
            max_value=5,
            value=3,
            key="num_questions"
        )
        difficulty_level = st.selectbox(
            "Difficulty level",
            DIFFICULTY_LEVELS,
            key="difficulty"
        )
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="temperature"
        )
        
        def confirm_step2() -> None:
            """Callback to confirm Step 2 completion."""
            st.session_state.step2_confirmed = True
            st.success("Step 2 confirmed successfully.")
        
        st.button(
            "Confirm number, difficulty and creativity of questions",
            key="step2_button",
            on_click=confirm_step2
        )
        if st.session_state.step2_confirmed:
            st.success("Step 2 confirmed successfully.")
            st.subheader("Step 3: Generate questions using OpenAI API.")
            if st.button("Generate"):
                generated_questions = llm_manager.generate_questions(
                    st.session_state.job_description,
                    st.session_state.num_questions,
                    st.session_state.difficulty,
                    st.session_state.temperature
                )
                st.session_state.generated_questions = generated_questions
                st.success("Questions generated successfully.")
            
            if "questions_file_path" in st.session_state:
                st.subheader("Step 4: Practice answering the questions and get feedback from OpenAI.")
                st.write("Your answers and feedback will be automatically saved.")
                
                st.button("Start Practice Interview", key="start_practice", on_click=lambda: st.session_state.update({"practice_active": True}))

                if st.session_state.get("practice_active"):
                    interview_manager.practice_interview(st.session_state.questions_file_path, llm_manager)

if __name__ == "__main__":
    main()
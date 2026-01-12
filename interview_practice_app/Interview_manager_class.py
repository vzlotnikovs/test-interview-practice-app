import streamlit as st
from typing import List
from interview_practice_app.LLM_manager_class import Question, LLM_Manager

question_review_options = ["Accept the question as-is", "Modify the question and accept it with modifications", "Reject the question"]

class Interview_Manager:
    """
    Manages all interview-related interactions.
    """

    def __init__(self, list_of_questions: List[Question] | None = None) -> None:
        """
        Initializes a new Interview_Manager instance.

        Args:
            list_of_questions (List[Question] | None, optional): A list of initial questions. Defaults to None.
        """
        self._list_of_questions = list_of_questions or []

    def __str__(self) -> str:
        """
        Returns:
            str: A formatted string describing the manager.
        """
        return "Interview Manager for interview practice."

    def practice_interview(self, questions_file_path: str, llm_manager: LLM_Manager) -> None:
        """
        Allows the user to practice answering the questions and get feedback from LLM.

        Args:
            questions_file_path (str): Path to the JSON file containing generated questions.
            llm_manager (LLM_Manager): Instance of LLM_Manager to evaluate answers.
        """
        import json
        import os
        from datetime import datetime
        from interview_practice_app.LLM_manager_class import Question

        if "current_q_index" not in st.session_state:
            st.session_state.current_q_index = 0
        if "interview_results" not in st.session_state:
            st.session_state.interview_results = []
        if "feedback_received" not in st.session_state:
            st.session_state.feedback_received = False
        
        try:
            with open(questions_file_path, "r") as f:
                data = json.load(f)
                questions_data = data.get("questions", [])
                questions = [Question(**q) for q in questions_data]
        except Exception as e:
            st.error(f"Error loading questions: {e}")
            return

        if not questions:
            st.warning("No questions found in the file.")
            return

        total_questions = len(questions)
        current_index = st.session_state.current_q_index

        if current_index < total_questions:
            current_q = questions[current_index]
            
            progress = (current_index + 1) / total_questions
            st.progress(progress, text=f"Question {current_index + 1} of {total_questions}")

            st.subheader(f"Question: {current_q.question}")
            st.info(f"Category: {current_q.category} | Difficulty: {current_q.difficulty_level}")

            user_answer = st.text_area("Your Answer:", height=200, key=f"answer_{current_index}")

            def submit_answer() -> None:
                """
                Submits the user's answer for evaluation.

                Validates input, calls LLM for feedback, and updates session state.

                Raises:
                    ValidationError: If answer validation fails (caught and displayed as warning).
                """
                if user_answer.strip():
                    with st.spinner("Evaluating your answer..."):
                        feedback = llm_manager.evaluate_answer(current_q, user_answer)
                        
                        result_entry = {
                            "question": current_q.model_dump(),
                            "user_answer": user_answer,
                            "feedback": feedback.model_dump()
                        }
                        st.session_state.interview_results.append(result_entry)
                        st.session_state.feedback_received = True
                else:
                    st.warning("Please enter an answer before submitting.")

            def next_question() -> None:
                """
                Advances the session to the next question.

                Increments the question index and resets feedback state.
                """
                st.session_state.current_q_index += 1
                st.session_state.feedback_received = False
                st.rerun()

            if not st.session_state.feedback_received:
                st.button("Submit Answer", on_click=submit_answer)
            else:
                last_result = st.session_state.interview_results[-1]
                feedback_obj = last_result["feedback"]
                
                st.markdown("### Feedback")
                st.write(feedback_obj["feedback_text"])
                
                if current_index < total_questions - 1:
                    st.button("Next Question", on_click=next_question)
                else:
                    if st.button("Finish Interview"):
                       st.session_state.current_q_index += 1
                       st.rerun()

        else:
            st.success("Interview Completed!")
            
            if st.session_state.interview_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                
                first_q = st.session_state.interview_results[0]["question"]
                company = first_q.get("company_name", "practice")
                job = first_q.get("job_title", "interview")
                
                safe_company = "".join([c if c.isalnum() else "_" for c in company])
                safe_job = "".join([c if c.isalnum() else "_" for c in job])
                
                output_dir = "interview_practice_app/output"
                output_filename = f"results_{safe_company}_{safe_job}_{timestamp}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                if "results_saved" not in st.session_state:
                     with open(output_path, "w") as f:
                        json.dump({"session_results": st.session_state.interview_results}, f, indent=2)
                     st.session_state.results_saved = True
                     st.info(f"Results saved to: {output_path}")
                elif st.session_state.get("results_saved"):
                     st.info("Results already saved.")
            
            if st.button("Start New Session"):
                st.session_state.clear()
                st.rerun()

        
import streamlit as st
import os
import json
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from typing import List

OPENAI_MODEL = "gpt-5-nano"
LLM_PROMPTS_DIR = "interview_practice_app/LLM_prompts"
OUTPUT_DIR = "interview_practice_app/output"

class Question(BaseModel):
    company_name: str
    job_title: str
    question: str
    difficulty_level: str
    category: str
    answer_guide: str

class QuestionsList(BaseModel):
    questions: List[Question]


class Feedback(BaseModel):
    feedback_text: str

class LLM_Manager:
    """
    Manages all interactions with the LLM as part of the interview practice.

    The class has two main methods:
    - generate_questions: Generates a list of questions based on the job description provided by the user (copy-pasted or uploaded).
    - evaluate_answers: Evaluates the user's answers to the generated questions and provides feedback. Saves the user's answers and the feedback in a file.

    """
    def __init__(self, api_key: str | None = None, client: OpenAI | None = None) -> None:
        """
        Initializes a new LLM_Manager instance.

        Args:
            api_key (str): The API key to use for the OpenAI client.
        """
        load_dotenv()
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self._api_key is None:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable by updating the .env file.")
        self._client = client or OpenAI(api_key=self._api_key)

    def __str__(self) -> str:
        """
        Returns a string representation of the LLM_Manager.

        Returns:
            str: A formatted string containing LLM_Manager details.
        """
        return f"""
        LLM Manager for interview practice (configured for OpenAI).
        """

    @property
    def api_key(self) -> str:
        """str: The API key used for the OpenAI client."""
        return self._api_key

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )

    def _safe_api_call(self, instructions_text: str, input_text: str, response_format: type[BaseModel]) -> BaseModel:
        """
        Internal wrapper for API call with retries.

        Args:
            instructions_text (str): The system instructions for the LLM.
            input_text (str): The user input text.
            response_format (type[BaseModel]): The Pydantic model class to enforce structure.

        Returns:
            BaseModel: The parsed response instance of the specified model.

        Raises:
            ValueError: If the response cannot be parsed or is empty.
        """
        response = self._client.responses.parse(
            model=OPENAI_MODEL,
            instructions=instructions_text,
            input=input_text,
            temperature=1,
            text_format=response_format,
        )
        if not response.output_parsed:
            raise ValueError("OpenAI returned empty response - try again.")
        return response.output_parsed

    def generate_questions(
        self,
        job_description: str,
        number_of_questions: int,
        difficulty_level: str,
        output_path: str | None = None
    ) -> QuestionsList:
        """
        Generates a list of questions based on the job description provided by the user (copy-pasted or uploaded).

        Args:
            job_description (str): The job description provided by the user (copy-pasted or uploaded).
            number_of_questions (int): The number of questions to generate.
            difficulty_level (str): The difficulty of the questions to generate.
            output_path (str | None): The path to save the generated questions.

        Returns:
            QuestionsList: A list of questions generated based on the job description.
        """
        env = Environment(loader=FileSystemLoader(LLM_PROMPTS_DIR))
        template = env.get_template("input_prompt.txt")
        
        LLM_instructions = template.render(
            {
                "job_description": job_description, 
                "number_of_questions": number_of_questions, 
                "difficulty_level": difficulty_level,
            }
        )
        
        st.write("Generating question(s) using OpenAI Responses API - please wait...")
        questions_list = self._safe_api_call(
            instructions_text=LLM_instructions,
            input_text=job_description,
            response_format=QuestionsList
        )
        # Extract company/job from first question for filename (fallback to defaults)
        first_question = questions_list.questions[0] if questions_list.questions else None
        company_slug = first_question.company_name if first_question and first_question.company_name != "Undefined" else "unknown_company"
        job_slug = first_question.job_title if first_question and first_question.job_title != "Undefined" else "unknown_role"
        
        # SANITIZE: Remove invalid filename characters and limit length
        def sanitize_filename(name: str) -> str:
            """
            Sanitizes a string to be safe for use as a filename.

            Args:
                name (str): The original string (e.g., company or job title).

            Returns:
                str: A sanitized string with invalid characters removed or replaced.
            """
            # Remove/replace invalid characters: / \ : * ? " < > |
            name = re.sub(r'[<>:"/\\|?*]', '_', name)
            # Replace multiple spaces/slashes with single underscore
            name = re.sub(r'\s+[/\s]+', '_', name)
            # Trim and limit length (Windows max ~260 chars)
            name = name.strip().replace(' ', '_')[:50]
            return name if name else "unknown"

        company_filename = sanitize_filename(company_slug)
        job_filename = sanitize_filename(job_slug)
                
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            file_name = f"questions_{company_filename}_{job_filename}_{number_of_questions}_{difficulty_level}.json" 
            output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # Save path to session state for other modules to access
        if "questions_file_path" not in st.session_state:
            st.session_state.questions_file_path = output_path
        else:
            st.session_state.questions_file_path = output_path

        with open(output_path, "w") as f:
            json.dump(questions_list.model_dump(), f, indent=2)
        st.write("Questions have been successfully generated and saved.")
        return questions_list
    
    def evaluate_answer(self, question: Question, user_answer: str) -> Feedback:
        """
        Evaluates the user's answer to a question and provides feedback.

        Args:
            question (Question): The question object containing the question and answer guide.
            user_answer (str): The user's answer to the question.

        Returns:
            Feedback: The feedback object containing the feedback text.
        """
        env = Environment(loader=FileSystemLoader(LLM_PROMPTS_DIR))
        template = env.get_template("evaluation_prompt.txt")

        LLM_instructions = template.render(
            {
                "question": question.question,
                "answer_guide": question.answer_guide,
                "user_answer": user_answer
            }
        )
        
        feedback = self._safe_api_call(
            instructions_text=LLM_instructions,
            input_text=user_answer,
            response_format=Feedback
        )
        return feedback

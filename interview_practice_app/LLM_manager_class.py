import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

OPENAI_MODEL = "gpt-4.1-nano"

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

    def _safe_api_call(self: OpenAI, instructions: str, input_text: str) -> str:
        """Internal wrapper for API call with retries."""
        response = self.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=input_text,
            temperature=1,
        )
        if not response.output_text:
            raise ValueError("OpenAI returned empty response - try again.")
        return response.output_text

    def generate_questions(LLM_input: str, LLM_instructions: str) -> str:
        """
        Generates a list of questions based on the job description provided by the user (copy-pasted or uploaded).

        Args:
            job_description (str): The job description provided by the user (copy-pasted or uploaded).
            num_questions (int): The number of questions to generate.
            difficulty (str): The difficulty of the questions to generate.

        Returns:
            list[str]: A list of questions generated based on the job description.
        """
        print("Generating question(s) using OpenAI Responses API - please wait...")
        data = self._safe_api_call(LLM_instructions, LLM_input)
        return data
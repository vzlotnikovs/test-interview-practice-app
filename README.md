# Interview Practice App

Target audience: user preparing for a job interview.

## Workflow: 

1. User copy-pastes or uploads a job description.
2. User defines the number, difficulty, and creativity (temperature) of interview questions to be generated.
3. The app analyzes the job description and generates a list of questions that are likely to be asked in an interview for that job.
4. The app allows the user to practice answering the questions and get feedback from OpenAI.
5. The app allows the user to save the user's answers and the feedback in a file.

## Features

- **Job description analysis**: Analyze a pasted or uploaded job description and generate a list of questions that are likely to be asked in an interview for that job.
- **Question generation**: Generate a list of questions that are likely to be asked in an interview for that job. The number of questions and difficulty of questions are user-defined.
- **Question practice**: Practice answering the questions and get feedback from OpenAI.
- **Question saving**: Save the user's answers and the feedback in a file.

## Optional Medium Tasks Completed

- Added a text field to include the job description (for the role the user is applying for) and getting interview preparation for that particular position (RAG).
- Implemented two structured JSON output formats (QuestionsList and Feedback)

## Installation

### Prerequisites / Dependencies

- Python 3.13 or higher.
- An OpenAI API key. 
- See pyproject.toml for full list of dependencies.

### Steps

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install .
   ```

   Or if you are using `uv`:

   ```bash
   uv sync
   ```

3. Rename .env.example to .env and fill with your OpenAI API key.

## Usage

To start the application, run:

```bash
streamlit run main.py
```

Follow the on-screen menu to navigate through the different modes:

1. **Step 1**: Upload a job description or copy-paste it into the text box.
2. **Step 2**: Define the number, difficulty, and creativity (temperature) of questions to be generated.
3. **Step 3**: Generate questions using OpenAI API.
4. **Step 4**: Practice answering the questions and get feedback from OpenAI.

## Running Tests

To run the unit tests, execute:

```bash
pytest tests
```

## Type Checking

To run the type checking, execute:

```bash
mypy .
```

## Code Formatting

```bash
ruff format .
```

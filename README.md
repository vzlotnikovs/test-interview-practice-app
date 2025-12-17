# Interview Practice App

Target audience: user preparing for a job interview.

Workflow: 

1. User pastes or uploads a job description.
2. User defines the number and difficulty of interview questions to be generated.
3. The app analyzes the job description and generates a list of questions that are likely to be asked in an interview for that job.
4. The app allows the user to attempt to answer each question and uses OpenAI API to evaluate the answer and provide feedback.
5. The app saves the user's answers and the feedback in a file.
6. The app allows the user to view the user's answers and the feedback.

## Features

- **Job description analysis**: Analyze a pasted or uploaded job description and generate a list of questions that are likely to be asked in an interview for that job.
- **Question generation**: Generate a list of questions that are likely to be asked in an interview for that job. The number of questions and difficulty of questions are user-defined.
- **Question evaluation**: Evaluate the user's answers and provide feedback.
- **Question saving**: Save the user's answers and the feedback in a file.
- **Question viewing**: View the user's answers and the feedback.

## Installation

### Prerequisites / Dependencies

- Python 3.14 or higher.
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
python main.py
```

Follow the on-screen menu to navigate through the different modes:

1. **Placeholder Feature 1**: Placeholder descrtiption.
2. **Placeholder Feature 2**: Placeholder descrtiption.
3. **Placeholder Feature 3**: Placeholder descrtiption.
4. **Exit**: Close the application.

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


import os
import re

from dotenv import load_dotenv
from google import genai


_MODEL = "gemini-3.5-flash-lite"


def _initial_prompt(task: str) -> str:
    return f"""Write only executable Python code for this task:
{task}

If the task describes an initial bug, preserve that bug and do not catch or proactively repair it. Do not use markdown fences. Keep the solution concise."""


def _repair_prompt(task: str, previous_code: str, error: str) -> str:
    return f"""Repair the failing Python program below.

This is a repair iteration. Ignore any task instruction that asks you to preserve or recreate a bug. Fix the reported error and return code that runs successfully.

Task:
{task}

Previous code:
{previous_code}

Execution error:
{error}

Return only the complete corrected executable Python code. Do not use markdown fences."""


def _strip_fences(response: str) -> str:
    return re.sub(r"^```(?:python)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)


def generate_code(task: str, previous_code: str = "", error: str = "") -> str:
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = _repair_prompt(task, previous_code, error) if previous_code and error else _initial_prompt(task)
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config={"max_output_tokens": 512, "temperature": 0.2},
    )
    return _strip_fences(response.text)
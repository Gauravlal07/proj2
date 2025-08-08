import re

def preprocess_prompt(task: str) -> str:
    """
    Detect and rewrite known prompt patterns to avoid LLM copying example answers.
    """

    # Detect if this is the highest-grossing films Wikipedia question
    if "highest grossing films" in task.lower() and "wikipedia" in task.lower():
        # Remove any JSON array examples from the prompt
        task = re.sub(r"\[[^\]]+\]", "", task)

        # Append scraping instructions
        extra = """
Important:
- You must fetch the live Wikipedia page at https://en.wikipedia.org/wiki/List_of_highest-grossing_films
- Parse the table data at runtime, do not use hardcoded values.
- Output only the JSON array in the requested order, nothing else.
- All numeric values must be computed from the scraped data.
"""
        task = task.strip() + "\n" + extra

    return task

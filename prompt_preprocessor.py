import re

def remove_example_output(text: str) -> str:
    """
    Remove code-blocked JSON examples and trailing example arrays from the task text.
    """
    if not text:
        return text
    # Remove ```json ... ``` blocks
    text = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
    # Remove any triple-backtick code fences that are JSON-like
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove trailing standalone JSON array examples e.g. [2, "Titanic", ...]
    text = re.sub(r"\n?\s*\[[^\]]+\]\s*$", "", text, flags=re.DOTALL)
    return text.strip()

def preprocess_prompt(task: str) -> str:
    """
    If the task appears to ask for the 'highest-grossing films' scrape,
    append strong runtime instructions to ensure live scraping and no example copying.
    Otherwise, only return the cleaned task.
    """
    t = remove_example_output(task)

    if "highest grossing films" in t.lower() and "wikipedia" in t.lower():
        extra_instructions = (
            "\n\nIMPORTANT (for the assistant):\n"
            "- You MUST fetch the live page at https://en.wikipedia.org/wiki/List_of_highest-grossing_films.\n"
            "- Parse the relevant table at runtime and compute answers from the live data; do NOT guess or hardcode.\n"
            "- Remove any example outputs; return ONLY the JSON array answer required by the user.\n"
        )
        t = t.strip() + extra_instructions

    return t

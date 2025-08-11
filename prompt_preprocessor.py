# prompt_preprocessor.py
import re

def remove_example_output(text: str) -> str:
    """
    Remove code-blocked JSON examples, trailing example arrays, or
    any standalone JSON-like array/object that appears to be an example.
    Keeps actual task text intact where possible.
    """
    if not text:
        return text
    # 1) remove fenced code blocks (```...```) of any language
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2) remove inline JSON blocks starting with ```json ... ``` or ```json\n ... ```
    text = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)

    # 3) remove standalone lines or trailing JSON arrays/objects (heuristic)
    #    Remove any large literal JSON array/object at the end of the text (over 10 chars).
    text = re.sub(r"\n\s*[\{\[][\s\S]{10,}\}\]\s*$", "", text)

    # 4) remove any standalone single-line array examples like: [1, "Titanic", ...]
    #    This matches when the line starts with optional whitespace then '[' and closes with ']'
    text = re.sub(r"(?m)^\s*\[[^\]]{0,1000}\]\s*$", "", text)

    # 5) collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def preprocess_prompt(task: str) -> str:
    """
    If the task is about the highest-grossing films wiki page, append hard runtime instructions.
    Otherwise return cleaned task.
    """
    t = remove_example_output(task)

    if "highest grossing films" in t.lower() and "wikipedia" in t.lower():
        extra_instructions = (
            "\n\nIMPORTANT (for the assistant):\n"
            "- Fetch the live page at https://en.wikipedia.org/wiki/List_of_highest-grossing_films.\n"
            "- Parse the relevant table at runtime and compute answers from the live data; do NOT guess or hardcode.\n"
            "- Return ONLY a JSON array of four elements: [int, string, float, string].\n"
            "- Do not include any example arrays or extra text in the output.\n"
            "- If plotting, include a PNG data URI prefixed exactly with 'data:image/png;base64,'.\n"
            "- If scraping fails, return [\"Error\", \"N/A\", 0.0, \"\"].\n"
        )
        t = t.strip() + extra_instructions

    return t

# prompt_preprocessor.py
import re
import json
from typing import Tuple, Dict

def remove_example_output(text: str) -> str:
    if not text:
        return text
    # remove ```json ... ``` and other fenced code blocks
    text = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # remove trailing standalone JSON arrays like [2, "Titanic", ...]
    text = re.sub(r"\n?\s*\[[^\]]+\]\s*$", "", text, flags=re.DOTALL)
    return text.strip()

def parse_numeric_thresholds(text: str) -> Dict[str, int]:
    """
    Returns a dict with parsed numeric params (in integers, raw dollars).
    Keys: threshold, year_cutoff, over_threshold (if present)
    """
    params = {"threshold": None, "year_cutoff": None, "over_threshold": None}

    # find "$X bn" or "$X billion"
    m = re.search(r"\$\s*([\d,.]+)\s*(bn|billion)", text, flags=re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        if m.group(2).lower().startswith("b"):
            params["threshold"] = int(val * 1_000_000_000)

    # find "before YYYY"
    m2 = re.search(r"before\s+(\d{4})", text, flags=re.I)
    if m2:
        params["year_cutoff"] = int(m2.group(1))

    # find "over $X bn"
    m3 = re.search(r"over\s+\$\s*([\d,.]+)\s*(bn|billion)", text, flags=re.I)
    if m3:
        val = float(m3.group(1).replace(",", ""))
        params["over_threshold"] = int(val * 1_000_000_000)

    return params

def preprocess_prompt(task: str) -> Tuple[str, Dict[str, int], bool]:
    """
    Returns (cleaned_task, params, is_highest_grossing_question)
    cleaned_task: text with example outputs removed
    params: parsed numeric params (may be None)
    is_highest_grossing_question: True when we detect the specific Wikipedia scraping task
    """
    cleaned = remove_example_output(task)
    params = parse_numeric_thresholds(cleaned)
    is_target = ("highest grossing films" in cleaned.lower() and "wikipedia" in cleaned.lower())
    return cleaned, params, is_target

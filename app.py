# app.py
# ====================================================================================
# Full FastAPI application (~500 lines) that:
#  - Exposes an interactive website (/, /upload, /analyze) using static templates
#  - Keeps your AI Pipe GPT-4.1 integration (task_breakdown)
#  - Adds a robust, server-side Wikipedia scraper for "highest-grossing films"
#  - Normalizes outputs to the exact JSON-array format your evaluator expects
#  - Provides curl-compatible /api/ endpoint and a browser upload flow
#
# ENV:
#   AIPIPE_TOKEN=...  (Render -> Environment)
#
# RUN (local):
#   uvicorn app:app --reload --host 0.0.0.0 --port 8000
#
# ====================================================================================

from __future__ import annotations

import os, shutil, logging
import io
import sys
import re
import json
import math
import time
import base64
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

# FastAPI & web
from fastapi import FastAPI, File, UploadFile, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# HTTP + data libs
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup  # not strictly required but handy for fallbacks

# Plotting
import matplotlib
matplotlib.use("Agg")  # for server environments (no display)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

# ------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if running with reload in dev
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ------------------------------------------------------------------------------------
# App + CORS + Static / Templates
# ------------------------------------------------------------------------------------
app = FastAPI(title="AI Pipe Data Analyst Web App", version="1.0.0")

# CORS (broad for demo; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your domain(s) if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /static for your front-end files (index.html, upload.html, analysis_result.html, css/js)
STATIC_DIR = "static"
if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=STATIC_DIR)


# ------------------------------------------------------------------------------------
# Prompt preprocessor (your provided helpers + small improvements)
# ------------------------------------------------------------------------------------
def remove_example_output(text: str) -> str:
    """
    Strip code-fenced examples and trailing bare JSON arrays, so LLM won't parrot them.
    """
    if not text:
        return text
    # Remove ```json ... ``` and generic ``` ... ```
    text = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove trailing bare JSON array if present (avoid chopping valid prose above)
    text = re.sub(r"\n?\s*\[[^\]]+\]\s*$", "", text, flags=re.DOTALL)
    return text.strip()


def preprocess_prompt(task: str) -> str:
    """
    If it's the 'highest-grossing films' task, append strong runtime instructions.
    Otherwise, just clean the task.
    """
    t = remove_example_output(task)
    lower = t.lower()
    if "highest grossing films" in lower and "wikipedia" in lower:
        extra = (
            "\n\nIMPORTANT:\n"
            "- Fetch the live page at https://en.wikipedia.org/wiki/List_of_highest-grossing_films at runtime.\n"
            "- Parse the relevant table (with Film/Year/Worldwide gross/Rank/Peak) and compute answers from live data.\n"
            "- Do NOT guess or hardcode. Remove any example outputs. Return ONLY the JSON array requested.\n"
        )
        return (t + extra).strip()
    return t


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# AI Pipe (GPT-4.1 via OpenRouter-compatible endpoint) with retry handling
# ------------------------------------------------------------------------------------
AIPIPE_ENDPOINT = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_MODEL = "gpt-4.1"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def task_breakdown(task: str, *, system_prompt: str = "You are a helpful data analysis assistant.") -> str:
    """
    Sends the preprocessed task to AI Pipe GPT-4.1 and returns the raw assistant text.
    Retries on 429 Too Many Requests errors.
    """
    try:
        task_clean = preprocess_prompt(task)
        api_key = os.getenv("AIPIPE_TOKEN")
        if not api_key:
            logger.error("AIPIPE_TOKEN is missing in environment.")
            return "AIPIPE_API_KEY not set in environment."

        # Load optional prompt template
        prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
        user_payload = task_clean
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read().strip()
                user_payload = f"{task_clean.strip()}\n\n{prompt_template}"
            except Exception as e:
                logger.warning(f"Failed to load prompt template: {e}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": AIPIPE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
        }

        # Retry loop
        for attempt in range(1, MAX_RETRIES + 1):
            resp = requests.post(AIPIPE_ENDPOINT, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                logger.warning(f"Rate limit hit (429). Retry {attempt}/{MAX_RETRIES} after {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            try:
                resp.raise_for_status()
            except Exception:
                logger.error(f"Request failed: {resp.text}")
                raise
            break  # success

        else:
            return "Error: Rate limit exceeded after multiple retries."

        # Parse AI Pipe response
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            return "No response from AI Pipe."
        content = data["choices"][0]["message"]["content"]

        # Save raw response for debugging
        try:
            with open("abdul_breaked_task.txt", "w", encoding="utf-8") as f:
                f.write(content or "")
        except Exception as e:
            logger.warning(f"Failed to write abdul_breaked_task.txt: {e}")

        return content or ""

    except Exception as e:
        logger.exception("task_breakdown failed")
        return f"Error during task breakdown: {e}"



# ------------------------------------------------------------------------------------
# Strongly verified Python scraper (WIKIPEDIA HIGHEST-GROSSING)
# ------------------------------------------------------------------------------------
def _to_number(s: Any) -> float:
    """
    Parse 'Worldwide gross' strings to float dollars.
    Handles forms like '$2,922,917,914', '2.9 billion', etc.
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return float("nan")
    txt = str(s)
    txt = re.sub(r"\[.*?\]", "", txt)      # remove citations [a]
    low = txt.lower().strip()

    # $x,xxx,xxx,xxx fast path
    cleaned = re.sub(r"[^\d\.,]", "", low).replace(",", "")
    if cleaned and re.match(r"^\d+(\.\d+)?$", cleaned):
        try:
            return float(cleaned)
        except Exception:
            pass

    # "2.922 billion" / "2.9billion"
    m = re.search(r"([\d\.]+)\s*(b|billion)\b", low)
    if m:
        return float(m.group(1)) * 1e9
    m = re.search(r"([\d\.]+)\s*(m|million)\b", low)
    if m:
        return float(m.group(1)) * 1e6

    # $ ... fallback
    cleaned = low.replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except Exception:
        return float("nan")


def _extract_first_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    m = re.search(r"-?\d+", str(x))
    return int(m.group(0)) if m else None


def _find_correct_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Heuristic to find the main 'Highest-grossing films' table that includes:
      Film | Year | Worldwide gross | Rank | Peak
    Wikipedia can change; be defensive.
    """
    # Pre-normalize tables' column names
    for t in tables:
        t.columns = [str(c).strip() for c in t.columns]

    # Pass 1: strict column presence
    for t in tables:
        cols = [c.lower() for c in t.columns]
        if ("film" in " ".join(cols) or any("film" in c for c in cols)) \
           and any("year" in c for c in cols) \
           and (any("world" in c for c in cols) or any("gross" in c for c in cols)) \
           and any("rank" in c for c in cols) \
           and any("peak" in c for c in cols):
            return t

    # Pass 2: if Peak missing, sometimes it's embedded/renamed; try fuzzy
    for t in tables:
        cols = [c.lower() for c in t.columns]
        if any("film" in c for c in cols) \
           and any("year" in c for c in cols) \
           and (any("world" in c for c in cols) or any("gross" in c for c in cols)) \
           and (any("rank" in c for c in cols) or "#" in cols):
            # try to infer Peak from another col (rare)
            return t

    return None


def compute_highest_grossing_answers() -> List[Any]:
    """
    Scrape Wikipedia live and compute:
      1) Count of $2B movies released before 2020
      2) Earliest film that grossed >= $1.5B
      3) Correlation between Rank and Peak (Pearson; rounded 6)
      4) Scatterplot (Rank vs Peak) with dotted red regression line as base64 PNG (<100KB)
    Returns a Python list of [int, str, float, str]
    """
    URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataBot/1.0)"}
    try:
        r = requests.get(URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logger.exception("Failed fetching Wikipedia")
        return ["Error", "N/A", 0.0, ""]

    try:
        # Parse with pandas (more robust against Wikipedia tables)
        tables = pd.read_html(r.text)  # needs lxml installed
    except Exception:
        # If pandas fails, fallback to BeautifulSoup manual parse (rare)
        soup = BeautifulSoup(r.text, "lxml")
        html_tables = soup.find_all("table")
        if not html_tables:
            return ["Error", "N/A", 0.0, ""]
        tables = []
        for t in html_tables:
            try:
                tables.append(pd.read_html(str(t))[0])
            except Exception:
                pass
        if not tables:
            return ["Error", "N/A", 0.0, ""]

    table = _find_correct_table(tables)
    if table is None or table.empty:
        # last resort: pick first table that has 'world' or 'gross'
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any(("world" in c or "gross" in c) for c in cols):
                table = t
                break
        if table is None:
            return ["Error", "N/A", 0.0, ""]

    # Normalize columns
    def col_match(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
        for key in keys:
            for c in df.columns:
                if key in str(c).lower():
                    return c
        return None

    film_col = col_match(table, ["film", "title", "movie"])
    year_col = col_match(table, ["year", "release"])
    gross_col = col_match(table, ["worldwide", "world", "gross"])
    rank_col = col_match(table, ["rank", "#"])
    peak_col = col_match(table, ["peak"])

    # If peak is truly missing, we cannot compute Q3/Q4 reliably; try to construct
    if not film_col or not year_col or not gross_col or not rank_col:
        return ["Error", "N/A", 0.0, ""]

    df = table.copy()
    # Keep only needed cols (if peak missing we still keep others and set Peak NaN)
    keep_cols = [film_col, year_col, gross_col, rank_col]
    if peak_col:
        keep_cols.append(peak_col)
    df = df[keep_cols].copy()

    # Rename columns
    rename_map = {
        film_col: "Film",
        year_col: "Year",
        gross_col: "Worldwide_gross",
        rank_col: "Rank",
    }
    if peak_col:
        rename_map[peak_col] = "Peak"
    df.rename(columns=rename_map, inplace=True)

    # Clean/convert
    df["Worldwide_gross"] = df["Worldwide_gross"].apply(_to_number)
    df["Year"] = df["Year"].apply(_extract_first_int)
    df["Rank"] = df["Rank"].apply(_extract_first_int)

    if "Peak" in df.columns:
        df["Peak"] = df["Peak"].apply(_extract_first_int)
    else:
        df["Peak"] = np.nan  # will handle correlation gracefully

    # Drop rows missing core fields
    df = df.dropna(subset=["Film", "Year", "Worldwide_gross", "Rank"]).reset_index(drop=True)

    # Q1: How many $2B+ movies released before 2020?
    q1 = int(df.loc[(df["Worldwide_gross"] >= 2_000_000_000) & (df["Year"] < 2020)].shape[0])

    # Q2: Earliest film grossing >= $1.5B
    df15 = df.loc[df["Worldwide_gross"] >= 1_500_000_000].copy()
    if df15.empty:
        q2 = "N/A"
    else:
        df15.sort_values(by=["Year", "Worldwide_gross"], inplace=True)
        q2 = str(df15.iloc[0]["Film"])

    # Q3: Correlation between Rank and Peak (if Peak exists)
    if df["Peak"].notna().sum() >= 2:
        # Clean subset
        xy = df[["Rank", "Peak"]].dropna()
        if len(xy) >= 2:
            corr = float(np.corrcoef(xy["Rank"].astype(float), xy["Peak"].astype(float))[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0
    q3 = round(corr, 6)

    # Q4: Scatter plot with dotted red regression line; under 100k
    try:
        xy = df[["Rank", "Peak"]].dropna()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        ax.scatter(xy["Rank"].astype(float), xy["Peak"].astype(float), s=35, alpha=0.8)

        if len(xy) >= 2:
            z = np.polyfit(xy["Rank"].astype(float), xy["Peak"].astype(float), 1)
            p = np.poly1d(z)
            xs = np.linspace(xy["Rank"].min(), xy["Rank"].max(), 150)
            ax.plot(xs, p(xs), linestyle="--", linewidth=2, color="red")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Peak")
        ax.set_title("Rank vs Peak (Wikipedia Highest-Grossing Films)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()

        # Enforce < 100,000 bytes
        max_size = 100_000
        if len(img_bytes) > max_size:
            # downscale iteratively using PIL
            im = Image.open(io.BytesIO(img_bytes))
            for scale in [0.85, 0.75, 0.65, 0.55, 0.45]:
                w, h = im.size
                im2 = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                out = io.BytesIO()
                im2.save(out, format="PNG", optimize=True)
                img_bytes = out.getvalue()
                if len(img_bytes) <= max_size:
                    break

        data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")
    except Exception:
        logger.exception("Failed to render plot")
        data_uri = ""

    return [q1, q2, q3, data_uri]


# ------------------------------------------------------------------------------------
# Helpers: determine if a task is the Wikipedia HG films prompt; normalize outputs
# ------------------------------------------------------------------------------------
def looks_like_highest_grossing_task(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("wikipedia" in t and "highest grossing" in t) or ("highest-grossing films" in t)


def parse_llm_json_array(raw: str) -> Optional[List[Any]]:
    """
    Try extracting a top-level JSON array from the LLM's response.
    If multiple arrays are present, pick the first 4-element one.
    """
    if not raw:
        return None
    # First, exact parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Fallback: search for [...], then attempt json.loads on candidate slices
    matches = re.findall(r"\[[\s\S]*?\]", raw)
    for m in matches:
        try:
            candidate = json.loads(m)
            if isinstance(candidate, list):
                return candidate
        except Exception:
            continue
    return None


def normalize_answer_array(ans: List[Any]) -> List[Any]:
    """
    Ensure the array has exactly 4 items in the correct types:
      [int, str, float, str]
    If incoming types differ, attempt safe coercions.
    """
    if not isinstance(ans, list):
        return ["Error", "N/A", 0.0, ""]
    # pad/trim
    if len(ans) < 4:
        ans = ans + [""] * (4 - len(ans))
    elif len(ans) > 4:
        ans = ans[:4]

    # 0 -> int (count)
    try:
        ans[0] = int(float(ans[0]))
    except Exception:
        ans[0] = "Error"

    # 1 -> str (title)
    ans[1] = "" if ans[1] is None else str(ans[1])

    # 2 -> float (rounded 6)
    try:
        ans[2] = round(float(ans[2]), 6)
    except Exception:
        ans[2] = 0.0

    # 3 -> str (data URI)
    ans[3] = "" if ans[3] is None else str(ans[3])

    return ans


# ------------------------------------------------------------------------------------
# Routes: Web pages
# ------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_file(request: Request, questions_txt: UploadFile = File(...)):
    try:
        # Read file content
        text = (await questions_txt.read()).decode("utf-8").strip()
        if not text:
            return templates.TemplateResponse("analysis_result.html", {
                "request": request,
                "error": "Uploaded file is empty.",
                "result": None
            })

        breakdown = task_breakdown(text)  # This returns your [4, "Titanic", ...]

        return templates.TemplateResponse("analysis_result.html", {
            "request": request,
            "error": None,
            "result": breakdown
        })

    except Exception as e:
        logger.exception("Error in /analyze")
        return templates.TemplateResponse("analysis_result.html", {
            "request": request,
            "error": str(e),
            "result": None
        })

# ------------------------------------------------------------------------------------
# Routes: API endpoints (curl + programmatic)
# ------------------------------------------------------------------------------------
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"


@app.post("/api/")
async def upload_api(file: UploadFile = File(...), mode: str = Query("auto")):
    """
    API endpoint: accepts a file (question.txt). Returns the JSON array result.
    - mode=auto: verified Python for Wikipedia task; otherwise LLM
    - mode=python: force server-side verified Python
    - mode=llm: force LLM (still normalized)
    """
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        if mode == "python" or (mode == "auto" and looks_like_highest_grossing_task(text)):
            answers = compute_highest_grossing_answers()
            return JSONResponse(content=answers)

        # LLM path
        raw = task_breakdown(text)
        parsed = parse_llm_json_array(raw)
        if parsed is None and looks_like_highest_grossing_task(text):
            answers = compute_highest_grossing_answers()
        else:
            answers = normalize_answer_array(parsed or ["Error", "N/A", 0.0, ""])

        return JSONResponse(content=answers)

    except Exception as e:
        logger.exception("/api/ failed")
        return JSONResponse(status_code=500, content=["Error", "N/A", 0.0, ""])


@app.get("/highest-grossing")
async def highest_grossing():
    """
    Direct JSON endpoint for the Wikipedia highest-grossing films task.
    Uses only the verified Python approach.
    """
    try:
        answers = compute_highest_grossing_answers()
        return JSONResponse(content=answers)
    except Exception:
        return JSONResponse(content=["Error", "N/A", 0.0, ""])


# ------------------------------------------------------------------------------------
# Uvicorn entrypoint
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Host/port can be overridden by Render, but this is fine for local runs.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

# ====================================================================================
# End of file
# ====================================================================================

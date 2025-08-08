# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
#   "requests",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "beautifulsoup4",
#   "lxml",
#   "pillow",
# ]
# ///

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import io
import base64
import re

# scraping/data libs
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# LLM
import google.generativeai as genai

# local preprocessor
from prompt_preprocessor import preprocess_prompt, remove_example_output

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Live scraper / compute
# -----------------------
def compute_answers():
    """
    Scrape Wikipedia 'List of highest-grossing films' and return
    [q1_count (int), q2_title (str), q3_corr (float rounded to 6), q4_data_uri (str)]
    """

    URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}

    def to_number(x):
        if pd.isna(x):
            return np.nan
        s = re.sub(r'\[.*?\]', '', str(x))  # strip footnotes
        s = s.replace(',', '')
        s = re.sub(r'[^0-9.\-]', '', s)
        try:
            return float(s) if s != '' else np.nan
        except:
            return np.nan

    # Fetch page and read any tables
    resp = requests.get(URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    # Find a table that contains Rank & Peak & a gross column
    table = None
    for t in tables:
        cols_lower = [str(c).lower() for c in t.columns]
        if any("rank" in c for c in cols_lower) and any("peak" in c for c in cols_lower) and any("gross" in c or "world" in c for c in cols_lower):
            table = t
            break
    if table is None:
        # fallback: first table that has 'world' or 'gross'
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("world" in c or "gross" in c for c in cols_lower):
                table = t
                break

    if table is None:
        raise RuntimeError("Could not locate the Wikipedia table for highest-grossing films.")

    # helper to match columns flexibly
    def col_match(df, names):
        for name in names:
            for c in df.columns:
                if name in str(c).lower():
                    return c
        return None

    # Normalize column names
    film_col = col_match(table, ["film", "title", "movie"])
    year_col = col_match(table, ["year", "release"])
    gross_col = col_match(table, ["world", "gross", "worldwide"])
    rank_col = col_match(table, ["rank"])
    peak_col = col_match(table, ["peak"])

    if not all([film_col, year_col, gross_col, rank_col, peak_col]):
        # Try fuzzy checks — sometimes column headers have extra characters
        # If still missing, raise error
        raise RuntimeError(f"Missing required columns. Detected: film={film_col}, year={year_col}, gross={gross_col}, rank={rank_col}, peak={peak_col}")

    df = table[[film_col, year_col, gross_col, rank_col, peak_col]].copy()
    df.columns = ["Film", "Year", "Worldwide_gross", "Rank", "Peak"]

    # Clean numeric columns
    df["Worldwide_gross"] = df["Worldwide_gross"].apply(to_number)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # Peak may contain strings; extract first numeric
    df["Peak"] = df["Peak"].astype(str).str.extract(r'(-?\d+\.?\d*)', expand=False).astype(float)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    df = df.dropna(subset=["Worldwide_gross", "Year", "Rank", "Peak"])

    # Q1: number of $2B+ movies released before 2000
    q1_count = int(df[(df["Worldwide_gross"] >= 2_000_000_000) & (df["Year"] < 2000)].shape[0])

    # Q2: earliest film grossing >= $1.5B
    df_1_5 = df[df["Worldwide_gross"] >= 1_500_000_000].sort_values("Year")
    q2_title = str(df_1_5.iloc[0]["Film"]) if not df_1_5.empty else ""

    # Q3: Pearson correlation between Rank and Peak (rounded to 6 decimals)
    corr = df["Rank"].corr(df["Peak"])
    if pd.isna(corr):
        q3_corr = 0.0
    else:
        q3_corr = round(float(corr), 6)

    # Q4: scatterplot Rank vs Peak + dotted red regression line -> encode as base64 PNG URI, <100KB
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Rank"], df["Peak"], s=20)
    try:
        z = np.polyfit(df["Rank"], df["Peak"], 1)
        p = np.poly1d(z)
        xs = np.linspace(df["Rank"].min(), df["Rank"].max(), 200)
        ax.plot(xs, p(xs), linestyle="--", color="red", linewidth=1)
    except Exception:
        pass
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.set_title("Rank vs Peak")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    img_bytes = buf.getvalue()
    buf.close()

    # If too large, try simple downscaling with PIL to get under 100KB
    max_size = 100_000
    if len(img_bytes) > max_size:
        try:
            im = Image.open(io.BytesIO(img_bytes))
            # progressively reduce size until under limit
            for scale in [0.8, 0.7, 0.6, 0.5]:
                w, h = im.size
                im2 = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                out = io.BytesIO()
                im2.save(out, format="PNG", optimize=True)
                img_bytes = out.getvalue()
                out.close()
                if len(img_bytes) <= max_size:
                    break
        except Exception:
            # fallback: truncate bytes (last resort)
            img_bytes = img_bytes[:max_size]

    data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

    return [q1_count, q2_title, q3_corr, data_uri]


# -----------------------
# Prompt / LLM handling
# -----------------------
def task_breakdown(task: str) -> str:
    """
    Preprocesses the incoming task (removes example outputs / appends runtime instructions)
    then calls Gemini and returns the response text.
    """
    try:
        # Preprocess the task (remove example outputs and add explicit scrape instructions when appropriate)
        task_clean = remove_example_output(task)
        task_clean = preprocess_prompt(task_clean)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # don't crash; return informative message
            return "GEMINI_API_KEY not set in environment."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # load the fixed prompt template for breakdowns (keeps separation of concerns)
        prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
        if not os.path.exists(prompt_path):
            return f"Prompt file missing: {prompt_path}"

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Combine preprocessed task + prompt template
        contents = [task_clean.strip(), prompt_template.strip()]

        response = model.generate_content(contents)

        out_path = "abdul_breaked_task.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        return response.text

    except Exception as e:
        return f"Error during task breakdown: {str(e)}"


# -----------------------
# Endpoints
# -----------------------
@app.get("/")
async def root():
    return {"message": "Hello!"}


@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        breakdown = task_breakdown(text)
        return {"filename": file.filename, "content": text, "breakdown": breakdown}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/highest-grossing")
async def highest_grossing():
    try:
        answers = compute_answers()
        return JSONResponse(content=answers)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Local dev run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

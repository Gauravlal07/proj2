# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "beautifulsoup4",
#   "lxml",
#   "pillow",
#   "requests",
# ]
# ///

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os

# extra imports for scraping
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
from PIL import Image

app = FastAPI()

# --- CORS middleware setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Wikipedia scraping function ===
def compute_answers():
    URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}

    def to_number(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = re.sub(r'\[.*?\]', '', s)
        s = s.replace(',', '')
        s = re.sub(r'[^0-9.\-]', '', s)
        try:
            return float(s) if s != '' else np.nan
        except:
            return np.nan

    tables = pd.read_html(requests.get(URL, headers=HEADERS, timeout=15).text)

    # pick table with Rank & Peak
    def find_best_table():
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("rank" in c for c in cols) and any("peak" in c for c in cols):
                return t
        return None

    table = find_best_table()
    if table is None:
        raise RuntimeError("Table not found")

    def col_match(df, names):
        for name in names:
            for c in df.columns:
                if name in str(c).lower():
                    return c
        return None

    df = table.copy()
    df = df.rename(columns={
        col_match(table, ["film", "title", "movie"]): "Film",
        col_match(table, ["year"]): "Year",
        col_match(table, ["world", "gross"]): "Worldwide_gross",
        col_match(table, ["rank"]): "Rank",
        col_match(table, ["peak"]): "Peak",
    })

    df["Worldwide_gross"] = df["Worldwide_gross"].apply(to_number)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Peak"] = df["Peak"].astype(str).str.extract(r'(-?\d+\.?\d*)').astype(float)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Worldwide_gross", "Rank", "Peak"])

    # Q1
    q1_count = int(df[(df["Worldwide_gross"] >= 2_000_000_000) & (df["Year"] < 2000)].shape[0])

    # Q2
    q2_title = df[df["Worldwide_gross"] >= 1_500_000_000].sort_values("Year").iloc[0]["Film"]

    # Q3
    q3_corr = round(df["Rank"].corr(df["Peak"]), 6)

    # Q4
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Rank"], df["Peak"])
    z = np.polyfit(df["Rank"], df["Peak"], 1)
    p = np.poly1d(z)
    ax.plot(df["Rank"], p(df["Rank"]), "r--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.set_title("Rank vs Peak")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    img_str = base64.b64encode(buf.getvalue()).decode()

    return [q1_count, str(q2_title), float(q3_corr), f"data:image/png;base64,{img_str}"]

# --- Gemini task breakdown function ---
def task_breakdown(task: str) -> str:
    """Breaks down a task into programmable steps using Google Gemini."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, "r") as f:
            prompt = f.read()
        contents = [task.strip(), prompt.strip()]
        response = model.generate_content(contents)
        output_path = "abdul_breaked_task.txt"
        with open(output_path, "w") as f:
            f.write(response.text)
        return response.text
    except Exception as e:
        return f"Error during task breakdown:\n  {str(e)}"

# --- Root endpoint ---
@app.get("/")
async def root():
    return {"message": "Hello!"}

# --- File upload endpoint ---
@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
        breakdown = task_breakdown(text)
        return {
            "filename": file.filename,
            "content": text,
            "breakdown": breakdown,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Highest-grossing films endpoint ---
@app.get("/highest-grossing")
async def highest_grossing():
    try:
        answers = compute_answers()
        return JSONResponse(content=answers)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Local dev run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

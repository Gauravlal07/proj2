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

# imports needed for scraping
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
from PIL import Image
import google.generativeai as genai

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Live scrape + computation endpoint
def compute_answers():
    URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}

    def to_number(x):
        if pd.isna(x): return np.nan
        s = re.sub(r'\[.*?\]', '', str(x)).replace(',', '')
        s = re.sub(r'[^0-9.\-]', '', s)
        try: return float(s) if s else np.nan
        except: return np.nan

    tables = pd.read_html(requests.get(URL, headers=HEADERS, timeout=15).text)
    table = next((t for t in tables if any("rank" in c.lower() for c in t.columns) and any("peak" in c.lower() for c in t.columns)), None)
    if table is None:
        raise RuntimeError("Relevant table not found on Wikipedia page.")

    def col_match(df, names):
        for name in names:
            for c in df.columns:
                if name in str(c).lower():
                    return c
        return None

    df = table.copy()
    df = df.rename(columns={
        col_match(df, ["film", "title", "movie"]): "Film",
        col_match(df, ["year"]): "Year",
        col_match(df, ["world", "gross"]): "Worldwide_gross",
        col_match(df, ["rank"]): "Rank",
        col_match(df, ["peak"]): "Peak",
    })

    df["Worldwide_gross"] = df["Worldwide_gross"].apply(to_number)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Peak"] = df["Peak"].astype(str).str.extract(r'(-?\d+\.?\d*)').astype(float)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Worldwide_gross", "Year", "Rank", "Peak"])

    q1_count = int(df[(df["Worldwide_gross"] >= 2_000_000_000) & (df["Year"] < 2000)].shape[0])

    over_1_5 = df[df["Worldwide_gross"] >= 1_500_000_000].sort_values("Year")
    q2_title = str(over_1_5.iloc[0]["Film"]) if not over_1_5.empty else ""

    q3_corr = round(df["Rank"].corr(df["Peak"]), 6) if df.shape[0] >= 2 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Rank"], df["Peak"])
    try:
        z = np.polyfit(df["Rank"], df["Peak"], 1)
        p = np.poly1d(z)
        xs = np.linspace(df["Rank"].min(), df["Rank"].max(), 200)
        ax.plot(xs, p(xs), "r--")
    except:
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

    # enforce <100KB
    if len(img_bytes) > 100_000:
        img_bytes = img_bytes[:100_000]

    data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

    return [q1_count, q2_title, float(q3_corr), data_uri]

from prompt_preprocessor import preprocess_prompt

def task_breakdown(task: str) -> str:
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

        # NEW: rewrite incoming question
        task = preprocess_prompt(task)

        contents = [task.strip(), prompt.strip()]
        response = model.generate_content(contents)

        output_path = "abdul_breaked_task.txt"
        with open(output_path, "w") as f:
            f.write(response.text)

        return response.text

    except Exception as e:
        return f"Error during task breakdown:\n  {str(e)}"

# Endpoints
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
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# app.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import requests
import logging
import json
import re
import io
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

# local prompt preprocessor
from prompt_preprocessor import preprocess_prompt, remove_example_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve the `static/` directory as the site root (index.html will be served)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# -----------------------
# AIPipe / task breakdown
# -----------------------
def task_breakdown(task: str) -> str:
    """
    Preprocesses the incoming task and calls AIPipe (OpenRouter-compatible)
    using the AIPIPE_TOKEN environment variable. Returns the assistant text.
    """
    try:
        task_clean = remove_example_output(task)
        task_clean = preprocess_prompt(task_clean)

        api_key = os.getenv("AIPIPE_TOKEN")
        if not api_key:
            return "AIPIPE_TOKEN not set in environment."

        prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
        if not os.path.exists(prompt_path):
            return f"Prompt file missing: {prompt_path}"

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        full_prompt = f"{task_clean.strip()}\n\n{prompt_template.strip()}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.0
        }

        resp = requests.post(
            "https://aipipe.org/openrouter/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            return "No response from AI Pipe."

        response_text = data["choices"][0]["message"]["content"]

        out_path = "abdul_breaked_task.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response_text)

        return response_text

    except Exception as e:
        logger.exception("task_breakdown failed")
        return f"Error during task breakdown: {str(e)}"

# -----------------------
# Compute answers (scrape + analyze)
# -----------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

def parse_currency_to_float(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = re.sub(r'\[.*?\]', '', s)  # remove refs like [1]
    s = s.replace('\xa0', ' ')
    s = s.strip()
    if s == "":
        return np.nan
    # handle word magnitudes first
    m = re.search(r'([0-9]{1,3}(?:[,\d]*\d|\d*)(?:\.\d+)?)\s*(billion|bn|b\.)', s, flags=re.I)
    if m:
        return float(m.group(1).replace(',', '')) * 1e9
    m = re.search(r'([0-9]{1,3}(?:[,\d]*\d|\d*)(?:\.\d+)?)\s*(million|m)', s, flags=re.I)
    if m:
        return float(m.group(1).replace(',', '')) * 1e6
    # handle plain numeric with $ and commas
    digits = re.sub(r'[^\d\.\-]', '', s)
    if digits == "":
        # fallback: try removing only brackets & letters, keep commas
        digits2 = re.sub(r'[^\d,\.]', '', s).replace(',', '')
        try:
            return float(digits2)
        except:
            return np.nan
    try:
        return float(digits)
    except:
        return np.nan

def choose_table(tables):
    """
    Heuristic: prefer a table that contains 'worldwide' and 'film/title'
    and has parseable numeric gross values.
    """
    best = None
    best_score = -1
    for idx, tb in enumerate(tables):
        cols = [str(c).lower() for c in tb.columns.astype(str)]
        colstr = " ".join(cols)
        score = 0
        if any("worldwide" in c for c in cols) or any("gross" in c for c in cols):
            score += 2
        if any("film" in c or "title" in c for c in cols):
            score += 2
        if any("year" in c for c in cols):
            score += 1
        # check how many rows have parseable gross
        candidate_gross_cols = [c for c in tb.columns if 'world' in str(c).lower() or 'gross' in str(c).lower()]
        num_parseable = 0
        if candidate_gross_cols:
            col = candidate_gross_cols[0]
            for val in tb[col].astype(str).head(20):
                if not pd.isna(parse_currency_to_float(val)):
                    num_parseable += 1
        score += num_parseable
        # prefer larger tables with higher score
        if score > best_score:
            best_score = score
            best = (idx, tb)
    return best

def compute_answers():
    try:
        resp = requests.get(WIKI_URL, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text, flavor='lxml')
        if not tables:
            raise RuntimeError("No tables found on page")

        chosen = choose_table(tables)
        if chosen is None:
            raise RuntimeError("No suitable table found")
        idx, main_table = chosen
        logger.info("Using table index %s with columns: %s", idx, list(main_table.columns))

        # normalize column names
        main_table.columns = [str(c).strip() for c in main_table.columns]
        df = main_table.copy()

        # find film/title col
        film_col = None
        for c in df.columns:
            if re.search(r'film|title', str(c), re.I):
                film_col = c
                break
        if film_col is None:
            # fallback to first text column
            film_col = df.columns[0]

        # find gross column
        gross_col = None
        for c in df.columns:
            if re.search(r'worldwide|worldwide gross|gross', str(c), re.I):
                gross_col = c
                break
        if gross_col is None:
            # try other heuristics
            for c in df.columns:
                if re.search(r'\$', str(df[c].astype(str).head(5).to_list()), re.I):
                    gross_col = c
                    break

        # find year column
        year_col = None
        for c in df.columns:
            if re.search(r'year', str(c), re.I):
                year_col = c
                break

        # find rank column
        rank_col = None
        for c in df.columns:
            if re.search(r'rank|#', str(c), re.I):
                rank_col = c
                break
        # many Wikipedia tables have rank in first column or as index
        if rank_col is None:
            # try numeric first column
            for c in df.columns[:3]:
                if df[c].astype(str).str.match(r'^\s*\d+\s*$').any():
                    rank_col = c
                    break

        # find peak column (optional)
        peak_col = None
        for c in df.columns:
            if re.search(r'peak', str(c), re.I):
                peak_col = c
                break

        # create cleaned columns
        df_clean = pd.DataFrame()
        df_clean['Film'] = df[film_col].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()
        if gross_col is not None:
            df_clean['Worldwide gross raw'] = df[gross_col].astype(str)
            df_clean['Gross($)'] = df[gross_col].apply(parse_currency_to_float)
        else:
            df_clean['Worldwide gross raw'] = ""
            df_clean['Gross($)'] = np.nan

        if year_col is not None:
            df_clean['Year'] = df[year_col].astype(str).str.extract(r'(\d{4})').astype(float).astype('Int64', errors='ignore')
        else:
            df_clean['Year'] = pd.NA

        if rank_col is not None:
            df_clean['Rank'] = df[rank_col].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64', errors='ignore')
        else:
            df_clean['Rank'] = pd.NA

        if peak_col is not None:
            # parse numeric peak if possible
            df_clean['Peak raw'] = df[peak_col].astype(str)
            df_clean['Peak'] = df[peak_col].apply(lambda x: pd.to_numeric(re.sub(r'[^\d\.\-]', '', str(x)), errors='coerce'))
        else:
            df_clean['Peak'] = pd.NA

        # drop rows missing Rank and Gross
        # We'll keep rows that have either Gross or Peak (for plotting)
        df_clean = df_clean.reset_index(drop=True)

        # Q1: number of $2B movies released before 2020
        if 'Gross($)' in df_clean.columns:
            q1_count = int(df_clean.loc[(df_clean['Gross($)'] >= 2e9) & (pd.to_numeric(df_clean['Year'], errors='coerce') < 2020)].shape[0])
        else:
            q1_count = 0

        # Q2: earliest film that grossed over $1.5bn
        if 'Gross($)' in df_clean.columns and df_clean['Gross($)'].notna().any():
            over_15 = df_clean.loc[df_clean['Gross($)'] >= 1.5e9].copy()
            if not over_15.empty and over_15['Year'].notna().any():
                min_year = int(over_15['Year'].astype(float).min())
                earliest_row = over_15.loc[over_15['Year'].astype(float) == min_year].iloc[0]
                q2_film = str(earliest_row['Film'])
            elif not over_15.empty:
                q2_film = str(over_15.iloc[0]['Film'])
            else:
                q2_film = "N/A"
        else:
            q2_film = "N/A"

        # Q3: correlation between Rank and Peak (prefer Peak column if available)
        corrval = 0.0
        corr_source = None
        if df_clean['Rank'].notna().sum() >= 2:
            # use Peak if available & numeric
            if df_clean['Peak'].notna().sum() >= 2:
                xy = df_clean[['Rank', 'Peak']].dropna().astype(float)
                corrval = float(xy['Rank'].corr(xy['Peak']))
                corr_source = 'Peak'
            elif df_clean['Gross($)'].notna().sum() >= 2:
                xy = df_clean[['Rank', 'Gross($)']].dropna().astype(float)
                corrval = float(xy['Rank'].corr(xy['Gross($)']))
                corr_source = 'Gross'
            else:
                corrval = 0.0
        corrval = 0.0 if pd.isna(corrval) else round(float(corrval), 6)

        # Q4: plot Rank vs Peak (or Gross fallback)
        plot_x = None
        plot_y = None
        y_label = None
        if corr_source == 'Peak':
            tmp = df_clean[['Rank', 'Peak']].dropna().astype(float)
            plot_x = tmp['Rank'].values
            plot_y = tmp['Peak'].values
            y_label = "Peak"
        elif corr_source == 'Gross':
            tmp = df_clean[['Rank', 'Gross($)']].dropna().astype(float)
            plot_x = tmp['Rank'].values
            plot_y = tmp['Gross($)'].values
            y_label = "Worldwide gross ($)"
        else:
            # not enough data to plot
            plot_x = np.array([])
            plot_y = np.array([])
            y_label = "Value"

        if plot_x.size == 0 or plot_y.size == 0:
            plot_data_uri = ""
        else:
            # create scatter and regression
            fig, ax = plt.subplots(figsize=(6,4), dpi=110)
            ax.scatter(plot_x, plot_y, s=40, alpha=0.75)
            if plot_x.size >= 2:
                z = np.polyfit(plot_x, plot_y, 1)
                p = np.poly1d(z)
                xx = np.linspace(plot_x.min(), plot_x.max(), 200)
                ax.plot(xx, p(xx), linestyle=':', color='red', linewidth=1.8)
            ax.set_xlabel("Rank")
            ax.set_ylabel(y_label)
            ax.set_title("Rank vs " + y_label)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
            plt.close(fig)
            img_bytes = buf.getvalue()

            # iteratively shrink until under 100KB (100,000 bytes)
            max_bytes = 100000
            if len(img_bytes) > max_bytes:
                # try reducing size / dpi and quantize
                # Step 1: reduce dpi and figure size
                for (w, h, dpi) in [(5,3,80), (4,3,70), (3.5,2.5,60), (3,2,50)]:
                    fig, ax = plt.subplots(figsize=(w,h), dpi=dpi)
                    ax.scatter(plot_x, plot_y, s=20, alpha=0.75)
                    if plot_x.size >= 2:
                        ax.plot(xx, p(xx), linestyle=':', color='red', linewidth=1.2)
                    ax.set_xlabel("Rank")
                    ax.set_ylabel(y_label)
                    ax.set_title("Rank vs " + y_label)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
                    plt.close(fig)
                    img_bytes = buf.getvalue()
                    if len(img_bytes) <= max_bytes:
                        break

            # if still too big, quantize using PIL (8-bit palette)
            if len(img_bytes) > max_bytes:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                    # convert to P (palette) to reduce size
                    img_p = img.convert('P', palette=Image.ADAPTIVE)
                    bufq = io.BytesIO()
                    img_p.save(bufq, format='PNG', optimize=True)
                    img_bytes = bufq.getvalue()
                except Exception:
                    # if quantize fails, keep existing img_bytes
                    logger.exception("quantize failed")

            # final check: if still too big, truncate (last resort) -> return empty plot to avoid huge payload
            if len(img_bytes) > max_bytes:
                logger.warning("Plot still > %d bytes after reductions (%d). Returning empty plot.", max_bytes, len(img_bytes))
                plot_data_uri = ""
            else:
                plot_data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')

        answers = [int(q1_count), str(q2_film), float(corrval), str(plot_data_uri)]
        return answers

    except Exception as e:
        logger.exception("compute_answers failed")
        return ["Error", "N/A", 0.0, ""]


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
        logger.exception("upload_file error")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/highest-grossing")
async def highest_grossing():
    try:
        answers = compute_answers()
        # Ensure JSON serializable types: first int, second str, third float, fourth str
        if isinstance(answers, list) and len(answers) == 4:
            # fix types
            answers[0] = int(answers[0]) if isinstance(answers[0], (int, np.integer)) else answers[0]
            answers[1] = str(answers[1])
            answers[2] = float(answers[2]) if isinstance(answers[2], (float, np.floating, int, np.integer)) else 0.0
            answers[3] = str(answers[3])
            return JSONResponse(content=answers)
        else:
            return JSONResponse(content=["Error", "N/A", 0.0, ""])
    except Exception as e:
        logger.exception("highest_grossing endpoint error")
        return JSONResponse(content=["Error", "N/A", 0.0, ""])
@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
        breakdown = task_breakdown(text)  # keep your existing function call
        return {"filename": file.filename, "content": text, "breakdown": breakdown}
    except Exception as e:
        logger.exception("upload_file error")
        return JSONResponse(status_code=500, content={"error": str(e)})

# === NEW WEB ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Show homepage with links."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Show upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_file(request: Request, questions_txt: UploadFile = File(...)):
    """
    Accept uploaded file, run existing task_breakdown, show results.
    """
    try:
        # Save temp file
        file_path = f"temp_{questions_txt.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(questions_txt.file, buffer)

        # Read text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Call existing function
        breakdown = task_breakdown(text)

        # Remove temp file
        os.remove(file_path)

        return templates.TemplateResponse(
            "analysis_result.html",
            {"request": request, "filename": questions_txt.filename, "content": text, "breakdown": json.dumps(breakdown, indent=2)}
        )

    except Exception as e:
        logger.exception("analyze_file error")
        return templates.TemplateResponse(
            "analysis_result.html",
            {"request": request, "error": str(e)}
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

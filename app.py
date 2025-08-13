"""
app.py
------
FastAPI app that supports:

  • /              -> serves index.html
  • /upload        -> serves an upload form
  • /api/          -> cURL endpoint; POST -F file=@question.txt   (unchanged)
  • /analyze       -> Browser form POST; uses same logic as /api/ and renders analysis_result.html

Design goals:
  - Keep /api/ behavior intact for cURL users.
  - /analyze uses the *same* analysis function so results match.
  - Show result exactly like: [4,"Titanic", ...] in the browser.
  - Avoid "Undefined is not JSON serializable" by never pushing Jinja2 Undefined into JSON.
  - Robust logging, rate-limit/backoff, and safe error pages.
  - Minimal filesystem assumptions; auto-create templates if missing.
  - Mount /static for assets (CSS/JS/images), but templates themselves live in /static by request.

Environment:
  - AIPIPE_TOKEN must be set for AI Pipe (OpenRouter-compatible) calls.
  - AIPIPE_ENDPOINT / AIPIPE_MODEL can be overridden via env if needed.

Run locally:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import uuid
import shutil
import logging
import traceback
from typing import Any, Dict, Optional, Tuple, Callable, Iterable, List

import requests
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Request,
    Response,
    Depends,
    Header,
    status,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ------------------------------------------------------------------------------------
# Load environment early (Render/Heroku compatible)
# ------------------------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------------------------
# Config (env overridable)
# ------------------------------------------------------------------------------------
APP_NAME: str = os.getenv("APP_NAME", "proj2")
ENV: str = os.getenv("ENV", "production").lower()
DEBUG: bool = os.getenv("DEBUG", "0").strip() in {"1", "true", "yes"}
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
FORCE_TEMPLATE_SETUP: bool = os.getenv("FORCE_TEMPLATE_SETUP", "1") in {"1", "true", "yes"}

# AI Pipe (OpenRouter-compatible)
AIPIPE_ENDPOINT: str = os.getenv("AIPIPE_ENDPOINT", "https://aipipe.org/openrouter/v1/chat/completions")
AIPIPE_MODEL: str = os.getenv("AIPIPE_MODEL", "gpt-4.1")
AIPIPE_TIMEOUT_SEC: int = int(os.getenv("AIPIPE_TIMEOUT_SEC", "120"))
AIPIPE_TOKEN: Optional[str] = os.getenv("AIPIPE_TOKEN")  # must be set by deploy env

# Paths
STATIC_DIR: str = os.getenv("STATIC_DIR", "static")
TEMPLATES_DIR: str = STATIC_DIR  # per your request: templates live under "static/"
DEFAULT_INDEX_FILE: str = os.path.join(TEMPLATES_DIR, "index.html")
DEFAULT_UPLOAD_FILE: str = os.path.join(TEMPLATES_DIR, "upload.html")
DEFAULT_RESULT_FILE: str = os.path.join(TEMPLATES_DIR, "analysis_result.html")

# Logging config
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s"

class RequestIdFilter(logging.Filter):
    """Injects request_id into log record for consistent formatting."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(APP_NAME)
logger.addFilter(RequestIdFilter())

# ------------------------------------------------------------------------------------
# Utilities: request ID middleware support
# ------------------------------------------------------------------------------------
def get_request_id() -> str:
    # generate a compact request id
    return uuid.uuid4().hex[:12]

def attach_request_id_to_logger(request_id: str) -> None:
    # helper: populate default extra in the logger (per thread)
    # We’ll use LoggerAdapter for scoped logs per request.
    pass

class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.setdefault("request_id", self.extra.get("request_id", "-"))
        kwargs["extra"] = extra
        return msg, kwargs

# ------------------------------------------------------------------------------------
# Safe JSON helper
# ------------------------------------------------------------------------------------
def to_safe_json(obj: Any) -> Any:
    """
    Convert arbitrary value to a JSON-safe structure without raising.
    - Avoids "Undefined is not JSON serializable" by converting unknowns to strings.
    """
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            if isinstance(obj, dict):
                return {str(k): to_safe_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_safe_json(x) for x in obj]
            return str(obj)
        except Exception:
            return str(obj)

# ------------------------------------------------------------------------------------
# HTML templates default content (only used if files are missing)
# ------------------------------------------------------------------------------------
DEFAULT_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Proj2 – Home</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="/static/site.css" rel="stylesheet"/>
</head>
<body>
  <main class="container">
    <h1>Welcome 👋</h1>
    <p>This site mirrors the <code>/api/</code> cURL workflow in a browser.</p>
    <ul>
      <li><a href="/upload">Upload &amp; Analyze (browser)</a></li>
      <li>cURL: <code>curl -X POST "https://proj2-a38c.onrender.com/api/" -F "file=@question.txt"</code></li>
    </ul>
  </main>
</body>
</html>
"""

DEFAULT_UPLOAD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Upload & Analyze</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="/static/site.css" rel="stylesheet"/>
</head>
<body>
  <main class="container">
    <h1>Upload <code>question.txt</code> and Analyze</h1>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <div class="field">
        <label for="questions_txt">File (question.txt):</label>
        <input type="file" id="questions_txt" name="questions_txt" accept=".txt" required />
      </div>
      <button type="submit">Analyze</button>
    </form>
    <p class="hint">
      This uses the same pipeline as <code>/api/</code> so the result matches your cURL output exactly.
    </p>
  </main>
</body>
</html>
"""

# NOTE: show raw result EXACTLY, no formatting transforms.
DEFAULT_RESULT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Analysis Result</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="/static/site.css" rel="stylesheet"/>
</head>
<body>
  <main class="container">
    <h1>Analysis Result</h1>

    {% if error %}
      <div class="error">
        <strong>Error:</strong> {{ error }}
      </div>
    {% else %}
      <pre class="result">{{ result }}</pre>
    {% endif %}

    <p><a href="/upload">&larr; Analyze another file</a></p>
    <p><a href="/">&larr; Home</a></p>
  </main>
</body>
</html>
"""

DEFAULT_SITE_CSS = """
:root {
  --bg: #0b0f14;
  --panel: #111827;
  --text: #e5e7eb;
  --subtext: #9ca3af;
  --accent: #60a5fa;
  --error: #ef4444;
  --ok: #10b981;
  --muted: #1f2937;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","Apple Color Emoji","Segoe UI Emoji";line-height:1.5}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:820px;margin:32px auto;padding:24px;background:var(--panel);border:1px solid var(--muted);border-radius:16px}
h1{margin-top:0}
.field{margin:16px 0}
input[type="file"]{width:100%;padding:10px;background:var(--bg);border:1px solid var(--muted);border-radius:10px;color:var(--text)}
button{appearance:none;border:0;background:var(--accent);color:black;font-weight:700;padding:12px 16px;border-radius:10px;cursor:pointer}
button:hover{filter:brightness(1.1)}
.error{padding:12px;border:1px solid var(--error);background:rgba(239,68,68,.15);border-radius:10px;margin-bottom:12px}
.result{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--muted);padding:14px;border-radius:10px}
.hint{color:var(--subtext);font-size:.95rem}
code{background:var(--bg);padding:2px 6px;border-radius:6px;border:1px solid var(--muted)}
"""

# ------------------------------------------------------------------------------------
# App + middleware
# ------------------------------------------------------------------------------------
app = FastAPI(title=APP_NAME, debug=DEBUG)

# gzip to keep payloads smaller
app.add_middleware(GZipMiddleware, minimum_size=512)

# CORS (relaxed; lock down if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists; write default templates if missing
def ensure_static_with_templates() -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)
    # site.css
    css_path = os.path.join(STATIC_DIR, "site.css")
    if FORCE_TEMPLATE_SETUP and not os.path.exists(css_path):
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SITE_CSS)

    # index.html
    if FORCE_TEMPLATE_SETUP and not os.path.exists(DEFAULT_INDEX_FILE):
        with open(DEFAULT_INDEX_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_INDEX_HTML)

    # upload.html
    if FORCE_TEMPLATE_SETUP and not os.path.exists(DEFAULT_UPLOAD_FILE):
        with open(DEFAULT_UPLOAD_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_UPLOAD_HTML)

    # analysis_result.html
    if FORCE_TEMPLATE_SETUP and not os.path.exists(DEFAULT_RESULT_FILE):
        with open(DEFAULT_RESULT_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_RESULT_HTML)

ensure_static_with_templates()

# Mount /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------------------------------------------------------------
# Helpers – preprocess, retries, AI call
# ------------------------------------------------------------------------------------
WHITESPACE_RE = re.compile(r"[ \t]+")

def preprocess_prompt(task: str) -> str:
    """
    Sanitize/normalize the incoming text file content to minimize prompt glitches.
    Keep it conservative: strip leading/trailing whitespace, normalize tabs, remove BOMs.
    """
    if not task:
        return ""
    s = task.replace("\ufeff", "").strip()
    s = WHITESPACE_RE.sub(" ", s)
    return s

def is_retryable_status(status_code: int) -> bool:
    """Retry on 429, 502, 503, 504."""
    return status_code in (429, 502, 503, 504)

def post_with_backoff(
    url: str,
    *,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout: int,
    max_retries: int = 3,
    base_sleep: float = 5.0,
    logger_: logging.LoggerAdapter | logging.Logger = logger,
) -> requests.Response:
    """
    requests.post with basic exponential backoff on retryable statuses.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            if resp.status_code >= 400 and is_retryable_status(resp.status_code) and attempt <= max_retries:
                logger_.warning("Rate/Server error (%s). Retry %s/%s after %.1fs...",
                                resp.status_code, attempt, max_retries, base_sleep)
                time.sleep(base_sleep)
                continue
            return resp
        except requests.RequestException as ex:
            if attempt <= max_retries:
                logger_.warning("Network exception '%s'. Retry %s/%s after %.1fs...",
                                ex.__class__.__name__, attempt, max_retries, base_sleep)
                time.sleep(base_sleep)
                continue
            raise

def task_breakdown(task: str, *, system_prompt: str = "You are a helpful data analysis assistant.") -> str:
    """
    Sends the preprocessed task to AI Pipe and returns the raw assistant text.

    Important:
      - Returns a *string* ALWAYS.
      - On failure, returns a short 'Error: ...' string (so templates don't get Undefined).
      - Writes raw text to 'abdul_breaked_task.txt' for debug.
    """
    clean = preprocess_prompt(task)
    if not clean:
        return "Error: empty prompt."

    api_key = AIPIPE_TOKEN
    if not api_key:
        return "Error: AIPIPE_TOKEN is not set on the server."

    # Optional prompt template, appended if present
    prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
    user_payload = clean
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                tpl = f.read().strip()
            if tpl:
                user_payload = f"{clean}\n\n{tpl}"
        except Exception:
            pass

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

    # Per-request logger with request_id
    request_logger = logger  # will be replaced in endpoints with adapter
    try:
        resp = post_with_backoff(
            AIPIPE_ENDPOINT,
            headers=headers,
            json_payload=payload,
            timeout=AIPIPE_TIMEOUT_SEC,
            logger_=request_logger,
        )
        # HTTP error?
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            # Return a clean message so UI doesn't break
            return f"Error: upstream {resp.status_code} – {resp.text[:500]}"

        data = resp.json()
        # Shape: {choices: [{message: {content: "..."}}, ...]}
        choices = data.get("choices") or []
        if not choices:
            return "Error: empty response from AI."
        content = (choices[0].get("message") or {}).get("content")
        content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        # Persist raw for debugging
        try:
            with open("abdul_breaked_task.txt", "w", encoding="utf-8") as f:
                f.write(content or "")
        except Exception:
            pass
        return content or ""
    except Exception as e:
        # Don't raise—return plain string
        tb = traceback.format_exc(limit=2)
        return f"Error: {e.__class__.__name__}: {str(e)}\n{tb}"

# ------------------------------------------------------------------------------------
# Dependency: per-request logger adapter with request_id
# ------------------------------------------------------------------------------------
async def with_request_logger(request: Request, x_request_id: Optional[str] = Header(default=None)):
    rid = (x_request_id or request.headers.get("X-Request-Id") or get_request_id())
    adapter = RequestLoggerAdapter(logger, {"request_id": rid})
    # store for later if needed
    request.state.request_id = rid
    request.state.logger = adapter
    return adapter

# ------------------------------------------------------------------------------------
# Root + pages
# ------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, log: RequestLoggerAdapter = Depends(with_request_logger)):
    log.info("GET / (home)")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request, log: RequestLoggerAdapter = Depends(with_request_logger)):
    log.info("GET /upload")
    return templates.TemplateResponse("upload.html", {"request": request})

# ------------------------------------------------------------------------------------
# cURL API (unchanged behavior)
# ------------------------------------------------------------------------------------
@app.post("/api/")
async def api_entry(
    request: Request,
    file: UploadFile = File(...),
    log: RequestLoggerAdapter = Depends(with_request_logger),
):
    """
    Accepts multipart form file upload (question.txt) and returns JSON with
    original content + breakdown (result). This is the endpoint your cURL uses.
    """
    try:
        log.info("POST /api/ – file=%s", file.filename)
        raw = await file.read()
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace").strip()

        if not text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        breakdown = task_breakdown(text)
        # All fields converted to JSON-safe
        payload = to_safe_json({
            "filename": file.filename,
            "content": text,
            "breakdown": breakdown,
        })
        log.info("POST /api/ – success")
        return JSONResponse(status_code=200, content=payload)
    except Exception as e:
        log.error("POST /api/ – exception: %s", e, exc_info=DEBUG)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------------------------------------------------------------------
# Browser flow (/analyze) -> renders analysis_result.html
# ------------------------------------------------------------------------------------
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_file(
    request: Request,
    questions_txt: UploadFile = File(...),
    log: RequestLoggerAdapter = Depends(with_request_logger),
):
    """
    Accepts a file from the browser form, runs the exact same analysis,
    and renders analysis_result.html with the raw string visible in <pre> block.
    """
    temp_path = None
    try:
        log.info("POST /analyze – file=%s", questions_txt.filename)
        # Save temp (optional; mostly to be faithful to your previous flow)
        temp_path = f"temp_{uuid.uuid4().hex}_{questions_txt.filename}"
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(questions_txt.file, out)

        # Read content
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            with open(temp_path, "r", encoding="latin-1", errors="replace") as f:
                text = f.read().strip()

        if not text:
            ctx = {"request": request, "error": "Uploaded file is empty."}
            return templates.TemplateResponse("analysis_result.html", ctx)

        # Use the exact same analysis path as /api/
        breakdown = task_breakdown(text)

        # Show *exactly* the raw result. No JSON dumps, no extra formatting.
        ctx = {"request": request, "result": breakdown}
        log.info("POST /analyze – success")
        return templates.TemplateResponse("analysis_result.html", ctx)
    except Exception as e:
        log.error("POST /analyze – exception: %s", e, exc_info=DEBUG)
        ctx = {"request": request, "error": str(e)}
        return templates.TemplateResponse("analysis_result.html", ctx)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# ------------------------------------------------------------------------------------
# Health + fallback routes
# ------------------------------------------------------------------------------------
@app.get("/healthz")
async def healthz(request: Request, log: RequestLoggerAdapter = Depends(with_request_logger)):
    log.info("GET /healthz")
    return JSONResponse({"ok": True, "name": APP_NAME, "env": ENV})

@app.exception_handler(404)
async def not_found(request: Request, exc, log: RequestLoggerAdapter = Depends(with_request_logger)):
    log.info("404 for %s", request.url.path)
    # For "/" specifically, some platforms mount root differently—ensure redirect to index if needed
    if request.url.path.strip() == "":
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return PlainTextResponse("Not Found", status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc, log: RequestLoggerAdapter = Depends(with_request_logger)):
    log.error("500 for %s – %s", request.url.path, exc, exc_info=DEBUG)
    return PlainTextResponse("Internal Server Error", status_code=500)

# ------------------------------------------------------------------------------------
# Startup: log config + sanity checks
# ------------------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    lg = RequestLoggerAdapter(logger, {"request_id": "startup"})
    lg.info("Starting %s (env=%s, debug=%s)", APP_NAME, ENV, DEBUG)
    lg.info("Static dir: %s", os.path.abspath(STATIC_DIR))
    lg.info("Templates dir: %s", os.path.abspath(TEMPLATES_DIR))
    if not AIPIPE_TOKEN:
        lg.warning("AIPIPE_TOKEN is not set. /api/ and /analyze will return a helpful error.")

# ------------------------------------------------------------------------------------
# If run directly
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=DEBUG)

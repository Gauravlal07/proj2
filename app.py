# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "requests"
# ]
# ///

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import requests

# Local preprocessor
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
# Prompt / LLM handling
# -----------------------
def task_breakdown(task: str) -> str:
    """
    Preprocesses the incoming task, then calls AI Pipe GPT-4.1 via the
    OpenRouter-compatible endpoint, and returns the response text.
    """
    try:
        task_clean = remove_example_output(task)
        task_clean = preprocess_prompt(task_clean)

        api_key = os.getenv("AIPIPE_TOKEN")
        if not api_key:
            return "AIPIPE_API_KEY not set in environment."

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
            ]
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

# Local dev run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

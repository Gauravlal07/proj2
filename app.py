# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
# ]
# ///

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os

app = FastAPI()

# --- CORS middleware setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini task breakdown function ---
def task_breakdown(task: str) -> str:
    """Breaks down a task into programmable steps using Google Gemini."""

    try:
        # Load API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # fallback if "2.0-flash-lite" doesn't exist

        # Load prompt from file
        prompt_path = os.path.join("prompts", "abdul_task_breakdown.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            prompt = f.read()

        # Combine task with prompt
        contents = [task.strip(), prompt.strip()]
        response = model.generate_content(contents)

        # Save response to file
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


# --- Local dev run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

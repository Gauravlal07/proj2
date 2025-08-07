# AI Task Breakdown App

This FastAPI app accepts a task as a text file and uses Google Gemini to break it down into programmable Python steps.

## Endpoints

- `POST /api/` — Upload a `.txt` file containing your task.

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --reload

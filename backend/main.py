"""FastAPI server: PDF upload, ML pipeline progress, assessment + submit."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT / "questions pdf"
PDF_PATH = QUESTIONS_DIR / "questions.pdf"
ASSESSMENT_PATH = QUESTIONS_DIR / "assessment.json"

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

PHASE_LABELS = {
    "preprocessing": "Preprocessing",
    "qa_extraction": "Q&A extraction",
    "structuring": "Question structuring",
    "assessment_generation": "Assessment generation",
}

app = FastAPI(title="Model LMS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_worker(job_id: str):
    def progress(pct: int, phase_key: str, label: str):
        with jobs_lock:
            jobs[job_id] = {
                "percent": min(100, max(1, pct)),
                "phase": phase_key,
                "phaseLabel": PHASE_LABELS.get(phase_key, phase_key),
                "message": label,
                "done": False,
                "error": None,
            }

    try:
        with jobs_lock:
            jobs[job_id] = {
                "percent": 1,
                "phase": "preprocessing",
                "phaseLabel": PHASE_LABELS["preprocessing"],
                "message": "Starting…",
                "done": False,
                "error": None,
            }

        run_pipeline(PDF_PATH, ASSESSMENT_PATH, progress)

        with jobs_lock:
            jobs[job_id] = {
                "percent": 100,
                "phase": "assessment_generation",
                "phaseLabel": PHASE_LABELS["assessment_generation"],
                "message": "Complete",
                "done": True,
                "error": None,
            }
    except Exception as e:
        with jobs_lock:
            jobs[job_id] = {
                "percent": 0,
                "phase": "error",
                "phaseLabel": "Error",
                "message": str(e),
                "done": True,
                "error": str(e),
            }


@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF file required")

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    PDF_PATH.write_bytes(content)

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "percent": 1,
            "phase": "preprocessing",
            "phaseLabel": PHASE_LABELS["preprocessing"],
            "message": "Queued…",
            "done": False,
            "error": None,
        }

    t = threading.Thread(target=_job_worker, args=(job_id,), daemon=True)
    t.start()

    return {"jobId": job_id}


@app.get("/api/processing-status")
def processing_status(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return j


@app.get("/api/assessment")
def get_assessment():
    if not ASSESSMENT_PATH.is_file():
        raise HTTPException(404, "No assessment yet. Upload a PDF first.")
    with open(ASSESSMENT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    public = [
        {"id": q["id"], "stem": q["stem"], "options": q["options"]} for q in questions
    ]
    return {"questions": public, "total": len(public)}


@app.post("/api/submit")
def submit(payload: dict[str, Any] = Body(...)):
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "answers must be an object")

    if not ASSESSMENT_PATH.is_file():
        raise HTTPException(404, "No assessment loaded.")

    with open(ASSESSMENT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])

    correct = 0
    details = []
    for q in questions:
        qid = q["id"]
        correct_idx = int(q["correctIndex"])
        try:
            ans = answers.get(qid)
            if ans is None:
                picked = None
            else:
                picked = int(ans)
        except (TypeError, ValueError):
            picked = None
        ok = picked == correct_idx
        if ok:
            correct += 1
        details.append(
            {
                "id": qid,
                "correct": ok,
                "correctIndex": correct_idx,
                "yourIndex": picked,
            }
        )

    total = len(questions)
    out_of = total if total else 1
    return {
        "score": round(100 * correct / out_of, 1),
        "correct": correct,
        "total": total,
        "details": details,
    }


@app.get("/api/health")
def health():
    return {"ok": True}

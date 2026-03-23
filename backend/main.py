"""FastAPI server: PDF upload, ML pipeline progress, assessment + submit."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT / "questions pdf"
PDF_PATH = QUESTIONS_DIR / "questions.pdf"
ASSESSMENT_PATH = QUESTIONS_DIR / "assessment.json"
ATTEMPTS_PATH = QUESTIONS_DIR / "attempts.jsonl"

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
    time_taken_ms = payload.get("timeTakenMs")
    question_times_ms = payload.get("questionTimesMs") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "answers must be an object")
    if not isinstance(question_times_ms, dict):
        question_times_ms = {}

    if not ASSESSMENT_PATH.is_file():
        raise HTTPException(404, "No assessment loaded.")

    with open(ASSESSMENT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])

    attempt_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

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

        time_spent_ms = None
        if qid in question_times_ms and question_times_ms[qid] is not None:
            try:
                time_spent_ms = int(question_times_ms[qid])
            except (TypeError, ValueError):
                time_spent_ms = None

        details.append(
            {
                "id": qid,
                "correct": ok,
                "correctIndex": correct_idx,
                "yourIndex": picked,
                "timeSpentMs": time_spent_ms,
            }
        )

    total = len(questions)
    out_of = total if total else 1
    score = round(100 * correct / out_of, 1)

    attempt_record = {
        "attemptId": attempt_id,
        "createdAt": created_at,
        "totalTimeMs": time_taken_ms,
        "score": score,
        "correct": correct,
        "total": total,
        "details": details,
    }

    ATTEMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")

    return {
        "attemptId": attempt_id,
        "createdAt": created_at,
        "totalTimeMs": time_taken_ms,
        "score": score,
        "correct": correct,
        "total": total,
        "details": details,
    }


@app.get("/api/teacher-report")
def teacher_report():
    if not ASSESSMENT_PATH.is_file():
        raise HTTPException(404, "No assessment loaded.")
    if not ATTEMPTS_PATH.is_file():
        raise HTTPException(404, "No student attempts yet.")

    with open(ASSESSMENT_PATH, encoding="utf-8") as f:
        assessment = json.load(f)
    questions = assessment.get("questions", [])

    # Load attempts (jsonl).
    attempts: list[dict[str, Any]] = []
    with open(ATTEMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                attempts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not attempts:
        raise HTTPException(404, "No student attempts yet.")

    last = attempts[-1]

    # Build quick lookup maps.
    qid_to_index = {q["id"]: i for i, q in enumerate(questions)}
    qid_to_stem = {q["id"]: q["stem"] for q in questions}
    qid_to_options = {q["id"]: q["options"] for q in questions}

    last_details_by_qid: dict[str, Any] = {d["id"]: d for d in last.get("details", [])}

    # Heuristic fallback (in case XGBoost can't be trained).
    def heuristic_pain_points():
        pain_rows: list[dict[str, Any]] = []
        for q in questions:
            qid = q["id"]
            details_all = []
            for a in attempts:
                for d in a.get("details", []):
                    if d.get("id") == qid:
                        details_all.append(d)
            wrong = [d for d in details_all if d.get("yourIndex") is not None and not d.get("correct")]
            attempted = [d for d in details_all if d.get("yourIndex") is not None]
            incorrect_rate = (len(wrong) / len(attempted)) if attempted else 0.0

            wrong_times = [d.get("timeSpentMs") for d in wrong if isinstance(d.get("timeSpentMs"), int)]
            avg_wrong_time = (sum(wrong_times) / len(wrong_times)) if wrong_times else 0.0

            last_d = last_details_by_qid.get(qid, {})
            time_spent_ms = last_d.get("timeSpentMs")
            is_incorrect = not bool(last_d.get("correct")) if last_d else False

            pain_rows.append(
                {
                    "id": qid,
                    "stem": qid_to_stem.get(qid, ""),
                    "options": qid_to_options.get(qid, []),
                    "yourIndex": last_d.get("yourIndex"),
                    "correctIndex": last_d.get("correctIndex"),
                    "correct": last_d.get("correct"),
                    "timeSpentMs": time_spent_ms,
                    "incorrectRate": incorrect_rate,
                    "avgWrongTimeMs": avg_wrong_time,
                    "painScore": incorrect_rate + (avg_wrong_time / 100000.0),
                    "predictedIncorrectProbability": incorrect_rate,
                    "xgbUsed": False,
                    "incorrect": is_incorrect,
                }
            )

        pain_rows.sort(key=lambda r: r["painScore"], reverse=True)
        return pain_rows[:5]

    # Try training XGBoost difficulty model.
    try:
        import numpy as np
        from xgboost import XGBClassifier

        X_rows = []
        y_rows = []

        time_samples = []
        total_time_samples = []

        for a in attempts:
            a_total_time = a.get("totalTimeMs")
            total_time_val = None
            try:
                if a_total_time is not None:
                    total_time_val = int(a_total_time)
            except (TypeError, ValueError):
                total_time_val = None

            details_by_qid = {d["id"]: d for d in a.get("details", [])}
            for q in questions:
                qid = q["id"]
                d = details_by_qid.get(qid)
                if not d:
                    continue

                your_index = d.get("yourIndex")
                time_spent_ms = d.get("timeSpentMs")
                if isinstance(time_spent_ms, int):
                    time_samples.append(time_spent_ms)
                if total_time_val is not None:
                    total_time_samples.append(total_time_val)

                y_rows.append(0 if d.get("correct") else 1)  # 1 = incorrect
                X_rows.append(
                    {
                        "q_index": qid_to_index.get(qid, 0),
                        "your_index": your_index if isinstance(your_index, int) else -1,
                        "time_spent_ms": time_spent_ms if isinstance(time_spent_ms, int) else None,
                        "total_time_ms": total_time_val if total_time_val is not None else None,
                    }
                )

        if len(y_rows) < 30:
            raise ValueError("Not enough data for XGBoost.")

        unique_labels = set(y_rows)
        if len(unique_labels) < 2:
            raise ValueError("Only one class present for XGBoost.")

        median_time = float(np.median(time_samples)) if time_samples else 0.0
        median_total_time = float(np.median(total_time_samples)) if total_time_samples else 0.0

        X = np.array(
            [
                [
                    float(r["q_index"]),
                    float(r["your_index"]),
                    float(r["time_spent_ms"]) if r["time_spent_ms"] is not None else median_time,
                    float(r["total_time_ms"]) if r["total_time_ms"] is not None else median_total_time,
                ]
                for r in X_rows
            ],
            dtype=float,
        )
        y = np.array(y_rows, dtype=int)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        model.fit(X, y)

        # Score pain points for the last attempt.
        last_total_time = last.get("totalTimeMs")
        try:
            last_total_time_val = int(last_total_time) if last_total_time is not None else None
        except (TypeError, ValueError):
            last_total_time_val = None
        if last_total_time_val is None:
            last_total_time_val = int(median_total_time)

        pain_rows = []
        for q in questions:
            qid = q["id"]
            d = last_details_by_qid.get(qid, {})
            your_index = d.get("yourIndex")
            your_index_val = your_index if isinstance(your_index, int) else -1
            time_spent_ms = d.get("timeSpentMs")
            time_spent_val = time_spent_ms if isinstance(time_spent_ms, int) else median_time

            x = np.array(
                [[float(qid_to_index.get(qid, 0)), float(your_index_val), float(time_spent_val), float(last_total_time_val)]],
                dtype=float,
            )
            p_incorrect = float(model.predict_proba(x)[0, 1])
            pain_rows.append(
                {
                    "id": qid,
                    "stem": qid_to_stem.get(qid, ""),
                    "yourIndex": your_index if isinstance(your_index, int) else None,
                    "correctIndex": d.get("correctIndex"),
                    "correct": d.get("correct"),
                    "timeSpentMs": time_spent_ms if isinstance(time_spent_ms, int) else None,
                    "predictedIncorrectProbability": p_incorrect,
                    "xgbUsed": True,
                    "incorrect": not bool(d.get("correct")),
                }
            )

        pain_rows.sort(key=lambda r: r["predictedIncorrectProbability"], reverse=True)
        return {
            "lastAttempt": {
                "attemptId": last.get("attemptId"),
                "createdAt": last.get("createdAt"),
                "totalTimeMs": last.get("totalTimeMs"),
                "score": last.get("score"),
                "correct": last.get("correct"),
                "total": last.get("total"),
            },
            "painPoints": pain_rows[:5],
            "model": {"xgbUsed": True, "trainedRows": len(y_rows)},
        }
    except Exception:
        pain = heuristic_pain_points()
        return {
            "lastAttempt": {
                "attemptId": last.get("attemptId"),
                "createdAt": last.get("createdAt"),
                "totalTimeMs": last.get("totalTimeMs"),
                "score": last.get("score"),
                "correct": last.get("correct"),
                "total": last.get("total"),
            },
            "painPoints": pain,
            "model": {"xgbUsed": False, "reason": "fallback"},
        }


@app.get("/api/health")
def health():
    return {"ok": True}

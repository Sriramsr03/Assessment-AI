import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getAssessment, submitAnswers } from '../lib/assessmentApi.js';

export function StudentPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const loadAssessment = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAssessment();
      setQuestions(data.questions || []);
      setAnswers({});
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAssessment();
  }, [loadAssessment]);

  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === 'visible') void loadAssessment();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [loadAssessment]);

  const pick = useCallback((qid, index) => {
    setAnswers((prev) => ({ ...prev, [qid]: index }));
  }, []);

  const onSubmit = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      const r = await submitAnswers(answers);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Submit failed.');
    } finally {
      setSubmitting(false);
    }
  }, [answers]);

  return (
    <div className="page student-page">
      <div className="view-header">
        <h2>Student — assessment</h2>
        <div className="view-header-actions">
          <button type="button" className="btn secondary" onClick={() => void loadAssessment()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh questions'}
          </button>
          <Link to="/" className="btn secondary">
            Teacher upload
          </Link>
        </div>
      </div>
      <p className="muted">
        PDFs are processed on the <strong>Teacher</strong> page. Run <code>npm run dev:all</code> (or{' '}
        <code>npm run dev:api</code> + <code>npm run dev</code>). Use <strong>Refresh questions</strong> after
        processing finishes.
      </p>

      {loading && <p className="muted">Loading questions…</p>}
      {error && !loading && <p className="msg error">{error}</p>}

      {!loading && !error && questions.length === 0 && (
        <div className="empty">
          <p>
            No questions loaded yet. On <strong>Teacher</strong>, upload the PDF and wait until the bar reaches{' '}
            <strong>100%</strong>. If processing failed (e.g. missing Tesseract), fix it and upload again.
          </p>
          <p className="muted small-hint">
            <strong>Photo scans:</strong> install Tesseract OCR on PATH. Browser tabs that only open the PDF do not run
            the pipeline.
          </p>
          <div className="empty-actions">
            <button type="button" className="btn secondary" onClick={() => void loadAssessment()}>
              Refresh questions
            </button>
            <Link to="/" className="btn">
              Go to teacher upload
            </Link>
          </div>
        </div>
      )}

      {!loading && questions.length > 0 && !result && (
        <form
          className="quiz"
          onSubmit={(e) => {
            e.preventDefault();
            void onSubmit();
          }}
        >
          <ol className="quiz-list">
            {questions.map((q, i) => (
              <li key={q.id} className="quiz-item">
                <p className="quiz-stem">
                  <span className="quiz-num">{i + 1}.</span> {q.stem}
                </p>
                <ul className="quiz-options">
                  {q.options.map((opt, idx) => (
                    <li key={`${q.id}-${idx}`}>
                      <label className="quiz-option">
                        <input
                          type="radio"
                          name={q.id}
                          checked={answers[q.id] === idx}
                          onChange={() => pick(q.id, idx)}
                        />
                        <span className="opt-label">{String.fromCharCode(65 + idx)}.</span>
                        <span>{opt}</span>
                      </label>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ol>
          <div className="quiz-actions">
            <button type="submit" className="btn" disabled={submitting}>
              {submitting ? 'Submitting…' : 'Submit answers'}
            </button>
          </div>
        </form>
      )}

      {result && (
        <div className="result-panel">
          <h3>Your result</h3>
          <p className="result-score">
            Score: <strong>{result.score}%</strong> ({result.correct} / {result.total} correct)
          </p>
          <ul className="result-details">
            {result.details.map((d) => (
              <li key={d.id} className={d.correct ? 'ok' : 'bad'}>
                {d.id}: {d.correct ? 'Correct' : 'Incorrect'}
                {!d.correct && d.yourIndex != null && (
                  <span className="muted">
                    {' '}
                    (you: {String.fromCharCode(65 + d.yourIndex)}, answer: {String.fromCharCode(65 + d.correctIndex)})
                  </span>
                )}
                {!d.correct && d.yourIndex == null && (
                  <span className="muted"> (unanswered)</span>
                )}
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="btn secondary"
            onClick={() => {
              setResult(null);
              setAnswers({});
            }}
          >
            Review quiz again
          </button>
        </div>
      )}
    </div>
  );
}

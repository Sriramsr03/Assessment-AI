import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getTeacherReport } from '../lib/assessmentApi.js';

function formatMs(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '—';
  const s = Math.round(ms / 1000);
  return `${s}s`;
}

export function TeacherReportPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getTeacherReport();
      setReport(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load report.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="page">
      <div className="view-header">
        <h2>Teacher — student report</h2>
        <div className="view-header-actions">
          <button type="button" className="btn secondary" onClick={() => void load()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh report'}
          </button>
          <Link to="/" className="btn secondary">
            Back to upload
          </Link>
        </div>
      </div>

      {error && <p className="msg error">{error}</p>}
      {loading && !error && <p className="muted">Generating report…</p>}

      {!loading && !error && report && (
        <div className="result-panel">
          <h3>Latest attempt</h3>
          <p className="result-score">
            Score: <strong>{report.lastAttempt?.score ?? '—'}%</strong> ({report.lastAttempt?.correct ?? 0} /{' '}
            {report.lastAttempt?.total ?? 0} correct)
          </p>
          <p className="muted">
            Total time taken: <strong>{formatMs(report.lastAttempt?.totalTimeMs)}</strong>
          </p>

          <h3 style={{ marginTop: '1.25rem' }}>Pain points</h3>
          <ul className="result-details">
            {report.painPoints.map((p) => (
              <li key={p.id} className={p.incorrect ? 'bad' : 'ok'}>
                {p.id}: {p.correct ? 'Correct' : 'Incorrect'}
                {typeof p.predictedIncorrectProbability === 'number' && (
                  <span className="muted">
                    {' '}
                    (pain: {Math.round(p.predictedIncorrectProbability * 100)}%, time: {formatMs(p.timeSpentMs)})
                  </span>
                )}
              </li>
            ))}
          </ul>

          <p className="muted small-hint">
            Model: {report.model?.xgbUsed ? 'XGBoost' : 'Heuristic'} (last update computed on the server).
          </p>
        </div>
      )}
    </div>
  );
}


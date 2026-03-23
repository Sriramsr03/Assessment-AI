import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getProcessingStatus, uploadAndProcessPdf } from '../lib/assessmentApi.js';

export function TeacherPage() {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const dragDepth = useRef(0);
  const inputRef = useRef(null);

  const [pdfMode, setPdfMode] = useState('printed');

  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(null);

  const onFile = useCallback(async (fileList) => {
    const file = fileList?.[0];
    if (!file) return;
    setError(null);
    setProgress(null);
    setJobId(null);
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please use a PDF file.');
      return;
    }
    setBusy(true);
    try {
      const { jobId: id } = await uploadAndProcessPdf(file, pdfMode);
      setJobId(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.');
      setBusy(false);
    }
  }, [pdfMode]);

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getProcessingStatus(jobId);
        if (cancelled) return;
        setProgress(s);
        if (s.done) {
          setBusy(false);
        }
        if (s.error) {
          setBusy(false);
          setError(s.error);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Status error');
          setBusy(false);
        }
      }
    };
    const id = setInterval(tick, 200);
    void tick();
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId]);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragDepth.current += 1;
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setDragActive(false);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragDepth.current = 0;
      setDragActive(false);
      if (busy) return;
      void onFile(e.dataTransfer.files);
    },
    [busy, onFile],
  );

  const openFilePicker = useCallback(() => {
    if (!busy) inputRef.current?.click();
  }, [busy]);

  const onKeyDown = useCallback(
    (e) => {
      if (busy) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openFilePicker();
      }
    },
    [busy, openFilePicker],
  );

  const pct = progress?.percent ?? 0;
  const showProcessing = Boolean(jobId && busy && progress);

  return (
    <div className="page">
      <h2>Teacher — upload MCQ (PDF)</h2>
      <p className="muted">
        Upload a PDF with numbered questions (<code>1.</code> …) and options. <strong>Digital PDFs:</strong> options
        as <code>A)</code> lines or plain lines; correct answers from <strong>bold</strong> text.{' '}
        Choose <strong>Written</strong> for ticked answers (requires Tesseract). Start the API with{' '}
        <code>npm run dev:api</code> or <code>npm run dev:all</code>.
      </p>

      <div className="mode-picker">
        <label className="mode-label" htmlFor="pdfMode">
          PDF type
        </label>
        <select
          id="pdfMode"
          className="mode-select"
          value={pdfMode}
          disabled={busy}
          onChange={(e) => setPdfMode(e.target.value)}
        >
          <option value="printed">Printed / digital</option>
          <option value="written">Written / ticked</option>
        </select>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        className="dropzone-input"
        aria-hidden="true"
        tabIndex={-1}
        disabled={busy}
        onChange={(e) => {
          void onFile(e.target.files);
          e.target.value = '';
        }}
      />

      <div
        className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${busy ? 'dropzone-busy' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={openFilePicker}
        onKeyDown={onKeyDown}
        role="button"
        tabIndex={busy ? -1 : 0}
        aria-label="Drop a PDF here or click to browse"
      >
        <div className="dropzone-inner">
          <span className="dropzone-icon" aria-hidden="true">
            ↓
          </span>
          <p className="dropzone-title">
            {busy ? 'Processing…' : 'Drag & drop your PDF here'}
          </p>
          <p className="dropzone-hint">or click to browse your files</p>
        </div>
      </div>

      {showProcessing && progress && (
        <div className="processing-panel" aria-live="polite">
          <div className="processing-header">
            <span className="processing-spinner" aria-hidden="true" />
            <span className="processing-pct">{Math.round(pct)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <p className="processing-phase">{progress.phaseLabel}</p>
          <p className="processing-msg">{progress.message}</p>
        </div>
      )}

      <div className="actions">
        <Link to="/assessment" className="btn">
          Student — take assessment
        </Link>
      </div>

      {progress?.done && !progress?.error && (
        <p className="msg success">Assessment ready. Open the student page to take the quiz.</p>
      )}
      {error && <p className="msg error">{error}</p>}
    </div>
  );
}

export async function uploadAndProcessPdf(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/process-pdf', {
    method: 'POST',
    body: fd,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'Upload failed.');
  }
  return res.json();
}

export async function getProcessingStatus(jobId) {
  const res = await fetch(`/api/processing-status?job_id=${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error('Could not read processing status.');
  return res.json();
}

export async function getAssessment() {
  const res = await fetch('/api/assessment');
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'No assessment available.');
  }
  return res.json();
}

export async function submitAnswers(answers) {
  const res = await fetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'Submit failed.');
  }
  return res.json();
}

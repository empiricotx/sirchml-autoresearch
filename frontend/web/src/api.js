const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

export function getSessions() {
  return request("/sessions");
}

export function getSession(sessionId) {
  return request(`/sessions/${sessionId}`);
}

export function getRun(sessionId, runId) {
  return request(`/sessions/${sessionId}/runs/${runId}`);
}

import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";

export function LoadingState({ label = "Loading..." }) {
  return <div className="panel muted">{label}</div>;
}

export function ErrorState({ error }) {
  return <div className="panel error">Error: {error.message}</div>;
}

export function EmptyState({ label }) {
  return <div className="panel muted">{label}</div>;
}

export function MetricCard({ label, value, hint }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "n/a"}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

export function StatusPill({ value }) {
  const normalized = (value || "unknown").toLowerCase();
  return <span className={`status-pill status-${normalized}`}>{value || "unknown"}</span>;
}

export function MarkdownPanel({ title, content }) {
  if (!content) {
    return null;
  }
  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="markdown-body">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </section>
  );
}

export function SessionTable({ items }) {
  return (
    <div className="panel table-panel">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Status</th>
            <th>Started</th>
            <th>Runs</th>
            <th>Best AUC</th>
            <th>Incumbent</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.session_id}>
              <td>
                <Link to={`/sessions/${item.session_id}`}>{item.session_id}</Link>
              </td>
              <td>
                <StatusPill value={item.status} />
              </td>
              <td>{item.started_at || "n/a"}</td>
              <td>{item.total_runs ?? "n/a"}</td>
              <td>{formatNumber(item.best_primary_metric_value)}</td>
              <td>{item.final_incumbent_run_id || item.incumbent_run_id || "n/a"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunTable({ sessionId, items }) {
  return (
    <div className="panel table-panel">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Role</th>
            <th>Status</th>
            <th>AUC</th>
            <th>Delta</th>
            <th>Train Seconds</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.run_id}>
              <td>
                <Link to={`/sessions/${sessionId}/runs/${item.run_id}`}>{item.run_id}</Link>
              </td>
              <td>{item.run_role || "n/a"}</td>
              <td>
                <div className="run-status-cell">
                  <StatusPill value={item.status} />
                  {item.is_current_incumbent ? <span className="minor-badge">current</span> : null}
                  {item.is_final_incumbent ? <span className="minor-badge">final</span> : null}
                </div>
              </td>
              <td>{formatNumber(item.weighted_cv_auc)}</td>
              <td>{formatSignedNumber(item.decision_delta)}</td>
              <td>{formatNumber(item.train_seconds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function KeyValueGrid({ title, entries }) {
  const filteredEntries = entries.filter((entry) => entry.value !== undefined);
  if (filteredEntries.length === 0) {
    return null;
  }

  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="key-value-grid">
        {filteredEntries.map((entry) => (
          <div key={entry.label} className="key-value-item">
            <div className="key-value-label">{entry.label}</div>
            <div className="key-value-value">{entry.value ?? "n/a"}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function JsonPanel({ title, value }) {
  if (!value) {
    return null;
  }
  return (
    <section className="panel">
      <h2>{title}</h2>
      <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Number(value).toFixed(4);
}

export function formatSignedNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  const numericValue = Number(value);
  return `${numericValue >= 0 ? "+" : ""}${numericValue.toFixed(4)}`;
}

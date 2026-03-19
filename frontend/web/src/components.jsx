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

export function ImprovementPlot({ items }) {
  const points = buildImprovementSeries(items);
  if (points.length === 0) {
    return null;
  }

  const width = 960;
  const height = 280;
  const padding = { top: 24, right: 24, bottom: 44, left: 64 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xMax = Math.max(...points.map((point) => point.x), 1);
  const yValues = points.map((point) => point.y);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const ySpan = Math.max(yMax - yMin, 0.001);
  const yFloor = Math.max(0, yMin - ySpan * 0.15);
  const yCeiling = Math.min(1, yMax + ySpan * 0.15);
  const adjustedSpan = Math.max(yCeiling - yFloor, 0.001);

  const scaleX = (value) => padding.left + (value / xMax) * innerWidth;
  const scaleY = (value) => padding.top + (1 - (value - yFloor) / adjustedSpan) * innerHeight;
  const pathData = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
    .join(" ");

  const yTicks = [yFloor, yFloor + adjustedSpan / 2, yCeiling];

  return (
    <section className="panel">
      <div className="plot-header">
        <div>
          <h2>Best Performance Over Time</h2>
          <p className="plot-caption">Only runs that improved the best weighted CV AUC are shown.</p>
        </div>
      </div>
      <div className="plot-frame">
        <svg viewBox={`0 0 ${width} ${height}`} className="improvement-plot" role="img" aria-label="Best performance over time">
          {yTicks.map((tick) => {
            const y = scaleY(tick);
            return (
              <g key={tick}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} className="plot-gridline" />
                <text x={padding.left - 12} y={y + 4} textAnchor="end" className="plot-axis-label">
                  {tick.toFixed(3)}
                </text>
              </g>
            );
          })}
          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            className="plot-axis"
          />
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={height - padding.bottom}
            className="plot-axis"
          />
          <path d={pathData} className="plot-line" />
          {points.map((point) => (
            <g key={point.label}>
              <circle cx={scaleX(point.x)} cy={scaleY(point.y)} r="5" className="plot-point" />
              <text x={scaleX(point.x)} y={height - padding.bottom + 22} textAnchor="middle" className="plot-axis-label">
                r{String(point.x).padStart(3, "0")}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
}

function buildImprovementSeries(items) {
  let bestValue = Number.NEGATIVE_INFINITY;
  return items
    .filter((item) => item.weighted_cv_auc !== null && item.weighted_cv_auc !== undefined)
    .reduce((series, item) => {
      const auc = Number(item.weighted_cv_auc);
      const x = Number(item.session_run_index);
      if (Number.isNaN(auc) || Number.isNaN(x)) {
        return series;
      }
      if (auc > bestValue) {
        bestValue = auc;
        series.push({
          x,
          y: auc,
          label: item.run_id,
        });
      }
      return series;
    }, []);
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

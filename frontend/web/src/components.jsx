import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";

export function LoadingState({ label = "Loading..." }) {
  return (
    <div className="state-panel">
      <div className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ error }) {
  return <div className="state-panel state-panel-error">Error: {error.message}</div>;
}

export function EmptyState({ label }) {
  return <div className="state-panel">{label}</div>;
}

export function SectionIntro({ eyebrow, title, description, actions }) {
  return (
    <section className="section-intro">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        {description ? <p className="section-intro-copy">{description}</p> : null}
      </div>
      {actions ? <div className="section-intro-actions">{actions}</div> : null}
    </section>
  );
}

export function Panel({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`panel ${className}`.trim()}>
      {title || subtitle || actions ? (
        <div className="panel-header">
          <div>
            {title ? <h2>{title}</h2> : null}
            {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
          </div>
          {actions ? <div className="panel-actions">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function MetricCard({ label, value, hint, tone = "default" }) {
  return (
    <div className={`metric-card metric-card-${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "n/a"}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

export function StatusPill({ value }) {
  const normalized = String(value || "unknown").toLowerCase();
  return <span className={`status-pill status-${normalized}`}>{value || "unknown"}</span>;
}

export function MinorBadge({ value }) {
  return <span className="minor-badge">{value}</span>;
}

export function MarkdownPanel({ title, content }) {
  if (!content) {
    return null;
  }

  return (
    <Panel title={title}>
      <div className="markdown-body">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </Panel>
  );
}

export function JsonPanel({ title, value }) {
  if (!value) {
    return null;
  }

  return (
    <Panel title={title}>
      <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
    </Panel>
  );
}

export function SessionTable({ items }) {
  return (
    <Panel title="Sessions" subtitle={`${items.length} session${items.length === 1 ? "" : "s"}`} className="table-panel">
      <DataTable
        columns={["Session", "Mode", "Status", "Started", "Runs", "Best", "Final Incumbent"]}
        rows={items.map((item) => (
          <tr key={item.session_id}>
            <td>
              <Link to={`/sessions/${item.session_id}`} className="table-primary-link">
                {item.session_id}
              </Link>
            </td>
            <td>
              <span className="mono-chip">{item.experiment_mode || "standard"}</span>
            </td>
            <td>
              <StatusPill value={item.status} />
            </td>
            <td>{formatDateTime(item.started_at)}</td>
            <td>{item.total_runs ?? "n/a"}</td>
            <td>{formatNumber(item.best_primary_metric_value)}</td>
            <td>
              <span className="mono-text">{item.final_incumbent_run_id || item.incumbent_run_id || "n/a"}</span>
            </td>
          </tr>
        ))}
      />
    </Panel>
  );
}

export function RunTable({ sessionId, items }) {
  return (
    <Panel title="Runs" subtitle={`${items.length} recorded run${items.length === 1 ? "" : "s"}`} className="table-panel">
      <DataTable
        columns={["Run", "Role", "Status", "Primary Metric", "Delta", "Train Seconds", "Parameters"]}
        rows={items.map((item) => (
          <tr key={item.run_id}>
            <td>
              <Link to={`/sessions/${sessionId}/runs/${item.run_id}`} className="table-primary-link mono-text">
                {item.run_id}
              </Link>
            </td>
            <td>
              <div className="inline-badges">
                <MinorBadge value={item.run_role || "n/a"} />
                {item.is_current_incumbent ? <MinorBadge value="current" /> : null}
                {item.is_final_incumbent ? <MinorBadge value="final" /> : null}
              </div>
            </td>
            <td>
              <StatusPill value={item.status} />
            </td>
            <td>{formatNumber(item.primary_metric_value)}</td>
            <td>{formatSignedNumber(item.decision_delta)}</td>
            <td>{formatCompactNumber(item.train_seconds)}</td>
            <td>{formatInteger(item.num_params)}</td>
          </tr>
        ))}
      />
    </Panel>
  );
}

export function KeyValueGrid({ title, entries, compact = false }) {
  const filteredEntries = entries.filter((entry) => entry.value !== undefined && entry.value !== null && entry.value !== "");
  if (filteredEntries.length === 0) {
    return null;
  }

  return (
    <Panel title={title}>
      <div className={`key-value-grid${compact ? " key-value-grid-compact" : ""}`}>
        {filteredEntries.map((entry) => (
          <div key={entry.label} className="key-value-item">
            <div className="key-value-label">{entry.label}</div>
            <div className={`key-value-value${entry.monospace ? " mono-text" : ""}`}>{entry.value}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function ListPanel({ title, items, emptyLabel = "No items recorded." }) {
  return (
    <Panel title={title}>
      {items && items.length > 0 ? (
        <ul className="bullet-list">
          {items.map((item, index) => (
            <li key={`${title}-${index}`}>{renderListItem(item)}</li>
          ))}
        </ul>
      ) : (
        <p className="empty-copy">{emptyLabel}</p>
      )}
    </Panel>
  );
}

export function MetricRows({ title, metrics }) {
  const entries = Object.entries(metrics || {})
    .map(([key, value]) => ({ key, value }))
    .filter(({ value }) => value && value.current_value !== undefined && value.current_value !== null)
    .sort((left, right) => left.value.display_name.localeCompare(right.value.display_name));

  if (entries.length === 0) {
    return null;
  }

  return (
    <Panel title={title} className="table-panel">
      <DataTable
        columns={["Metric", "Current", "Vs Compared", "Vs Base", "Direction"]}
        rows={entries.map(({ key, value }) => (
          <tr key={key}>
            <td>{value.display_name || key}</td>
            <td>{formatMetricValue(value.current_value)}</td>
            <td>{formatSignedMetricValue(value.delta_vs_compared)}</td>
            <td>{formatSignedMetricValue(value.delta_vs_base)}</td>
            <td>{value.direction || "n/a"}</td>
          </tr>
        ))}
      />
    </Panel>
  );
}

export function ImprovementPlot({ items }) {
  const points = buildImprovementSeries(items);
  if (points.length === 0) {
    return null;
  }

  const width = 920;
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 44, left: 60 };
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
    <Panel title="Incumbent Progression" subtitle="Points appear only when a run established a new session best.">
      <div className="plot-frame">
        <svg viewBox={`0 0 ${width} ${height}`} className="improvement-plot" role="img" aria-label="Session incumbent progression">
          {yTicks.map((tick) => {
            const y = scaleY(tick);
            return (
              <g key={tick}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} className="plot-gridline" />
                <text x={padding.left - 10} y={y + 4} textAnchor="end" className="plot-axis-label">
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
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="plot-axis" />
          <path d={pathData} className="plot-line" />
          {points.map((point) => (
            <g key={point.label}>
              <circle cx={scaleX(point.x)} cy={scaleY(point.y)} r="4.5" className="plot-point" />
              <text x={scaleX(point.x)} y={height - padding.bottom + 18} textAnchor="middle" className="plot-axis-label">
                r{String(point.x).padStart(3, "0")}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Panel>
  );
}

function DataTable({ columns, rows }) {
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  );
}

function buildImprovementSeries(items) {
  const sortedItems = [...items].sort((left, right) => (left.session_run_index ?? 0) - (right.session_run_index ?? 0));
  const keptItems = sortedItems.filter((item) => item.status === "keep");
  const sourceItems = keptItems.length > 0 ? keptItems : sortedItems;
  let bestValue = Number.NEGATIVE_INFINITY;

  return sourceItems
    .sort((left, right) => (left.session_run_index ?? 0) - (right.session_run_index ?? 0))
    .filter((item) => item.primary_metric_value !== null && item.primary_metric_value !== undefined)
    .reduce((series, item) => {
      const metricValue = Number(item.primary_metric_value);
      const x = Number(item.session_run_index);
      if (Number.isNaN(metricValue) || Number.isNaN(x)) {
        return series;
      }
      if (metricValue > bestValue) {
        bestValue = metricValue;
        series.push({ x, y: metricValue, label: item.run_id });
      }
      return series;
    }, []);
}

function renderListItem(item) {
  if (typeof item === "string" || typeof item === "number") {
    return item;
  }

  if (item && typeof item === "object") {
    const parts = Object.entries(item)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([key, value]) => `${humanizeKey(key)}: ${value}`);
    return parts.join(" | ");
  }

  return String(item);
}

function humanizeKey(value) {
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(4);
}

export function formatCompactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(2);
}

export function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Number(value));
}

export function formatSignedNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  const numericValue = Number(value);
  return `${numericValue >= 0 ? "+" : ""}${numericValue.toFixed(4)}`;
}

export function formatDateTime(value) {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

export function formatMetricValue(value) {
  if (typeof value === "number") {
    return Math.abs(value) >= 100 ? formatInteger(value) : value.toFixed(4);
  }
  return value ?? "n/a";
}

export function formatSignedMetricValue(value) {
  if (typeof value === "number") {
    return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
  }
  return "n/a";
}

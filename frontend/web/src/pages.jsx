import { Link, useParams } from "react-router-dom";
import { getRun, getSession, getSessions } from "./api";
import {
  EmptyState,
  ErrorState,
  ImprovementPlot,
  JsonPanel,
  KeyValueGrid,
  ListPanel,
  LoadingState,
  MarkdownPanel,
  MetricCard,
  MetricRows,
  Panel,
  RunTable,
  SectionIntro,
  SessionTable,
  StatusPill,
  formatCompactNumber,
  formatDateTime,
  formatInteger,
  formatNumber,
} from "./components";
import { useAsyncResource } from "./hooks";

export function SessionListPage() {
  const { data, error, loading } = useAsyncResource(() => getSessions(), []);

  if (loading) {
    return <LoadingState label="Loading sessions..." />;
  }
  if (error) {
    return <ErrorState error={error} />;
  }

  const items = data?.items || [];
  if (items.length === 0) {
    return <EmptyState label="No sessions were found under the mounted sessions directory." />;
  }

  const completedCount = items.filter((item) => item.status === "completed").length;
  const totalRuns = items.reduce((count, item) => count + (item.total_runs || 0), 0);
  const comparativeCount = items.filter((item) => item.experiment_mode === "comparative").length;

  return (
    <div className="page-stack">
      <SectionIntro
        eyebrow="Session Viewer"
        title="Autonomous Search Sessions"
        description="Browse saved orchestration runs, compare primary metrics, and drill into per-run analysis artifacts."
      />
      <section className="metrics-grid">
        <MetricCard label="Sessions" value={formatInteger(items.length)} />
        <MetricCard label="Completed" value={formatInteger(completedCount)} />
        <MetricCard label="Recorded Runs" value={formatInteger(totalRuns)} />
        <MetricCard label="Comparative Sessions" value={formatInteger(comparativeCount)} />
      </section>
      <SessionTable items={items} />
    </div>
  );
}

export function SessionDetailPage() {
  const { sessionId } = useParams();
  const { data, error, loading } = useAsyncResource(() => getSession(sessionId), [sessionId]);

  if (loading) {
    return <LoadingState label={`Loading session ${sessionId}...`} />;
  }
  if (error) {
    return <ErrorState error={error} />;
  }
  if (!data) {
    return <EmptyState label="Session data was not available." />;
  }

  const context = data.context || {};
  const summary = data.summary || {};
  const state = data.state || {};
  const runs = data.runs || [];
  const primaryMetricName = context.experiment_mode === "comparative" ? "weighted_cv_overall_auc" : "weighted_cv_auc";
  const bestMetric = summary.best_primary_metric_value ?? state.incumbent_primary_metric_value;
  const bestRunId = summary.best_run_id || state.final_incumbent_run_id || state.incumbent_run_id;
  const featureCount = Array.isArray(context.feature_names) ? context.feature_names.length : null;

  return (
    <div className="page-stack">
      <div className="page-actions">
        <Link to="/" className="action-link">
          All Sessions
        </Link>
      </div>
      <SectionIntro
        eyebrow="Session"
        title={sessionId}
        description={state.objective || context.objective || "No objective recorded."}
        actions={<StatusPill value={state.status || summary.status} />}
      />
      <section className="metrics-grid">
        <MetricCard label="Primary Metric" value={primaryMetricName} hint={context.experiment_mode || "standard"} tone="wide" />
        <MetricCard label="Best Score" value={formatNumber(bestMetric)} />
        <MetricCard label="Total Runs" value={formatInteger(summary.total_runs ?? runs.length)} />
        <MetricCard label="Final Incumbent" value={bestRunId || "n/a"} hint="Run id" tone="wide" />
      </section>
      <div className="detail-grid">
        <KeyValueGrid
          title="Session Overview"
          compact
          entries={[
            { label: "Started", value: formatDateTime(state.started_at || summary.started_at) },
            { label: "Completed", value: formatDateTime(state.completed_at || summary.completed_at) },
            { label: "Duration (s)", value: formatCompactNumber(summary.duration_seconds ?? state.duration_seconds) },
            { label: "Initiated By", value: state.initiated_by || context.initiated_by || summary.initiated_by },
            { label: "Experiment Mode", value: context.experiment_mode },
            { label: "Base Run", value: context.base_run_id || summary.base_run_id, monospace: true },
            { label: "Recommended Next Start", value: summary.recommended_next_starting_run_id, monospace: true },
            { label: "Dataset", value: context.raw_data_path || context.dataset_config_payload?.raw_data_path },
            { label: "Feature Count", value: featureCount !== null ? formatInteger(featureCount) : null },
          ]}
        />
        <KeyValueGrid
          title="Run Outcomes"
          compact
          entries={[
            { label: "Keep Count", value: formatInteger(summary.keep_count ?? state.keep_count) },
            { label: "Discard Count", value: formatInteger(summary.discard_count ?? state.discard_count) },
            { label: "Crash Count", value: formatInteger(summary.crash_count ?? state.crash_count) },
            { label: "Rerun Count", value: formatInteger(summary.rerun_count ?? state.rerun_count) },
            { label: "Delta From Base", value: formatNumber(summary.delta_from_base) },
            { label: "Latest Run", value: state.latest_run_id, monospace: true },
          ]}
        />
      </div>
      <ImprovementPlot items={runs} />
      <div className="detail-grid">
        <ListPanel title="Patterns That Helped" items={summary.patterns_that_helped || []} />
        <ListPanel title="Patterns That Hurt" items={summary.patterns_that_hurt || []} />
      </div>
      <div className="detail-grid">
        <ListPanel title="Promising Near Misses" items={summary.most_promising_near_misses || []} />
        <ListPanel title="Recommended Next Mutations" items={summary.recommended_next_mutations || []} />
      </div>
      <div className="detail-grid">
        <ListPanel title="Hypotheses Supported" items={summary.hypotheses_supported || []} />
        <ListPanel title="Open Questions" items={summary.unresolved_questions || []} />
      </div>
      <Panel title="Search Coverage" subtitle="Values derived from the current session summary record.">
        <div className="search-coverage-grid">
          {Object.entries(summary.search_space_coverage || {}).map(([key, value]) => (
            <div key={key} className="coverage-item">
              <div className="key-value-label">{key.replace(/_/g, " ")}</div>
              <div className="key-value-value">
                {Array.isArray(value) ? value.join(", ") : typeof value === "number" ? formatInteger(value) : String(value)}
              </div>
            </div>
          ))}
        </div>
      </Panel>
      <RunTable sessionId={sessionId} items={runs} />
      <MarkdownPanel title="Session Summary" content={data.summary_markdown} />
      <JsonPanel title="Session Summary JSON" value={summary} />
      <JsonPanel title="Session State JSON" value={state} />
      <JsonPanel title="Session Context JSON" value={context} />
    </div>
  );
}

export function RunDetailPage() {
  const { sessionId, runId } = useParams();
  const { data, error, loading } = useAsyncResource(() => getRun(sessionId, runId), [sessionId, runId]);

  if (loading) {
    return <LoadingState label={`Loading run ${runId}...`} />;
  }
  if (error) {
    return <ErrorState error={error} />;
  }
  if (!data) {
    return <EmptyState label="Run data was not available." />;
  }

  const summaryPayload = data.summary || {};
  const summary = summaryPayload.summary || {};
  const architecture = summaryPayload.architecture || {};
  const datasetConfig = summaryPayload.dataset_config || {};
  const diagnostics = summaryPayload.diagnostics || {};
  const decision = data.decision || {};
  const runContext = data.run_context || {};
  const analysisInput = data.analysis_input || {};
  const agentAnalysis = data.agent_analysis || {};

  return (
    <div className="page-stack">
      <div className="page-actions">
        <Link to="/" className="action-link">
          All Sessions
        </Link>
        <Link to={`/sessions/${sessionId}`} className="action-link">
          Session {sessionId}
        </Link>
      </div>
      <SectionIntro
        eyebrow="Run"
        title={runId}
        description={runContext.description || "No run description recorded."}
        actions={<StatusPill value={decision.decision_status || "unknown"} />}
      />
      <section className="metrics-grid">
        <MetricCard label="Primary Metric" value={formatNumber(data.primary_metric_value)} hint={data.primary_metric_name || "n/a"} />
        <MetricCard label="Train Seconds" value={formatCompactNumber(summary.train_seconds)} />
        <MetricCard label="Parameters" value={formatInteger(summary.num_params)} />
        <MetricCard label="Compared Run" value={runContext.compared_against_run_id || "n/a"} tone="wide" />
      </section>
      <div className="detail-grid">
        <KeyValueGrid
          title="Run Overview"
          compact
          entries={[
            { label: "Run Role", value: runContext.run_role },
            { label: "Started", value: formatDateTime(runContext.started_at) },
            { label: "Completed", value: formatDateTime(runContext.completed_at) },
            { label: "Duration (s)", value: formatCompactNumber(runContext.duration_seconds) },
            { label: "Parent Run", value: runContext.parent_run_id, monospace: true },
            { label: "Best Known At Start", value: runContext.best_known_run_id_at_start, monospace: true },
            { label: "Git Branch", value: runContext.git_branch },
            { label: "Git Commit", value: runContext.git_commit, monospace: true },
          ]}
        />
        <KeyValueGrid
          title="Decision"
          compact
          entries={[
            { label: "Status", value: decision.decision_status },
            { label: "Metric Name", value: decision.decision_metric_name },
            { label: "Metric Value", value: formatNumber(decision.decision_metric_value) },
            { label: "Baseline", value: formatNumber(decision.decision_baseline_value) },
            { label: "Delta", value: formatNumber(decision.decision_delta) },
            { label: "Hypothesis Result", value: decision.hypothesis_result },
            { label: "Reason", value: decision.decision_reason },
          ]}
        />
      </div>
      <div className="detail-grid">
        <Panel title="Hypothesis">
          <p className="narrative-block">{runContext.hypothesis || "No hypothesis recorded."}</p>
        </Panel>
        <Panel title="Mutation Summary">
          <p className="narrative-block">{runContext.mutation_summary || "No mutation summary recorded."}</p>
        </Panel>
      </div>
      <MetricRows title="Analysis Metrics" metrics={analysisInput.metrics} />
      <div className="detail-grid">
        <ListPanel title="Rule-Based Interpretation" items={analysisInput.rule_based_interpretation || []} />
        <Panel title="Agent Analysis">
          <div className="analysis-stack">
            <div>
              <div className="key-value-label">Summary Label</div>
              <p className="narrative-block">{agentAnalysis.summary_label || "n/a"}</p>
            </div>
            <div>
              <div className="key-value-label">Freeform Analysis</div>
              <p className="narrative-block">{agentAnalysis.freeform_analysis || "No analysis recorded."}</p>
            </div>
            <div>
              <div className="key-value-label">Next Step Reasoning</div>
              <p className="narrative-block">{agentAnalysis.next_step_reasoning || "No recommendation recorded."}</p>
            </div>
          </div>
        </Panel>
      </div>
      <div className="detail-grid">
        <ListPanel title="Likely Helped" items={agentAnalysis.likely_helped || []} />
        <ListPanel title="Likely Hurt" items={agentAnalysis.likely_hurt || []} />
      </div>
      <div className="detail-grid">
        <KeyValueGrid
          title="Architecture"
          compact
          entries={Object.entries(architecture).map(([key, value]) => ({
            label: key.replace(/_/g, " "),
            value: Array.isArray(value) ? value.join(", ") || "[]" : String(value),
          }))}
        />
        <KeyValueGrid
          title="Diagnostics"
          compact
          entries={Object.entries(diagnostics).map(([key, value]) => ({
            label: key.replace(/_/g, " "),
            value: typeof value === "object" ? JSON.stringify(value) : String(value),
          }))}
        />
      </div>
      <KeyValueGrid
        title="Dataset Config"
        compact
        entries={Object.entries(datasetConfig).map(([key, value]) => ({
          label: key.replace(/_/g, " "),
          value: Array.isArray(value) ? value.join(", ") || "[]" : String(value),
        }))}
      />
      <MarkdownPanel title="Run Synopsis" content={data.synopsis_markdown} />
      <JsonPanel title="Run Summary JSON" value={summaryPayload} />
      <JsonPanel title="Analysis Input JSON" value={analysisInput} />
      <JsonPanel title="Agent Analysis JSON" value={agentAnalysis} />
    </div>
  );
}

import { Link, useParams } from "react-router-dom";
import { getRun, getSession, getSessions } from "./api";
import {
  EmptyState,
  ErrorState,
  JsonPanel,
  KeyValueGrid,
  LoadingState,
  MarkdownPanel,
  MetricCard,
  RunTable,
  SessionTable,
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

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <p className="eyebrow">Overview</p>
        <h1>Sessions</h1>
        <p>Browse recent experiment sessions and drill into their saved run artifacts.</p>
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

  const summary = data.summary || {};
  const state = data.state || {};
  const runs = data.runs || [];

  return (
    <div className="page-stack">
      <div className="page-actions">
        <Link to="/">Back to sessions</Link>
      </div>
      <section className="hero-panel">
        <p className="eyebrow">Session</p>
        <h1>{sessionId}</h1>
        <p>{state.objective || summary.objective || "No objective recorded."}</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Status" value={state.status || summary.status || "n/a"} />
        <MetricCard label="Best AUC" value={formatNumber(summary.best_primary_metric_value)} />
        <MetricCard label="Total Runs" value={summary.total_runs ?? runs.length} />
        <MetricCard label="Best Run" value={summary.best_run_id || state.incumbent_run_id || "n/a"} />
      </section>
      <KeyValueGrid
        title="Session Metadata"
        entries={[
          { label: "Started At", value: state.started_at || summary.started_at },
          { label: "Completed At", value: state.completed_at || summary.completed_at },
          { label: "Initiated By", value: state.initiated_by || summary.initiated_by },
          { label: "Keep Count", value: summary.keep_count ?? state.keep_count },
          { label: "Discard Count", value: summary.discard_count ?? state.discard_count },
          { label: "Crash Count", value: summary.crash_count ?? state.crash_count },
          { label: "Final Incumbent", value: summary.final_incumbent_run_id || state.final_incumbent_run_id },
          { label: "Recommended Next Start", value: summary.recommended_next_starting_run_id },
        ]}
      />
      <MarkdownPanel title="Session Summary" content={data.summary_markdown} />
      <section>
        <h2>Runs</h2>
        <RunTable sessionId={sessionId} items={runs} />
      </section>
      <JsonPanel title="Session Summary JSON" value={summary} />
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

  const summary = data.summary?.summary || {};
  const architecture = data.summary?.architecture || {};
  const decision = data.decision || {};
  const runContext = data.run_context || {};

  return (
    <div className="page-stack">
      <div className="page-actions">
        <Link to="/">Back to sessions</Link>
        <Link to={`/sessions/${sessionId}`}>Back to session</Link>
      </div>
      <section className="hero-panel">
        <p className="eyebrow">Run</p>
        <h1>{runId}</h1>
        <p>{runContext.description || "No run description recorded."}</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Decision" value={decision.decision_status || "n/a"} />
        <MetricCard label="AUC" value={formatNumber(summary.primary_metric_value)} />
        <MetricCard label="Train Seconds" value={formatNumber(summary.train_seconds)} />
        <MetricCard label="Params" value={summary.num_params ?? "n/a"} />
      </section>
      <KeyValueGrid
        title="Run Metadata"
        entries={[
          { label: "Role", value: runContext.run_role },
          { label: "Started At", value: runContext.started_at },
          { label: "Completed At", value: runContext.completed_at },
          { label: "Compared Against", value: runContext.compared_against_run_id },
          { label: "Parent Run", value: runContext.parent_run_id },
          { label: "Hypothesis", value: runContext.hypothesis },
          { label: "Mutation Summary", value: runContext.mutation_summary },
          { label: "Decision Reason", value: decision.decision_reason },
        ]}
      />
      <MarkdownPanel title="Run Synopsis" content={data.synopsis_markdown} />
      <JsonPanel title="Architecture" value={architecture} />
      <JsonPanel title="Decision JSON" value={decision} />
      <JsonPanel title="Summary JSON" value={data.summary} />
      <JsonPanel title="Analysis Input JSON" value={data.analysis_input} />
      <JsonPanel title="Agent Analysis JSON" value={data.agent_analysis} />
    </div>
  );
}

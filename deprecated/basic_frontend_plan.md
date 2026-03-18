# Basic Frontend Plan

## Objective Analysis

**What:** Add a very small, Dockerized read-only web UI for browsing session and run results generated under `sessions/`.

**Why:** The repository already records structured session artifacts, but they are currently easiest to inspect from the filesystem. A simple web UI would make it faster to compare sessions, inspect run outcomes, and read summaries without changing the experiment workflow.

**Who:** Anyone running or reviewing search sessions locally.

**Success looks like:** A user can start the stack with Docker, open a browser, see a list of sessions, drill into a session, inspect its runs, and read the saved summary/synopsis artifacts without editing or triggering experiments from the UI.

## Scope

### In scope

- Read-only frontend for `sessions/<session_id>/...`
- Session list view
- Session detail view
- Run detail view
- Docker-based local development and startup
- Thin API service that reads existing JSON/TSV/Markdown artifacts

### Out of scope

- Editing sessions or runs
- Starting, stopping, or retrying experiments from the browser
- Authentication or multi-user access control
- Live log streaming
- General dataset exploration
- Replacing the CLI workflow

## Existing Data Model

The current repository already provides almost everything needed for a read-only viewer:

- `sessions/<session_id>/session_context.json`
- `sessions/<session_id>/session_state.json`
- `sessions/<session_id>/results.tsv`
- `sessions/<session_id>/session_summary.json`
- `sessions/<session_id>/session_summary.md`
- `sessions/<session_id>/runs/<run_dir>/run_context.json`
- `sessions/<session_id>/runs/<run_dir>/decision.json`
- `sessions/<session_id>/runs/<run_dir>/analysis_input.json`
- `sessions/<session_id>/runs/<run_dir>/agent_analysis.json`
- `sessions/<session_id>/runs/<run_dir>/summary.json`
- `sessions/<session_id>/runs/<run_dir>/synopsis.md`

This means the frontend does not need to talk to the training harness directly. It only needs a stable read layer over files that already exist.

## Recommended Architecture

### Overview

Use a two-container local stack:

1. `api`
   - Small Python service, preferably FastAPI
   - Reads `sessions/` from a mounted volume
   - Normalizes JSON, TSV, and Markdown into frontend-friendly responses

2. `web`
   - Small React frontend, preferably Vite + React
   - Calls the API over HTTP
   - Renders tables, status badges, and Markdown summaries

### Why this shape

- Keeps the browser away from direct filesystem parsing
- Reuses the repo's Python environment and artifact knowledge on the backend
- Avoids forcing the frontend to parse TSV and repository-specific directory layouts
- Makes it easier to later add filtering, pagination, or live refresh

## Docker Plan

### Development setup

Use `docker compose` with:

- `api` container mounting the repo read-only for `sessions/`
- `web` container for the frontend dev server
- Shared network between `web` and `api`

### Container responsibilities

**API container**

- Install Python dependencies
- Serve read-only endpoints on a local port such as `8000`
- Mount:
  - repo source
  - `sessions/`

**Web container**

- Install Node dependencies
- Serve the frontend on a local port such as `3000` or `5173`
- Use `VITE_API_BASE_URL` or equivalent to point to the API container

### Production-lite local option

After the first version works in dev mode, add a simple production path:

- build static frontend assets
- serve them from Nginx or from the API service itself

That second step is optional for the first milestone. The initial goal is just a reliable local Docker workflow.

## UI Plan

### 1. Session List Page

Purpose: show all sessions in one place.

Display:

- session ID
- status
- objective
- started/completed timestamps
- total runs
- incumbent run ID
- best primary metric value
- keep/discard/crash counts

Useful behavior:

- sort by most recent session
- filter by status
- click into a session

### 2. Session Detail Page

Purpose: summarize one session and show its runs.

Display:

- top summary card from `session_summary.json`
- rendered `session_summary.md`
- run table sourced from `results.tsv`
- incumbent progression if available
- hypotheses supported vs unsupported

Useful behavior:

- sort runs by index or metric
- highlight kept/incumbent runs
- click into a run

### 3. Run Detail Page

Purpose: inspect a single run without opening files manually.

Display:

- run metadata from `run_context.json`
- decision metadata from `decision.json`
- key metrics from `summary.json`
- rendered `synopsis.md`
- optional raw JSON tabs for debugging

Useful behavior:

- show comparison baseline run ID
- highlight metric delta vs baseline
- expose hypothesis, mutation summary, and description clearly

## Minimal API Surface

The backend should stay narrow and map directly to UI needs.

### Suggested endpoints

- `GET /health`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/runs`
- `GET /sessions/{session_id}/runs/{run_id}`

### Response expectations

`GET /sessions`
- Return one normalized item per session
- Read from `session_state.json` and `session_summary.json` where present

`GET /sessions/{session_id}`
- Return merged session context, state, summary, and rendered/raw markdown content

`GET /sessions/{session_id}/runs`
- Return run rows derived from `results.tsv`, enriched with run role and decision status

`GET /sessions/{session_id}/runs/{run_id}`
- Return merged `run_context.json`, `decision.json`, `summary.json`, `analysis_input.json`, `agent_analysis.json`, and `synopsis.md`

## Implementation Phases

### Phase 1: File-reading API

Deliverables:

- backend app skeleton
- session discovery over `sessions/`
- normalized session list endpoint
- normalized run detail endpoint
- simple schema validation and error handling

Acceptance criteria:

- API starts in Docker
- API returns data for existing sessions
- Missing optional files do not crash the service

### Phase 2: Basic frontend

Deliverables:

- session list page
- session detail page
- run detail page
- Markdown rendering for summaries

Acceptance criteria:

- user can navigate from sessions to runs
- empty states render correctly when no sessions exist
- error state renders when API is unavailable

### Phase 3: Docker developer workflow

Deliverables:

- `docker-compose.yml`
- `Dockerfile` for API
- `Dockerfile` for web
- short README section for startup

Acceptance criteria:

- `docker compose up --build` starts both services
- frontend can load data from API inside Docker
- repo sessions are visible through the mounted volume

### Phase 4: Polish

Deliverables:

- filtering and sorting
- auto-refresh button or polling
- raw JSON inspection panel

Acceptance criteria:

- user can quickly find the latest session
- user can inspect a run without opening local files

## Risks And Decisions

### Risk: artifact shape drift

The frontend depends on files written by the session manager. If artifact fields change, the UI can break unless the API isolates those changes.

Mitigation:

- keep parsing logic in the API, not the frontend
- treat some files as optional
- return normalized response models

### Risk: markdown and JSON inconsistency

Some information appears in both structured JSON and Markdown summaries. Those can diverge in tone or detail.

Mitigation:

- use JSON for status/metrics/tables
- use Markdown only for narrative summary display

### Risk: overbuilding the first version

The repository does not need a full experiment platform yet.

Mitigation:

- keep the first release read-only
- avoid auth, write actions, and live orchestration
- optimize for local use only

## Recommended Tech Choices

### Backend

- FastAPI
- Pydantic response models
- standard library TSV and JSON parsing
- Markdown passthrough as raw text or pre-rendered HTML

### Frontend

- React
- Vite
- TanStack Query for API fetching
- React Router
- a lightweight table library only if needed

These are suggestions, not requirements. The important part is keeping the stack small and easy to run in Docker.

## Proposed Directory Additions

```text
frontend/
  web/
    Dockerfile
    package.json
    src/
  api/
    Dockerfile
    app/
      main.py
      schemas.py
      readers.py
docker-compose.yml
```

## First Milestone Definition

The first milestone should be considered complete when:

- Docker starts a backend and frontend locally
- the session list loads from real repository data
- a user can open one session and one run
- the UI shows status, metrics, and saved Markdown summaries
- no browser action mutates repository state

## Suggested Next Step

Implement the API first. The repository already has a clear file contract, and a stable API boundary will make the frontend much easier to build and change safely.

import { Link, Route, Routes } from "react-router-dom";
import { RunDetailPage, SessionDetailPage, SessionListPage } from "./pages";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <div>
            <p className="eyebrow">sirchml-autoresearch</p>
            <Link to="/" className="app-title">
              Session Viewer
            </Link>
            <p className="app-subtitle">Internal read-only explorer for autonomous session and run artifacts.</p>
          </div>
          <nav className="app-nav" aria-label="Primary">
            <Link to="/" className="nav-link">
              Sessions
            </Link>
          </nav>
        </div>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<SessionListPage />} />
          <Route path="/sessions/:sessionId" element={<SessionDetailPage />} />
          <Route path="/sessions/:sessionId/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}

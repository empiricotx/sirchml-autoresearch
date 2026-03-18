import { Link, Route, Routes } from "react-router-dom";
import { RunDetailPage, SessionDetailPage, SessionListPage } from "./pages";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">sirchml-autoresearch</p>
          <Link to="/" className="app-title">
            Session Viewer
          </Link>
        </div>
        <p className="app-subtitle">Read-only browser for session and run artifacts.</p>
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

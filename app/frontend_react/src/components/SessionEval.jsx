/**
 * Full session evaluation display.
 */

const DIRECTION_ICONS = {
  improving:         "📈 IMPROVING",
  declining:         "📉 DECLINING",
  stable:            "➡️  STABLE",
  insufficient_data: "📊 INSUFFICIENT DATA",
};

export default function SessionEval({ data, onClose }) {
  if (!data) return null;

  const maxScore = Math.max(...(data.score_trend || [1]));

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 1000, padding: "1rem",
    }}>
      <div style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        borderRadius: 12, padding: "2rem",
        maxWidth: 700, width: "100%",
        maxHeight: "90vh", overflowY: "auto",
      }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between",
          alignItems: "center", marginBottom: "1.5rem" }}>
          <h2 style={{ fontSize: "1.3rem" }}>📊 Session Evaluation</h2>
          <button className="btn btn-secondary"
            style={{ padding: "0.4rem 0.8rem" }}
            onClick={onClose}>
            ✕ Close
          </button>
        </div>

        {/* Key metrics */}
        <div className="grid-4" style={{ marginBottom: "1.5rem" }}>
          <div className="metric-box">
            <div className="metric-value" style={{ fontSize: "0.95rem" }}>
              {DIRECTION_ICONS[data.user_improvement] || "—"}
            </div>
            <div className="metric-label">Progress</div>
          </div>
          <div className="metric-box">
            <div className="metric-value">{data.avg_score_first_half?.toFixed(1)}</div>
            <div className="metric-label">First Half</div>
          </div>
          <div className="metric-box">
            <div className="metric-value">{data.avg_score_second_half?.toFixed(1)}</div>
            <div className="metric-label">Second Half</div>
          </div>
          <div className="metric-box">
            <div className="metric-value">{data.overall_grade?.toUpperCase()}</div>
            <div className="metric-label">Grade</div>
          </div>
        </div>

        {/* Score trend chart (simple SVG bars) */}
        {data.score_trend?.length > 0 && (
          <div className="card" style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)",
              marginBottom: "0.75rem", textTransform: "uppercase",
              letterSpacing: "0.06em" }}>
              Score Trend
            </div>
            <div style={{ display: "flex", alignItems: "flex-end",
              gap: "0.5rem", height: 80 }}>
              {data.score_trend.map((s, i) => (
                <div key={i} style={{ flex: 1, display: "flex",
                  flexDirection: "column", alignItems: "center", gap: 4 }}>
                  <div style={{
                    height: `${(s / 10) * 70}px`,
                    background: s >= 7
                      ? "var(--accent-green)"
                      : s >= 4
                      ? "var(--accent-orange)"
                      : "var(--accent-red)",
                    borderRadius: "3px 3px 0 0",
                    width: "100%",
                    minHeight: 4,
                    transition: "height 0.3s ease",
                  }} />
                  <span style={{ fontSize: "0.65rem",
                    color: "var(--text-secondary)" }}>
                    T{i + 1}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Breakdown */}
        <div className="grid-2" style={{ marginBottom: "1rem" }}>
          <div className="card" style={{ padding: "1rem" }}>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)",
              marginBottom: "0.5rem" }}>
              Argument Breakdown
            </div>
            <div style={{ fontSize: "0.9rem", lineHeight: 2 }}>
              <div>✅ Strong: <strong>{data.strong_count}</strong></div>
              <div>⚠️ Weak: <strong>{data.weak_count}</strong></div>
              <div>❌ Fallacies: <strong>{data.fallacy_count}</strong></div>
              <div>🏆 Best Turn: <strong>Turn {data.best_turn}</strong></div>
              <div>📉 Worst Turn: <strong>Turn {data.worst_turn}</strong></div>
            </div>
          </div>

          <div className="card" style={{ padding: "1rem" }}>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)",
              marginBottom: "0.5rem" }}>
              Cost Summary
            </div>
            <div style={{ fontSize: "0.9rem", lineHeight: 2 }}>
              <div>Tokens: <strong>{(data.total_tokens || 0).toLocaleString()}</strong></div>
              <div>Cost: <strong>${(data.total_cost_usd || 0).toFixed(6)}</strong></div>
              <div>Cache hits: <strong>{data.cache_hits}</strong></div>
            </div>
          </div>
        </div>

        {/* Coaching advice */}
        {data.improvement_advice && (
          <div className="alert alert-success">
            <strong>🎯 Coaching Advice</strong>
            <p style={{ marginTop: "0.4rem", lineHeight: 1.6 }}>
              {data.improvement_advice}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
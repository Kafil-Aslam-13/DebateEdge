/**
 * Argument quality + score display.
 */

const QUALITY_CONFIG = {
  strong:  { icon: "✅", color: "var(--accent-green)",  tagClass: "tag-green",  label: "STRONG"  },
  weak:    { icon: "⚠️", color: "var(--accent-orange)", tagClass: "tag-orange", label: "WEAK"    },
  fallacy: { icon: "❌", color: "var(--accent-red)",    tagClass: "tag-red",    label: "FALLACY" },
};

const GRADE_STARS = {
  excellent: "★★★★",
  good:      "★★★☆",
  average:   "★★☆☆",
  poor:      "★☆☆☆",
};

export default function ScoreCard({ result }) {
  const quality   = result.argument_quality || "";
  const score     = result.argument_score   || 0;
  const breakdown = result.score_breakdown  || {};
  const reasoning = result.quality_reasoning || "";
  const cfg       = QUALITY_CONFIG[quality] || {};
  const evalData  = result.evaluation       || {};

  return (
    <div className="card">
      <h3 style={{ marginBottom: "1rem", fontSize: "0.95rem",
        textTransform: "uppercase", letterSpacing: "0.06em",
        color: "var(--text-secondary)" }}>
        Your Argument
      </h3>

      {/* Quality + overall score */}
      <div style={{ display: "flex", alignItems: "center",
        gap: "1rem", marginBottom: "1rem" }}>
        <span style={{ fontSize: "1.5rem" }}>{cfg.icon}</span>
        <span className={`tag ${cfg.tagClass}`} style={{ fontSize: "0.85rem" }}>
          {cfg.label}
        </span>
        <span style={{ fontSize: "1.4rem", fontWeight: 600,
          fontFamily: "JetBrains Mono", color: "var(--accent-blue)" }}>
          {score}/10
        </span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${score * 10}%`,
            background: score >= 7
              ? "var(--accent-green)"
              : score >= 4
              ? "var(--accent-orange)"
              : "var(--accent-red)",
          }}
        />
      </div>

      {/* Breakdown */}
      <div className="grid-3" style={{ marginTop: "1rem" }}>
        {[
          ["Logic",    breakdown.logic],
          ["Evidence", breakdown.evidence],
          ["Clarity",  breakdown.clarity],
        ].map(([label, val]) => (
          <div key={label} className="metric-box">
            <div className="metric-value" style={{ fontSize: "1.2rem" }}>
              {val ?? "—"}/10
            </div>
            <div className="metric-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Reasoning */}
      {reasoning && (
        <p style={{ marginTop: "0.75rem", fontSize: "0.82rem",
          color: "var(--text-secondary)", lineHeight: 1.5 }}>
          💬 {reasoning}
        </p>
      )}

      {/* AI evaluation */}
      {evalData.grade && (
        <div style={{ marginTop: "0.75rem", padding: "0.75rem",
          background: "var(--bg-tertiary)", borderRadius: 6 }}>
          <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)",
            marginBottom: "0.25rem" }}>
            AI Response Quality
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ color: "var(--accent-orange)", letterSpacing: 2 }}>
              {GRADE_STARS[evalData.grade]}
            </span>
            <span style={{ fontWeight: 500, fontSize: "0.9rem" }}>
              {evalData.grade?.toUpperCase()} ({evalData.score}/10)
            </span>
          </div>
          {evalData.feedback && (
            <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)",
              marginTop: "0.25rem" }}>
              🎯 {evalData.feedback}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
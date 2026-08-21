/**
 * Memory context — debate summary and similar past arguments.
 */

import { useState } from "react";

export default function CoachingCard({ summary, similarArgs }) {
  const [showSummary, setShowSummary] = useState(false);

  if (!summary && (!similarArgs || similarArgs.length === 0)) return null;

  return (
    <div className="card">
      <h3 style={{ marginBottom: "1rem", fontSize: "0.95rem",
        textTransform: "uppercase", letterSpacing: "0.06em",
        color: "var(--text-secondary)" }}>
        🧠 Coaching Insights
      </h3>

      {/* Summary */}
      {summary && (
        <div style={{ marginBottom: "0.75rem" }}>
          <button
            className="btn btn-secondary"
            style={{ marginBottom: "0.5rem", fontSize: "0.82rem" }}
            onClick={() => setShowSummary(s => !s)}
          >
            {showSummary ? "▲ Hide" : "▼ Show"} Debate Summary
          </button>
          {showSummary && (
            <div style={{ padding: "0.75rem",
              background: "var(--bg-tertiary)", borderRadius: 6,
              fontSize: "0.85rem", lineHeight: 1.6,
              color: "var(--text-secondary)" }}>
              {summary}
            </div>
          )}
        </div>
      )}

      {/* Similar past arguments */}
      {similarArgs?.length > 0 && (
        <div className="alert alert-warning">
          <strong>Similar past argument detected</strong>
          <br />
          <span style={{ fontSize: "0.82rem" }}>
            Turn {similarArgs[0].turn_number}: "
            {(similarArgs[0].argument || "").slice(0, 100)}..."
          </span>
          <br />
          <span style={{ fontSize: "0.78rem", opacity: 0.8 }}>
            Quality then: {(similarArgs[0].quality || "").toUpperCase()} |
            Similarity: {Math.round((similarArgs[0].similarity || 0) * 100)}%
          </span>
        </div>
      )}
    </div>
  );
}
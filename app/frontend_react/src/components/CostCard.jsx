/**
 * Per-turn cost breakdown (collapsible).
 */

import { useState } from "react";

export default function CostCard({ cost }) {
  const [open, setOpen] = useState(false);
  if (!cost) return null;

  return (
    <div style={{ marginTop: "0.5rem" }}>
      <button
        className="btn btn-secondary"
        style={{ fontSize: "0.78rem", padding: "0.3rem 0.75rem" }}
        onClick={() => setOpen(o => !o)}
      >
        💰 {open ? "Hide" : "Show"} Turn Cost
      </button>

      {open && (
        <div className="card" style={{ marginTop: "0.5rem" }}>
          <div className="grid-3">
            <div className="metric-box">
              <div className="metric-value" style={{ fontSize: "1.1rem" }}>
                {(cost.total_tokens || 0).toLocaleString()}
              </div>
              <div className="metric-label">Tokens</div>
            </div>
            <div className="metric-box">
              <div className="metric-value" style={{ fontSize: "1.1rem" }}>
                ${(cost.total_cost_usd || 0).toFixed(6)}
              </div>
              <div className="metric-label">Cost</div>
            </div>
            <div className="metric-box">
              <div className="metric-value" style={{ fontSize: "1.1rem" }}>
                {cost.cache_hits || 0}
              </div>
              <div className="metric-label">Cache Hits</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
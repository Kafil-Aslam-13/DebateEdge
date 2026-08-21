/**
 * Fallacy detection result — only renders when fallacy detected.
 */

const SEVERITY_COLORS = {
  high:   "var(--accent-red)",
  medium: "var(--accent-orange)",
  low:    "var(--accent-blue)",
};

export default function FallacyCard({ fallacy }) {
  if (!fallacy?.detected) return null;

  const name     = (fallacy.name || "").replace(/_/g, " ").toUpperCase();
  const severity = fallacy.severity || "";
  const color    = SEVERITY_COLORS[severity] || "var(--accent-orange)";

  return (
    <div className="card" style={{ borderColor: "var(--accent-red)" }}>
      <h3 style={{ marginBottom: "1rem", color: "var(--accent-red)",
        fontSize: "0.95rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        ⚠️ Logical Fallacy Detected
      </h3>

      <div style={{ display: "flex", gap: "0.75rem",
        marginBottom: "0.75rem", flexWrap: "wrap" }}>
        <span className="tag tag-red">{name}</span>
        <span style={{ color, fontWeight: 500, fontSize: "0.85rem" }}>
          {severity.toUpperCase()} severity
        </span>
      </div>

      {fallacy.explanation && (
        <div className="alert alert-warning" style={{ marginBottom: "0.75rem" }}>
          <strong>What it means:</strong> {fallacy.explanation}
        </div>
      )}

      {fallacy.correction && fallacy.correction !== "none" && (
        <div className="alert alert-success">
          <strong>How to fix it:</strong> {fallacy.correction}
        </div>
      )}
    </div>
  );
}
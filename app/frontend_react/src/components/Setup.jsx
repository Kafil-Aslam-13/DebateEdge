/**
 * Setup screen — topic + side selection.
 * Calls /api/v1/debate/start on submit.
 */

import { useState } from "react";
import { api } from "../services/api";
import toast from "react-hot-toast";

export default function Setup({ onDebateStart }) {
  const [topic, setTopic]       = useState("");
  const [userSide, setUserSide] = useState("for");
  const [loading, setLoading]   = useState(false);

  const handleStart = async () => {
    if (!topic.trim() || topic.trim().length < 10) {
      toast.error("Please enter a debate topic (min 10 characters).");
      return;
    }

    setLoading(true);
    try {
      const { data } = await api.startDebate(topic.trim(), userSide);
      toast.success("Debate started!");
      onDebateStart({
        topic:    topic.trim(),
        userSide,
        aiSide:   data.ai_side,
        opening:  data.opening_statement,
      });
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      toast.error(`Failed to start debate: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 640, margin: "4rem auto", padding: "0 1rem" }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "2.5rem" }}>
        <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>🗣️</div>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 600, letterSpacing: "-0.03em" }}>
          DebateEdge
        </h1>
        <p style={{ color: "var(--text-secondary)", marginTop: "0.5rem" }}>
          AI Debate & Argument Coach — get scored, catch fallacies, improve
        </p>
      </div>

      {/* Setup card */}
      <div className="card">
        <h2 style={{ marginBottom: "1.5rem", fontSize: "1.1rem" }}>
          Start a New Debate
        </h2>

        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "block", marginBottom: "0.4rem",
            fontSize: "0.82rem", color: "var(--text-secondary)",
            textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Debate Topic
          </label>
          <input
            value={topic}
            onChange={e => setTopic(e.target.value)}
            placeholder="e.g. Social media does more harm than good"
            onKeyDown={e => e.key === "Enter" && handleStart()}
          />
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label style={{ display: "block", marginBottom: "0.4rem",
            fontSize: "0.82rem", color: "var(--text-secondary)",
            textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Your Side
          </label>
          <div style={{ display: "flex", gap: "0.75rem" }}>
            {["for", "against"].map(side => (
              <button
                key={side}
                className={`btn ${userSide === side ? "btn-primary" : "btn-secondary"}`}
                style={{ flex: 1, justifyContent: "center" }}
                onClick={() => setUserSide(side)}
              >
                Argue {side.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <button
          className="btn btn-primary"
          style={{ width: "100%", justifyContent: "center",
            padding: "0.75rem", fontSize: "1rem" }}
          onClick={handleStart}
          disabled={loading || !topic.trim()}
        >
          {loading
            ? <><span className="spinner" /> Starting...</>
            : "Start Debate →"
          }
        </button>
      </div>

      {/* Feature hints */}
      <div className="grid-3" style={{ marginTop: "1.5rem" }}>
        {[
          { icon: "🎯", label: "Scored Arguments", desc: "Logic, evidence, clarity" },
          { icon: "⚠️", label: "Fallacy Detection", desc: "Named + explained" },
          { icon: "📈", label: "Session Tracking", desc: "Improvement over time" },
        ].map(f => (
          <div key={f.label} className="card" style={{ textAlign: "center", padding: "1rem" }}>
            <div style={{ fontSize: "1.5rem" }}>{f.icon}</div>
            <div style={{ fontWeight: 500, marginTop: "0.3rem", fontSize: "0.85rem" }}>
              {f.label}
            </div>
            <div style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
              {f.desc}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
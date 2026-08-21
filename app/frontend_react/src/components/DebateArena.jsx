/**
 * Active debate view — history + argument input + results.
 */

import { useState, useRef, useEffect } from "react";
import { api } from "../services/api";
import ScoreCard from "./ScoreCard";
import FallacyCard from "./FallacyCard";
import CoachingCard from "./CoachingCard";
import CostCard from "./CostCard";
import SessionEval from "./SessionEval";
import toast from "react-hot-toast";

export default function DebateArena({ debate, onReset }) {
  const { topic, userSide, aiSide, opening } = debate;

  const [history, setHistory]       = useState([
    { role: "AI Opening", content: opening },
  ]);
  const [argument, setArgument]     = useState("");
  const [turnNumber, setTurnNumber] = useState(1);
  const [loading, setLoading]       = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [sessionEval, setSessionEval] = useState(null);
  const [loadingEval, setLoadingEval] = useState(false);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, lastResult]);

  const handleArgue = async () => {
    if (!argument.trim() || argument.trim().length < 10) {
      toast.error("Please write a meaningful argument (min 10 chars).");
      return;
    }

    setLoading(true);
    setLastResult(null);

    try {
      const { data } = await api.argue(
        topic, userSide, argument.trim(), turnNumber
      );

      setHistory(h => [
        ...h,
        { role: `You — Turn ${turnNumber}`, content: argument.trim() },
        { role: `AI — Turn ${turnNumber}`,  content: data.ai_response },
      ]);

      setLastResult(data);
      setTurnNumber(t => t + 1);
      setArgument("");

      if (!data.input_guard?.passed) {
        toast.error(`Argument blocked: ${data.input_guard?.reason}`);
      }

    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      toast.error(`Error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    setLoadingEval(true);
    try {
      const { data } = await api.evaluateSession();
      setSessionEval(data);
    } catch (err) {
      const msg = err.response?.data?.detail || "Need at least 2 turns.";
      toast.error(msg);
    } finally {
      setLoadingEval(false);
    }
  };

  const handleReset = async () => {
    try { await api.resetDebate(); } catch (_) {}
    onReset();
    toast.success("New debate started!");
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "1.5rem 1rem" }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between",
        alignItems: "flex-start", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 style={{ fontSize: "1.4rem", fontWeight: 600 }}>
            🗣️ DebateEdge
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem",
            marginTop: "0.2rem" }}>
            You argue <strong style={{ color: "var(--accent-blue)" }}>
              {userSide.toUpperCase()}
            </strong> &nbsp;·&nbsp;
            AI argues <strong style={{ color: "var(--accent-red)" }}>
              {aiSide.toUpperCase()}
            </strong> &nbsp;·&nbsp;
            Turn {turnNumber}
          </p>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem",
            marginTop: "0.2rem", fontStyle: "italic" }}>
            "{topic}"
          </p>
        </div>

        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button
            className="btn btn-secondary"
            onClick={handleEvaluate}
            disabled={loadingEval || turnNumber < 3}
            style={{ fontSize: "0.85rem" }}
          >
            {loadingEval
              ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Evaluating...</>
              : "📊 Evaluate Session"
            }
          </button>
          <button
            className="btn btn-danger"
            onClick={handleReset}
            style={{ fontSize: "0.85rem" }}
          >
            🔄 New Debate
          </button>
        </div>
      </div>

      <div className="divider" />

      {/* Debate history */}
      <div style={{ marginBottom: "1.5rem" }}>
        {history.map((entry, i) => {
          const isAI  = entry.role.startsWith("AI");
          const isYou = entry.role.startsWith("You");
          return (
            <div key={i} style={{
              marginBottom: "0.75rem",
              padding: "0.75rem 1rem",
              background: isAI
                ? "var(--bg-secondary)"
                : isYou
                ? "#1a2a3a"
                : "var(--bg-tertiary)",
              borderRadius: 8,
              borderLeft: `3px solid ${
                isAI  ? "var(--accent-red)"
                : isYou ? "var(--accent-blue)"
                : "var(--border)"
              }`,
            }}>
              <div style={{ fontSize: "0.72rem", fontWeight: 600,
                color: "var(--text-secondary)", marginBottom: "0.25rem",
                textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {entry.role}
              </div>
              <div style={{ fontSize: "0.9rem", lineHeight: 1.6 }}>
                {entry.content}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {/* Results from last turn */}
      {lastResult && (
        <div>
          <div className="divider" />
          <h3 style={{ marginBottom: "1rem", fontSize: "0.95rem",
            color: "var(--text-secondary)", textTransform: "uppercase",
            letterSpacing: "0.06em" }}>
            Analysis — Turn {turnNumber - 1}
          </h3>

          <div className="grid-2" style={{ marginBottom: "1rem" }}>
            <ScoreCard result={lastResult} />
            <FallacyCard fallacy={lastResult.fallacy} />
          </div>

          <CoachingCard
            summary={lastResult.debate_summary}
            similarArgs={lastResult.similar_past_args}
          />

          <CostCard cost={lastResult.cost} />
          <div className="divider" />
        </div>
      )}

      {/* Argument input */}
      <div className="card">
        <label style={{ display: "block", marginBottom: "0.5rem",
          fontSize: "0.82rem", color: "var(--text-secondary)",
          textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Your Argument — Turn {turnNumber}
        </label>
        <textarea
          value={argument}
          onChange={e => setArgument(e.target.value)}
          placeholder="Make your case with evidence and logic. Avoid fallacies!"
          rows={4}
          style={{ marginBottom: "0.75rem", resize: "vertical" }}
          onKeyDown={e => {
            if (e.key === "Enter" && e.ctrlKey) handleArgue();
          }}
        />
        <div style={{ display: "flex", justifyContent: "space-between",
          alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
            {argument.length}/2000 chars &nbsp;·&nbsp; Ctrl+Enter to submit
          </span>
          <button
            className="btn btn-primary"
            onClick={handleArgue}
            disabled={loading || !argument.trim()}
            style={{ minWidth: 160 }}
          >
            {loading
              ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Analysing...</>
              : `Submit Argument →`
            }
          </button>
        </div>
      </div>

      {/* Session evaluation modal */}
      {sessionEval && (
        <SessionEval
          data={sessionEval}
          onClose={() => setSessionEval(null)}
        />
      )}
    </div>
  );
}
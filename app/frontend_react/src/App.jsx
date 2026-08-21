/**
 * Root app — switches between Setup and DebateArena.
 */

import { useState, useEffect } from "react";
import { Toaster } from "react-hot-toast";
import Setup from "./components/Setup";
import DebateArena from "./components/DebateArena";
import { api } from "./services/api";

export default function App() {
  const [debate, setDebate]       = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    api.health()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#161b22",
            color: "#e6edf3",
            border: "1px solid #30363d",
          },
        }}
      />

      {/* API status banner */}
      {apiStatus === "offline" && (
        <div style={{
          background: "#3d1f1f",
          borderBottom: "1px solid var(--accent-red)",
          padding: "0.5rem 1rem",
          textAlign: "center",
          fontSize: "0.85rem",
          color: "#ffa0a0",
        }}>
          ⚠️ Backend API is offline. Make sure it's running at{" "}
          <code>{import.meta.env.VITE_BACKEND_URL || "http://localhost:8000"}</code>
        </div>
      )}

      {debate
        ? <DebateArena debate={debate} onReset={() => setDebate(null)} />
        : <Setup onDebateStart={setDebate} />
      }
    </>
  );
}
# 🗣️ DebateEdge — AI Debate & Argument Coach

DebateEdge is a production-grade, multi-agent AI system that debates against a
user in real time, scores every argument on logic/evidence/clarity, detects
named logical fallacies, retrieves supporting evidence via RAG, remembers
context across turns, and evaluates the user's improvement across a session —
all served through a FastAPI backend and a React frontend, deployed
independently on Railway and Vercel.

It was built end-to-end, in isolated iterations, specifically to demonstrate
production-level command of the modern GenAI stack: LangGraph orchestration,
multi-strategy retrieval, an LLM gateway with routing/fallback/caching,
observability, and cost-aware inference — not just a single prompt-in,
text-out wrapper.

**Live demo:** `https://debateedge-xxxx.vercel.app`
**API docs:** `https://debateedge-api-xxxx.up.railway.app/docs`

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Core Concepts Demonstrated](#core-concepts-demonstrated)
- [Project Structure](#project-structure)
- [Setup — Run Locally](#setup--run-locally)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## What It Does

1. **User picks a topic and a side** ("for" or "against").
2. **AI takes the opposite side** and opens with a strong statement.
3. **User submits an argument.** It is:
   - Classified as `strong` / `weak` / `fallacy` (structured LLM output)
   - Scored 0–10 on logic, evidence, and clarity
   - Checked for a **named logical fallacy** (ad hominem, strawman,
     slippery slope, etc.) via a tool-calling agent
   - Cross-checked against **semantically similar past arguments** in the
     session (so the AI can say "you made this same weak point in turn 2")
4. **AI responds** with a counterargument that:
   - Is routed to a fast or powerful model depending on the task
   - Is grounded in evidence retrieved via RAG (ChromaDB + Pinecone)
   - Adapts its coaching tone to the argument's quality
5. **Every turn is evaluated** by an LLM judge (relevance, evidence use,
   persuasiveness, coaching value) and logged as a graded turn.
6. **At session end**, the system produces a full report: score trend,
   improvement direction, fallacy pattern, and personalised coaching advice —
   plus the exact token/cost spend for the session.

---

## Architecture

```
┌─────────────────────┐        HTTPS        ┌──────────────────────────┐
│   React Frontend     │ ───────────────────▶│   FastAPI Backend         │
│   (Vercel)           │                      │   (Railway, Docker)      │
└─────────────────────┘                      └──────────┬───────────────┘
                                                          │
                                                          ▼
                                          ┌───────────────────────────────┐
                                          │      LangGraph State Machine   │
                                          │                                │
                                          │  classify → score → route      │
                                          │     │        │         │       │
                                          │  strong   weak    detect_      │
                                          │                   fallacy      │
                                          │     └────────┴─────────┘       │
                                          │              │                 │
                                          │      RAG retrieval              │
                                          │   (ChromaDB + Pinecone)          │
                                          │              │                 │
                                          │   generate_counterargument      │
                                          │      (via LLM Gateway)          │
                                          │              │                 │
                                          │       evaluate_turn              │
                                          │              │                 │
                                          │       update_memory              │
                                          │  (buffer + summary + vector)     │
                                          └───────────────────────────────┘
                                                          │
                                          ┌───────────────┴────────────────┐
                                          │         LLM Gateway             │
                                          │  litellm Router: fast/powerful  │
                                          │  fallback · cache · cost track │
                                          └───────────────┬─────────────────┘
                                                          │
                                                     Groq API
                                              (llama3-8b / llama3-70b)
```

Every node in the graph reads and writes a single typed `DebateState`
object — this is what makes the system a genuine multi-step agentic
pipeline rather than a single prompt call.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | **LangGraph** (`StateGraph`) | Conditional routing by argument quality, not a linear chain |
| LLM | **Groq** (llama3-8b + llama3-70b) | Free tier, fast inference, swappable via gateway |
| Gateway | **litellm Router** | Provider-agnostic routing, fallback chains, response caching |
| Fallacy Agent | **LangChain `create_agent`** + tools | Agent reasons over a fallacy knowledge base before classifying |
| Memory | Custom buffer / summary / vector | Recency, compression, and semantic recall — three distinct problems |
| RAG | **ChromaDB** (session) + **Pinecone** (persistent) | Vector *store* vs. vector *database* used for their actual strengths |
| Embeddings | **HuggingFace** (local) + **Cohere** (API) | Cost/quality tradeoff shown explicitly |
| Structured Output | `PydanticOutputParser`, `with_structured_output()` | Chosen per-task based on whether Python-level validation is needed |
| Evaluation | LLM-as-judge + rule-based trend analysis | Hybrid: LLM judges quality, math judges trend |
| Backend | **FastAPI** + Pydantic v2 | Async, typed, self-documenting (`/docs`) |
| Frontend | **React 18** + Vite | Component-based UI, fully decoupled from backend internals |
| Deployment | **Railway** (backend, Docker) + **Vercel** (frontend, static) | Independent scaling and deploy lifecycles |
| CI | **GitHub Actions** | Syntax checks + pytest on every push |

---

## Core Concepts Demonstrated

This project was deliberately built so that **no single technique is used
twice for the same reason** — each choice is justified by the task, not by
habit. This section is the "if a reviewer asks why," reference.

### 1. Prompt Engineering — 4 distinct techniques
- **Few-shot prompting** for argument classification (pattern recognition
  benefits from examples; zero-shot was unreliable in testing)
- **Prompt partials** (`.partial()`) for the three coaching-tone system
  prompts (strong / weak / fallacy) built from one base template
- **Plain instruction prompts** for generative tasks like fallacy
  explanation, where examples would over-constrain the output
- **f-string system prompts with baked-in examples** for the agent (`create_agent`
  takes a plain string, not a `ChatPromptTemplate`)

### 2. Output Parsing — matched to actual need
- `StrOutputParser` for conversational, unstructured responses
- `with_structured_output()` (JSON-mode constrained generation) for
  classification/scoring, where the model supports native structured output
- `PydanticOutputParser` with cross-field `model_validator`s for fallacy
  detection, where **Python-level validation logic** (if no fallacy, force
  all related fields to "none") is required — `with_structured_output()`
  cannot run custom validators

### 3. Agents vs. Chains
- The debate response and scoring are plain **LCEL chains** — no reasoning
  loop needed, one-shot transformation
- Fallacy detection is a **tool-calling agent** (`create_agent`) with two
  tools (`lookup_fallacy`, `list_fallacies`) — the model can verify a
  fallacy definition before committing to a classification

### 4. Memory — three separate problems, three separate solutions
- **Buffer memory**: explicit `HumanMessage`/`AIMessage` sliding window —
  answers "what did we just say"
- **Summary memory**: an LCEL summarization chain triggered only when the
  buffer nears its limit — answers "what happened earlier, compressed"
- **Vector memory**: ChromaDB + HuggingFace embeddings — answers "have I
  seen a semantically similar argument before," which neither of the above
  can do

### 5. RAG — vector store vs. vector database, used correctly
- **ChromaDB** (in-memory, session-scoped) with **similarity search** over
  an 11-document curated knowledge base — small corpus, want the single
  best match
- **Pinecone** (persistent, cross-session) with **MMR (Max Marginal
  Relevance)** — larger corpus, want diverse, non-redundant evidence
- Pinecone **degrades gracefully**: if `PINECONE_API_KEY` is unset, the
  system runs on ChromaDB alone with zero code changes

### 6. LLM Gateway
- Single `LLMGateway.complete()` entry point used by every graph node
- Routes by `task_type` (classification/scoring → fast model, debate →
  powerful model)
- Automatic fallback chain via `litellm.Router` if the primary model fails
- In-memory response cache (SHA-256 of messages+model) — identical prompts
  cost zero tokens on repeat
- Every call tracked for tokens and estimated USD cost

### 7. Evaluation
- **Turn-level**: LLM-as-judge scores the AI's own counterargument on
  relevance / evidence / persuasion / coaching value
- **Session-level**: rule-based (pure arithmetic — first-half vs.
  second-half score average) improvement trend, combined with an
  LLM-generated natural-language coaching summary — a hybrid where rules
  do the analysis and the LLM does the communication

### 8. Cost Management
- Token budget enforcement (arguments truncated at a hard character cap)
- Debate-history compression before injection into the prompt
- Score-based smart model routing (a consistently strong debater is routed
  to the cheaper model — coaching depth isn't needed)
- Per-turn and per-session cost reporting with cache-hit-rate savings

### 9. Observability
- **LangSmith**: automatic tracing of every LangChain chain/LLM call
  (env-var based, zero code changes to chains)
- **Logfire**: `logfire.span()` wraps every graph node — node-level
  latency is visible; custom business metrics (argument score trend,
  fallacy counts, cache hit rate) are emitted as first-class metrics, not
  just log lines

### 10. Validation (current implementation note)
- Input/output checks are implemented as **direct rule-based and
  LLM-as-judge validation** inside the graph (length limits, PII regex,
  prompt-injection pattern matching, topic-relevance and response-quality
  judging) rather than a third-party guardrails framework — kept simple and
  dependency-light for this iteration. See [Roadmap](#roadmap).

---

## Project Structure

```
DebateEdge/
├── src/
│   ├── core/            # config, logging, exceptions, constants
│   ├── gateway/          # LLM Gateway (litellm Router) + cost optimizer
│   ├── memory/           # buffer / summary / vector memory
│   ├── retrieval/        # ChromaDB + Pinecone + embeddings
│   ├── prompts/          # all prompt templates, by domain
│   ├── parsers/          # output parsers, by domain
│   ├── agents/           # fallacy detection agent
│   ├── graphs/           # LangGraph state + node definitions
│   ├── evaluation/       # turn + session evaluator
│   └── services/         # orchestration layer between graph and API
├── app/
│   ├── api/               # FastAPI app, routes, Pydantic schemas
│   └── frontend_react/    # React + Vite frontend
├── tests/                # pytest suite, one file per subsystem
├── configs/              # config.yaml, models.yaml (gateway routing)
├── docker/Dockerfile     # backend container (Railway)
├── railway.json          # Railway build/deploy config
├── .github/workflows/    # CI (test) pipeline
└── main.py               # CLI entrypoint for local debate testing
```

---

## Setup — Run Locally

### Backend

```bash
conda create -n debateedge python=3.12 -y
conda activate debateedge
pip install -r requirements.txt

cp .env.example .env
# add GROQ_API_KEY at minimum — everything else degrades gracefully

uvicorn app.api.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

### Frontend

```bash
cd app/frontend_react
npm install
cp .env.example .env   # VITE_BACKEND_URL=http://localhost:8000
npm run dev
# opens http://localhost:3000
```

### CLI (no frontend/backend needed)

```bash
python main.py
```

### Run tests

```bash
pytest tests/ -v --cov=src --cov=app
```

---

## Deployment

| Service | Platform | Config file |
|---|---|---|
| Backend | Railway (Docker) | `railway.json`, `docker/Dockerfile` |
| Frontend | Vercel (static, Vite build) | `app/frontend_react/vercel.json` |

Both deploy automatically from `main` via each platform's native GitHub
integration. Environment variables (`GROQ_API_KEY`, `FRONTEND_URL`, etc.)
are set directly in the Railway/Vercel dashboards.

---

## API Reference

Full interactive docs at `/docs` (Swagger UI). Summary:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/debate/start` | Start a session, get AI opening statement |
| `POST` | `/api/v1/debate/argue` | Submit an argument, get full turn analysis |
| `GET` | `/api/v1/debate/evaluate` | Full session evaluation (needs ≥2 turns) |
| `POST` | `/api/v1/debate/reset` | Clear session memory |
| `GET` | `/api/v1/debate/cost` | Session token/cost report |
| `GET` | `/api/v1/health` | Health check |

---

## Known Limitations

- Validation is rule-based + LLM-judge, not a dedicated guardrails
  framework (see Roadmap).
- Memory and cost tracking are process-local (in-memory), not persisted to
  a database — a restart clears session state.
- Pinecone/Cohere are optional; without them the system runs on
  ChromaDB + HuggingFace only, which is fully functional but less rich.

## Roadmap

- [ ] Swap direct validation for a dedicated guardrails layer
      (rule-based + hub-validator + LLM-judge, three-tier)
- [ ] Persist session state to Redis/Postgres for multi-instance scaling
- [ ] Add streaming responses (SSE) from the counterargument node
- [ ] User accounts + debate history across sessions

---

## Author

**Kafil Aslam**
[GitHub](https://github.com/Kafil-Aslam-13) ·
[LinkedIn](https://linkedin.com/in/kafil-aslam-69a880228)

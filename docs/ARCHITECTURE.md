# DebateEdge — Technical Deep Dive

This document exists for anyone (a recruiter, an interviewer, or future-you)
who wants to understand *why* the system is built the way it is, not just
*what* it does. The README covers the "what." This covers the "why," at the
level of individual design decisions.

---

## 1. Why LangGraph and not a single prompt chain

A debate turn is not one transformation — it's a **branching decision
process**:

```
classify_argument
        │
   score_argument
        │
  route_by_quality  ──┬── strong ──┐
                       ├── weak ───┤
                       └── fallacy ┴─▶ detect_fallacy_details
                                            │
                                    rag_retrieval
                                            │
                              generate_counterargument
                                            │
                                     evaluate_turn
                                            │
                                     update_memory
```

A `PromptTemplate | LLM | Parser` chain cannot branch. `route_by_quality`
is a Python function that inspects state and returns the name of the next
node — this is what LangGraph's conditional edges give you that a linear
chain cannot. The routing logic is testable in isolation (see
`tests/test_graphs.py`) without ever calling an LLM, because it's pure
Python over a `TypedDict`.

**The state object (`DebateState`)** is the single source of truth passed
between every node. Each node reads only the keys it needs and writes only
the keys it owns — no node ever mutates a key it doesn't "own." This
discipline is what keeps 10+ nodes maintainable instead of turning into a
tangle of shared mutable state.

---

## 2. Why three different output parsers, not one

| Parser | Used for | Why this one and not another |
|---|---|---|
| `StrOutputParser` | Debate responses, fallacy explanations | Output is prose meant for a human to read directly — wrapping it in a schema would add nothing |
| `with_structured_output()` | Argument classification, scoring | The model natively supports JSON-mode constrained generation. Constraining *at generation time* means malformed output is structurally impossible — no post-hoc fixing needed |
| `PydanticOutputParser` + `model_validator` | Fallacy detection | Needs a **cross-field rule** — "if `contains_fallacy` is False, force `fallacy_name`/`severity`/`correction` to `'none'`." `with_structured_output()` constrains the *shape* of the JSON but cannot execute arbitrary Python validation logic after generation. `PydanticOutputParser` parses, then runs the model's validators, which is exactly where this consistency rule needs to live |

The decision tree used throughout the codebase:

```
Need structured output?
 ├─ No  → StrOutputParser
 └─ Yes → Does the model support native JSON-mode?
           ├─ Yes → does the schema need cross-field Python validation?
           │         ├─ No  → with_structured_output()
           │         └─ Yes → PydanticOutputParser (validators do the work)
           └─ No  → JsonOutputParser + manual safe-parse fallback
```

---

## 3. Why an agent for fallacy detection but chains everywhere else

`generate_counterargument` and `score_argument` are one-shot
transformations: given fixed inputs, produce one output. No branching
reasoning is needed — an LCEL chain (`prompt | llm | parser`) is the
correct, simplest tool.

Fallacy detection is different: the model may need to **verify** a
suspected fallacy against a definition before committing to it. This is
exactly what an agent with tools is for. `create_agent` is given two
tools:

- `lookup_fallacy(name)` — returns the definition + example of a specific
  fallacy from a small in-code knowledge base
- `list_fallacies()` — lists all detectable fallacies, used when the model
  is unsure which one applies

The agent can call `lookup_fallacy("ad_hominem")` to confirm a match
before returning its final JSON classification, rather than guessing from
pattern memory alone. This is the practical distinction between "a chain"
and "an agent": an agent can take an intermediate action and use the
result before answering.

---

## 4. Why three separate memory objects instead of one

Each memory type answers a **different question** — conflating them into
one object would mean picking a lossy compromise for all three:

| Memory | Question it answers | Implementation |
|---|---|---|
| Buffer | "What did we just say?" | Explicit `list[HumanMessage \| AIMessage]`, sliding window, `pop_oldest()` used to feed the summary before truncation |
| Summary | "What happened earlier, compressed?" | An LCEL chain (`ChatPromptTemplate → ChatGroq → .content`) invoked *only* when the buffer nears its limit — not every turn, to save tokens |
| Vector | "Have I seen something like this before?" | ChromaDB + HuggingFace embeddings, `similarity_search_with_score`, filtered by a similarity threshold |

The buffer and summary are both about **recency and compression** along
a timeline. The vector store is about **meaning**, independent of when
something was said. No amount of clever windowing or summarizing
substitutes for actual semantic search — that's the reason all three
exist simultaneously rather than picking the "best" one.

---

## 5. Why ChromaDB *and* Pinecone (not just one)

This is a deliberate demonstration of the difference between a
**vector store** and a **vector database**:

| | ChromaDB (here) | Pinecone (here) |
|---|---|---|
| Persistence | In-memory, session-scoped | Cloud, survives restarts |
| Corpus size | 11 curated documents | Designed for a much larger, growing corpus |
| Retrieval strategy | `similarity_search` — small corpus, want the single best match | `max_marginal_relevance_search` (MMR) — larger corpus, want *diverse* results so the AI doesn't cite three near-duplicate sources |
| Embeddings | HuggingFace (local, free, fast) | Cohere (API, richer semantic representation) |
| Failure mode | N/A — always available | Degrades to "no results" if `PINECONE_API_KEY` is unset — the system does **not** crash, it just runs on ChromaDB alone |

The graceful-degradation path is intentional and tested
(`tests/test_retrieval.py::test_pinecone_graceful_degradation`): the app
is fully functional with zero paid API keys, and gets *richer* — not
different in kind — when Pinecone/Cohere are configured.

---

## 6. Why a custom LLM Gateway instead of calling `ChatGroq` directly

Before the gateway existed, every graph node imported `ChatGroq` and built
its own model instance. The gateway (`src/gateway/llm_gateway.py`)
consolidates every LLM call in the codebase behind one method:

```python
gateway.complete(messages, task_type="classification")
```

What this buys, concretely:

- **Routing by task, not by node.** `configs/models.yaml` maps
  `task_routing: {classification: fast, debate: powerful, ...}`. Changing
  which model handles which task is a config edit, not a code change.
- **Fallback.** `litellm.Router` is configured with a fallback chain — if
  the primary model for a task fails, the router automatically retries
  against the fallback model, without any node needing its own try/except
  around model selection.
- **Caching.** Every call is keyed by `SHA-256(messages + model_alias)`.
  Identical prompts (which happen often — e.g. re-evaluating the same
  turn) return instantly from an in-memory dict at zero additional token
  cost.
- **Cost tracking.** Every call — cached or not — is recorded by
  `CostTracker` with prompt/completion tokens and an estimated USD cost,
  which feeds directly into the Sprint 10 cost-optimization layer and the
  session-end cost report shown to the user.

This is the single most "production engineering" piece of the system —
it is the difference between a demo that calls an API directly and a
system built the way an ML platform team would actually build it.

---

## 7. Why evaluation is split into "turn" and "session," and why each
   uses a different judging method

**Turn-level evaluation** asks a subjective question — *"was this AI
counterargument actually good?"* — which requires judgment, not
arithmetic. This is delegated to an LLM judge that scores four
dimensions (relevance, evidence use, persuasiveness, coaching value).

**Session-level evaluation** asks a question that is **pure arithmetic**
— *"is the user's score trending up or down?"* — computed by comparing
the mean of the first half of scores to the mean of the second half.
There is no reason to spend an LLM call determining whether `7 > 4`.

The session evaluator is intentionally **hybrid**: the trend, grade, and
counts are 100% rule-based (`_compute_trend`, `_compute_grade` — see
`tests/test_evaluation.py`, which tests these with zero API calls), and
only the final **coaching advice paragraph** is generated by an LLM,
because turning a set of statistics into encouraging, specific natural
language is a generation task, not a computation task.

This hybrid split — rules for analysis, LLM for communication — is a
pattern that shows up twice in the codebase (also in
`_generate_advice`) and is one of the more defensible design choices to
walk an interviewer through.

---

## 8. Why cost optimization is a separate module, not scattered logic

`CostOptimizer` centralizes four independent techniques so that adding a
fifth (e.g., dynamic batching) doesn't require touching graph nodes:

1. **Argument truncation** — hard character cap, cuts at word boundary
2. **History compression** — keeps last 3 turns verbatim before a prompt
   injection point; anything older relies on the summary memory instead
3. **Score-based smart routing** — a debater whose last three scores
   average above 7 is routed to the cheap model for their next
   counterargument (they don't need the powerful model's coaching depth);
   a struggling debater is kept on the powerful model
4. **Reporting** — per-turn and per-session token/cost/cache-hit summaries

Each of these is a genuinely independent lever — a real cost-optimization
project on an LLM product would very likely implement most of these in
roughly this shape.

---

## 9. What is *not* a "guardrails framework," and why that's stated
   plainly

The current implementation validates input (length, PII regex,
prompt-injection pattern matching) and output (toxicity, topic/response
relevance) with a mix of rule-based checks and LLM-as-judge calls, wired
directly into two graph nodes (`input_guardrail_node`,
`output_guardrail_node`). It does **not** currently use a third-party
guardrails library (e.g., Guardrails AI Hub validators).

This is documented explicitly (in the README's Known Limitations and
here) rather than glossed over, because overstating this in an interview
is an easy way to get caught out by a specific follow-up question. The
graph structure already has the exact two insertion points
(`NODE_INPUT_GUARD`, `NODE_OUTPUT_GUARD`) where a dedicated
guardrails library could be dropped in later without touching the rest of
the pipeline — that's listed as the first Roadmap item.

---

## 10. Why the backend and frontend are deployed as two independent
    services

FastAPI (Railway, Docker) and React (Vercel, static build) are separate
deployables with separate lifecycles, separate scaling characteristics,
and separate CI build steps (`test-backend` / `test-frontend` as
independent GitHub Actions jobs). CORS is configured to read the
frontend's deployed URL from an environment variable
(`FRONTEND_URL`) rather than hardcoding it, so the two services can be
redeployed independently without a code change on either side — only an
env var update on Railway.

This mirrors how a real product team would split a Python inference
service from a JS frontend, rather than serving both from a single
process — which is a meaningfully different (and less production-shaped)
architecture.

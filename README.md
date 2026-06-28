# Context-Aware Agent Memory System with Verifiable Retention Scoring

A research project exploring proactive memory management for LLM agents. Instead of keeping all conversation history in context until the window fills up, this system continuously validates which memories actually influence agent reasoning and demotes the ones that don't.

## The Problem

Current LLM systems (LangChain, raw API calls) handle long conversations by appending every message to context until the window fills up, then reactively truncating based on recency or semantic similarity. These are heuristics — they don't verify whether a memory chunk actually affects the model's answers.

**Result:** token usage grows linearly (or worse) with conversation length, and there's no guarantee that important context is retained.

## The Approach

Three-tier memory architecture with verifiable retention scoring:

```
HOT  → Recent turns, verbatim, always in context
WARM → Older turns, compressed summaries, continuously tested
COLD → Vector store, retrieved only when semantically relevant
```

Before any WARM chunk is moved to COLD, the system **proves** it is safe to discard:
1. Generates a probe question targeting that chunk's content
2. Runs the agent with and without the chunk
3. Compares answers using cosine similarity
4. Only demotes if similarity ≥ threshold (chunk wasn't changing the reasoning)

## Architecture

```
user_input
    │
    ▼
[assemble_context]  ← HOT (verbatim) + WARM (summaries) + COLD (top-k retrieval)
    │
    ▼
[call_model]        ← LangGraph node, swappable backend (Groq / Anthropic / Ollama)
    │
    ▼
[write_memory]      ← write to HOT, trigger demotion cycle
    │
    ├── HOT → WARM  (when HOT exceeds N turns: summarize oldest half)
    └── WARM → COLD (probe-verified: only demote if agent doesn't need the chunk)
                         │
                         ├── score_chunk()  → generate probe question
                         ├── answer_with    → agent sees the chunk
                         ├── answer_without → agent doesn't see the chunk
                         └── cosine similarity → keep or demote
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph |
| Hot / Warm storage | PostgreSQL + SQLAlchemy |
| Cold vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM backends | Groq / Anthropic / Ollama (switchable via `.env`) |
| Token counting | tiktoken |

## Project Structure

```
context_agent/
├── agent/
│   ├── graph.py          # LangGraph pipeline with retry logic for rate limits
│   └── llm_factory.py    # Swappable LLM backend (Groq / Anthropic / Ollama / Gemini)
├── memory/
│   ├── store.py          # Read/write for all three tiers
│   ├── demotion.py       # HOT→WARM→COLD demotion logic (Phase 1 + Phase 2)
│   ├── probe.py          # Probe question generation + with/without agent runs
│   └── scorer.py         # Cosine similarity comparison via sentence-transformers
├── measurement/
│   └── tracker.py        # Token counter, retention records, CSV logging
├── run_experiment.py     # 25-turn scripted experiment (Phase 2 with retention scoring)
├── eval_recall.py        # Head-to-head recall evaluation: Phase 1 vs Phase 2
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/your-username/context-agent.git
cd context-agent
pip install -r requirements.txt
```

**2. Start PostgreSQL**
```bash
docker run -d --name context_agent_db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=context_agent \
  -p 5433:5432 postgres
```

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env`:

```env
# Groq (free tier — recommended)
LLM_BACKEND=groq
GROQ_API_KEY=your_key
MODEL_NAME=llama-3.3-70b-versatile
SUMMARY_MODEL=llama-3.1-8b-instant
PROBE_MODEL=llama-3.1-8b-instant

# Memory tier limits
HOT_TURN_LIMIT=4
WARM_ENTRY_LIMIT=2

# Retention scoring
USE_RETENTION_SCORING=true
RETENTION_SIMILARITY_THRESHOLD=0.80
```

**4. Run Phase 2 experiment (25 turns with retention scoring)**
```bash
python run_experiment.py
```

**5. Run recall evaluation (Phase 1 vs Phase 2 head-to-head)**
```bash
python eval_recall.py
```

## Results

### Phase 1 — Token savings (age-based demotion, 25 turns)

| Turn | Tiered Tokens | Naive Tokens | Ratio |
|------|--------------|-------------|-------|
| 1    | 28           | 86          | 3.1x  |
| 5    | 1,051        | 1,868       | 1.8x  |
| 10   | 2,818        | 5,736       | 2.0x  |
| 15   | 2,589        | 10,126      | 3.9x  |
| 20   | 3,760        | 14,551      | 3.9x  |
| 25   | 2,265        | 18,629      | 8.2x  |

**Final turn token savings: ~87%**

### Phase 2 — Retention scoring results (25 turns, threshold=0.80)

| Metric | Value |
|--------|-------|
| Total probes run | 28 |
| Chunks kept (score < 0.80 — still needed) | 20 |
| Chunks demoted (score ≥ 0.80 — redundant) | 8 |
| Average similarity score | 0.799 |
| Final turn token savings | 84.7% |

### Recall evaluation — Phase 1 vs Phase 2 (12 recall questions, 37 setup turns)

| Question | Phase 1 | Phase 2 |
|----------|---------|---------|
| What is my name? | PASS | PASS |
| What is the size of my dataset? | PASS | PASS |
| What two ML algorithms was I considering? | PASS | PASS |
| What distributed computing framework am I using? | PASS | PASS |
| What cloud platform am I deploying on? | PASS | PASS |
| What was my API P99 latency and SLA? | PASS | PASS |
| What evaluation metrics did we discuss? | PASS | PASS |
| What was the diversity algorithm I asked about? | PASS | PASS |
| What model serialization formats did we compare? | PASS | PASS |
| What experiment tracking tool am I using? | PASS | PASS |
| What API framework am I using? | PASS | PASS |
| What was the cold-start solution we discussed? | PASS | PASS |
| **TOTAL** | **12/12** | **12/12** |

**Phase 2 matches Phase 1 recall (100%) while guaranteeing no chunk is discarded unless proven redundant.**

## Key Design Decisions

**Why not just use LangChain memory?**
LangChain's memory modules use recency or semantic similarity as heuristics for what to keep. This system uses a verifiable signal — if removing a chunk doesn't change the model's answer, it provably wasn't needed.

**Why three tiers instead of two?**
HOT keeps verbatim recent context for coherence. WARM keeps compressed summaries of older turns cheaply. COLD handles long-range retrieval without polluting the context window on every turn.

**Why probe before demoting?**
Age-based demotion (Phase 1) gets lucky in simple conversations but has no guarantee. A chunk from turn 3 might still be the only place a critical fact lives at turn 40. Phase 2 checks empirically before discarding.

**What does the similarity threshold control?**
At 0.85: more aggressive demotions, lower recall risk in simple conversations.
At 0.80: conservative, preserves chunks that have even moderate influence. Recommended default.

## Groq Free Tier Notes

- `llama-3.3-70b-versatile`: 100k tokens/day — use for main model
- `llama-3.1-8b-instant`: 500k tokens/day (separate limit) — use for summary and probe calls
- Both models: 6k tokens/minute — the agent retries automatically on 429 errors
- For `eval_recall.py` (49 turns × 2 phases), set `MODEL_NAME=llama-3.1-8b-instant` to stay within daily limits

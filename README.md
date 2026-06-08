# Context-Aware Agent Memory System with Verifiable Retention Scoring

A research project exploring proactive memory management for LLM agents. Instead of keeping all conversation history in context until the window fills up, this system continuously validates which memories actually influence agent reasoning — and demotes the ones that don't.

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
    └── WARM → COLD (when WARM exceeds M entries: embed oldest into vector store)
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
│   ├── graph.py          # LangGraph pipeline
│   └── llm_factory.py    # Swappable LLM backend (Groq / Anthropic / Ollama / Gemini)
├── memory/
│   ├── store.py          # Read/write for all three tiers
│   └── demotion.py       # HOT→WARM→COLD demotion logic
├── measurement/
│   └── tracker.py        # Token counter, naive vs tiered comparison
├── run_experiment.py     # 50-turn scripted experiment
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
  -p 5432:5432 postgres
```

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env` with your chosen backend:

```env
# Groq (free tier)
LLM_BACKEND=groq
GROQ_API_KEY=your_key
MODEL_NAME=llama-3.3-70b-versatile
SUMMARY_MODEL=llama-3.1-8b-instant

# OR Anthropic
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=your_key
MODEL_NAME=claude-haiku-4-5-20251001
SUMMARY_MODEL=claude-haiku-4-5-20251001

# OR local Ollama
LLM_BACKEND=ollama
MODEL_NAME=qwen2.5:7b
SUMMARY_MODEL=qwen2.5:1.5b
```

**4. Run the experiment**
```bash
python run_experiment.py
```

## Key Design Decisions

**Why not just use LangChain memory?**
LangChain's memory modules use recency or semantic similarity as heuristics for what to keep. This system uses a verifiable signal — if removing a chunk doesn't change the model's answer, it provably wasn't needed.

**Why three tiers instead of two?**
HOT keeps verbatim recent context for coherence. WARM keeps compressed summaries of older turns cheaply. COLD handles long-range retrieval without polluting the context window on every turn.


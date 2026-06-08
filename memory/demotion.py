import os
import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm_factory import get_llm
from memory.store import (
    count_hot, count_warm,
    delete_hot_turns, delete_warm_by_id,
    get_oldest_hot_turns, get_oldest_warm,
    write_cold, write_warm,
)

load_dotenv()

HOT_TURN_LIMIT = int(os.getenv("HOT_TURN_LIMIT", "10"))
WARM_ENTRY_LIMIT = int(os.getenv("WARM_ENTRY_LIMIT", "5"))
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen2.5:1.5b")

_enc = tiktoken.get_encoding("cl100k_base")
_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = get_llm(SUMMARY_MODEL, max_tokens=512)
    return _summarizer


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _summarize_turns(turns: list[dict]) -> str:
    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    llm = _get_summarizer()
    response = llm.invoke([
        SystemMessage(content="Summarize the following conversation excerpt into 2-4 sentences. Preserve key facts, decisions, and named entities. Be concise."),
        HumanMessage(content=transcript),
    ])
    return response.content


def demote_hot_to_warm(session_id: str):
    """
    When hot tier exceeds HOT_TURN_LIMIT individual messages,
    take the oldest half and compress them into a single warm summary.
    """
    hot_count = count_hot(session_id)
    if hot_count <= HOT_TURN_LIMIT:
        return

    # Demote the oldest half of the hot tier
    n_to_demote = hot_count // 2
    oldest_turns = get_oldest_hot_turns(session_id, n_to_demote)

    if not oldest_turns:
        return

    summary = _summarize_turns(oldest_turns)
    token_count = count_tokens(summary)

    turn_start = oldest_turns[0]["turn_number"]
    turn_end = oldest_turns[-1]["turn_number"]

    write_warm(session_id, turn_start, turn_end, summary, token_count)
    delete_hot_turns(session_id, [t["turn_number"] for t in oldest_turns])

    print(f"[demotion] HOT→WARM: turns {turn_start}-{turn_end} summarized ({token_count} tokens)")


def demote_warm_to_cold(session_id: str):
    """
    When warm tier exceeds WARM_ENTRY_LIMIT entries,
    embed and push the oldest entry into the cold vector store.
    """
    warm_count = count_warm(session_id)
    if warm_count <= WARM_ENTRY_LIMIT:
        return

    oldest = get_oldest_warm(session_id)
    if oldest is None:
        return

    write_cold(session_id, oldest["turn_range"], oldest["summary"])
    delete_warm_by_id(oldest["id"])

    print(f"[demotion] WARM→COLD: turns {oldest['turn_range']} embedded into vector store")


def run_demotion_cycle(session_id: str):
    """Single entry point — call this after every turn write."""
    demote_hot_to_warm(session_id)
    demote_warm_to_cold(session_id)

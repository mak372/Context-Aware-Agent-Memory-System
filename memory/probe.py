import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm_factory import get_llm
from memory.scorer import is_redundant
from memory.store import read_hot

load_dotenv()

PROBE_MODEL = os.getenv("PROBE_MODEL", os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant"))

_probe_llm = None


def _get_probe_llm():
    global _probe_llm
    if _probe_llm is None:
        _probe_llm = get_llm(PROBE_MODEL, max_tokens=256)
    return _probe_llm


def _generate_probe_question(chunk_summary: str) -> str:
    """Ask the LLM to generate a question that specifically requires this chunk to answer."""
    llm = _get_probe_llm()
    response = llm.invoke([
        SystemMessage(content="Generate one short, specific question whose correct answer requires the following information. Output only the question, nothing else."),
        HumanMessage(content=chunk_summary),
    ])
    return response.content.strip()


def _build_hot_context(session_id: str) -> str:
    """Get current HOT tier as a plain text context string."""
    hot_entries = read_hot(session_id)
    return "\n".join(f"{e['role'].upper()}: {e['content']}" for e in hot_entries)


def _answer_with_chunk(hot_context: str, chunk_summary: str, probe_question: str) -> str:
    """Run inference with the chunk included in context."""
    llm = _get_probe_llm()
    system = f"You are a helpful assistant. Here is conversation history:\n{hot_context}\n\nAdditional context:\n{chunk_summary}"
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=probe_question),
    ])
    return response.content.strip()


def _answer_without_chunk(hot_context: str, probe_question: str) -> str:
    """Run inference without the chunk."""
    llm = _get_probe_llm()
    system = f"You are a helpful assistant. Here is conversation history:\n{hot_context}"
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=probe_question),
    ])
    return response.content.strip()


def score_chunk(session_id: str, chunk: dict) -> dict:
    """
    Run the full probe cycle for a WARM chunk.
    Returns: {redundant, score, probe_question, answer_with, answer_without}
    """
    chunk_summary = chunk["summary"]
    hot_context = _build_hot_context(session_id)

    probe_question = _generate_probe_question(chunk_summary)
    answer_with = _answer_with_chunk(hot_context, chunk_summary, probe_question)
    answer_without = _answer_without_chunk(hot_context, probe_question)

    redundant, score = is_redundant(answer_with, answer_without)

    print(f"[probe] chunk turns {chunk['turn_range']} | score={score:.3f} | {'REDUNDANT→demote' if redundant else 'NEEDED→keep'}")
    print(f"  Q: {probe_question[:80]}")

    return {
        "redundant": redundant,
        "score": score,
        "probe_question": probe_question,
        "answer_with": answer_with,
        "answer_without": answer_without,
        "turn_range": chunk["turn_range"],
    }

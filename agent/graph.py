import os
import time
import uuid
from typing import TypedDict

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent.llm_factory import get_llm
from memory.demotion import count_tokens, run_demotion_cycle
from memory.store import init_db, query_cold, read_hot, read_warm, write_hot

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")

_enc = tiktoken.get_encoding("cl100k_base")
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(MODEL_NAME, max_tokens=1024)
    return _llm


class AgentState(TypedDict):
    session_id: str
    turn_number: int
    user_input: str
    response: str
    tokens_used: int      # tokens sent to model this turn
    probe_results: list   # retention scoring results from this turn's demotion cycle
    _messages: list       # assembled messages for this turn, passed between nodes


def assemble_context(state: AgentState) -> AgentState:
    """Build the message list the model will see this turn."""
    session_id = state["session_id"]

    warm_entries = read_warm(session_id)
    hot_entries = read_hot(session_id)
    cold_snippets = query_cold(session_id, state["user_input"])

    messages = []

    # System prompt with warm summaries prepended
    system_parts = ["You are a helpful assistant with memory across a long conversation."]
    if warm_entries:
        system_parts.append("\n\n--- Earlier conversation summaries ---")
        for w in warm_entries:
            system_parts.append(f"[Turns {w['turn_range'][0]}-{w['turn_range'][1]}]: {w['summary']}")

    if cold_snippets:
        system_parts.append("\n\n--- Relevant archived context ---")
        for snippet in cold_snippets:
            system_parts.append(snippet)

    messages.append(SystemMessage(content="\n".join(system_parts)))

    # Verbatim hot turns
    for entry in hot_entries:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        else:
            messages.append(AIMessage(content=entry["content"]))

    # Current user input
    messages.append(HumanMessage(content=state["user_input"]))

    # Count tokens being sent
    full_text = " ".join(
        m.content for m in messages if hasattr(m, "content") and isinstance(m.content, str)
    )
    tokens_used = count_tokens(full_text)

    state["_messages"] = messages  # type: ignore[typeddict-unknown-key]
    state["tokens_used"] = tokens_used
    return state


def call_model(state: AgentState) -> AgentState:
    messages = state["_messages"]  # type: ignore[typeddict-item]
    llm = _get_llm()
    for attempt in range(5):
        try:
            response = llm.invoke(messages)
            state["response"] = response.content
            return state
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("call_model failed after 5 retries")


def write_memory(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    turn = state["turn_number"]

    write_hot(session_id, turn * 2 - 1, "user", state["user_input"], count_tokens(state["user_input"]))
    write_hot(session_id, turn * 2, "assistant", state["response"], count_tokens(state["response"]))

    probe_results = run_demotion_cycle(session_id)
    state["probe_results"] = probe_results
    return state


# ---------- Build the graph ----------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("assemble_context", assemble_context)
    g.add_node("call_model", call_model)
    g.add_node("write_memory", write_memory)

    g.set_entry_point("assemble_context")
    g.add_edge("assemble_context", "call_model")
    g.add_edge("call_model", "write_memory")
    g.add_edge("write_memory", END)

    return g.compile()


def create_session() -> str:
    init_db()
    return str(uuid.uuid4())


def chat(graph, session_id: str, turn_number: int, user_input: str) -> dict:
    result = graph.invoke({
        "session_id": session_id,
        "turn_number": turn_number,
        "user_input": user_input,
        "response": "",
        "tokens_used": 0,
        "probe_results": [],
        "_messages": [],
    })
    return {
        "response": result["response"],
        "tokens_used": result["tokens_used"],
        "probe_results": result["probe_results"],
    }

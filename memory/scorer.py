import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

RETENTION_SIMILARITY_THRESHOLD = float(os.getenv("RETENTION_SIMILARITY_THRESHOLD", "0.85"))

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare_answers(answer_a: str, answer_b: str) -> float:
    """Returns cosine similarity between two answers. 1.0 = identical, 0.0 = completely different."""
    embedder = _get_embedder()
    emb_a = embedder.encode(answer_a).tolist()
    emb_b = embedder.encode(answer_b).tolist()
    return _cosine_similarity(emb_a, emb_b)


def is_redundant(answer_with: str, answer_without: str) -> tuple[bool, float]:
    """
    Returns (redundant, score).
    redundant=True means removing the chunk didn't change the answer → safe to demote.
    redundant=False means the chunk is influencing reasoning → keep it.
    """
    score = compare_answers(answer_with, answer_without)
    return score >= RETENTION_SIMILARITY_THRESHOLD, score

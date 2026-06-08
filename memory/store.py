import os
from datetime import datetime
from typing import Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine, text)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/context_agent")
COLD_TOP_K = int(os.getenv("COLD_TOP_K", "3"))

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class HotMemory(Base):
    __tablename__ = "hot_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    turn_number = Column(Integer, nullable=False)
    role = Column(String, nullable=False)   # "user" or "assistant"
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class WarmMemory(Base):
    __tablename__ = "warm_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    turn_range_start = Column(Integer, nullable=False)
    turn_range_end = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


# ---------- ChromaDB (cold tier) ----------

_chroma_client = None
_chroma_collection = None
_embedder = None


def _get_cold_store():
    global _chroma_client, _chroma_collection, _embedder
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_data")
        _chroma_collection = _chroma_client.get_or_create_collection("cold_memory")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _chroma_collection, _embedder


# ---------- HOT tier ----------

def write_hot(session_id: str, turn_number: int, role: str, content: str, token_count: int):
    with SessionLocal() as db:
        entry = HotMemory(
            session_id=session_id,
            turn_number=turn_number,
            role=role,
            content=content,
            token_count=token_count,
        )
        db.add(entry)
        db.commit()


def read_hot(session_id: str) -> list[dict]:
    with SessionLocal() as db:
        rows = (
            db.query(HotMemory)
            .filter(HotMemory.session_id == session_id)
            .order_by(HotMemory.turn_number.asc())
            .all()
        )
        return [
            {"role": r.role, "content": r.content, "turn_number": r.turn_number, "token_count": r.token_count}
            for r in rows
        ]


def count_hot(session_id: str) -> int:
    with SessionLocal() as db:
        return db.query(HotMemory).filter(HotMemory.session_id == session_id).count()


def delete_hot_turns(session_id: str, turn_numbers: list[int]):
    with SessionLocal() as db:
        db.query(HotMemory).filter(
            HotMemory.session_id == session_id,
            HotMemory.turn_number.in_(turn_numbers),
        ).delete(synchronize_session=False)
        db.commit()


def get_oldest_hot_turns(session_id: str, n: int) -> list[dict]:
    with SessionLocal() as db:
        rows = (
            db.query(HotMemory)
            .filter(HotMemory.session_id == session_id)
            .order_by(HotMemory.turn_number.asc())
            .limit(n)
            .all()
        )
        return [
            {"role": r.role, "content": r.content, "turn_number": r.turn_number, "token_count": r.token_count}
            for r in rows
        ]


# ---------- WARM tier ----------

def write_warm(session_id: str, turn_range_start: int, turn_range_end: int, summary: str, token_count: int):
    with SessionLocal() as db:
        entry = WarmMemory(
            session_id=session_id,
            turn_range_start=turn_range_start,
            turn_range_end=turn_range_end,
            summary=summary,
            token_count=token_count,
        )
        db.add(entry)
        db.commit()


def read_warm(session_id: str) -> list[dict]:
    with SessionLocal() as db:
        rows = (
            db.query(WarmMemory)
            .filter(WarmMemory.session_id == session_id)
            .order_by(WarmMemory.turn_range_start.asc())
            .all()
        )
        return [
            {
                "summary": r.summary,
                "turn_range": (r.turn_range_start, r.turn_range_end),
                "token_count": r.token_count,
                "id": r.id,
            }
            for r in rows
        ]


def count_warm(session_id: str) -> int:
    with SessionLocal() as db:
        return db.query(WarmMemory).filter(WarmMemory.session_id == session_id).count()


def get_oldest_warm(session_id: str) -> Optional[dict]:
    with SessionLocal() as db:
        row = (
            db.query(WarmMemory)
            .filter(WarmMemory.session_id == session_id)
            .order_by(WarmMemory.turn_range_start.asc())
            .first()
        )
        if row is None:
            return None
        return {
            "id": row.id,
            "summary": row.summary,
            "turn_range": (row.turn_range_start, row.turn_range_end),
            "token_count": row.token_count,
        }


def delete_warm_by_id(warm_id: int):
    with SessionLocal() as db:
        db.query(WarmMemory).filter(WarmMemory.id == warm_id).delete()
        db.commit()


# ---------- COLD tier ----------

def write_cold(session_id: str, turn_range: tuple[int, int], summary: str):
    collection, embedder = _get_cold_store()
    embedding = embedder.encode(summary).tolist()
    doc_id = f"{session_id}_{turn_range[0]}_{turn_range[1]}"
    collection.upsert(
        ids=[doc_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{"session_id": session_id, "turn_start": turn_range[0], "turn_end": turn_range[1]}],
    )


def query_cold(session_id: str, query_text: str, top_k: int = COLD_TOP_K) -> list[str]:
    collection, embedder = _get_cold_store()
    if collection.count() == 0:
        return []
    query_embedding = embedder.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"session_id": session_id},
    )
    return results["documents"][0] if results["documents"] else []

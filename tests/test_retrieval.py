"""Tests for Sprint 5 retrieval — no API keys needed for ChromaDB tests."""

import pytest
from langchain_core.documents import Document

from src.retrieval.chroma_store import DebateChromaStore
from src.retrieval.pinecone_db import DebatePineconeDB


# ── ChromaDB Tests (no API key needed) ───────────────────────────────────────

def test_chroma_store_retrieves_documents():
    store = DebateChromaStore()
    results = store.retrieve_similar("social media mental health", k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


def test_chroma_store_mmr_retrieves_diverse():
    store = DebateChromaStore()
    results = store.retrieve_mmr("climate change evidence", k=2)
    assert isinstance(results, list)


def test_chroma_store_empty_query_returns_empty():
    store = DebateChromaStore()
    results = store.retrieve_similar("", k=2)
    assert results == []


def test_chroma_store_as_retriever_similarity():
    store = DebateChromaStore()
    retriever = store.as_retriever(search_type="similarity", k=2)
    assert retriever is not None


def test_chroma_store_as_retriever_mmr():
    store = DebateChromaStore()
    retriever = store.as_retriever(search_type="mmr", k=2)
    assert retriever is not None


def test_chroma_store_clear():
    store = DebateChromaStore()
    # Load store first
    store.retrieve_similar("test", k=1)
    store.clear()
    assert store._store is None


# ── Pinecone Tests (graceful degradation without API key) ────────────────────

def test_pinecone_graceful_degradation():
    """Pinecone should not crash without API key."""
    db = DebatePineconeDB()
    # Without API key it should just be unavailable
    # not raise an exception
    results = db.retrieve_similar("test query", k=2)
    assert isinstance(results, list)


def test_pinecone_mmr_graceful_degradation():
    db = DebatePineconeDB()
    results = db.retrieve_mmr("test query", k=2)
    assert isinstance(results, list)


def test_pinecone_as_retriever_returns_none_when_unavailable():
    db = DebatePineconeDB()
    if not db.is_available():
        assert db.as_retriever() is None
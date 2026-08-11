"""Tests for Sprint 4 memory — no LLM calls for buffer + vector."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.buffer_memory import DebateBufferMemory
from src.memory.vector_memory import DebateVectorMemory


# ── Buffer Memory ─────────────────────────────────────────────────────────────

def test_buffer_stores_messages():
    mem = DebateBufferMemory(window_size=6)
    mem.add_turn("Social media is harmful.", "That is debatable.")
    assert mem.get_message_count() == 2


def test_buffer_window_enforced():
    """Messages beyond window_size are dropped."""
    mem = DebateBufferMemory(window_size=4)
    mem.add_turn("Arg 1", "Resp 1")
    mem.add_turn("Arg 2", "Resp 2")
    mem.add_turn("Arg 3", "Resp 3")
    assert mem.get_message_count() == 4  # window=4, not 6


def test_buffer_returns_message_objects():
    """get_messages() returns HumanMessage/AIMessage objects."""
    mem = DebateBufferMemory(window_size=6)
    mem.add_turn("My argument", "AI response")
    messages = mem.get_messages()
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_buffer_get_as_tuples():
    mem = DebateBufferMemory(window_size=6)
    mem.add_turn("My argument", "AI response")
    tuples = mem.get_as_tuples()
    assert tuples[0] == ("human", "My argument")
    assert tuples[1] == ("assistant", "AI response")


def test_buffer_clear():
    mem = DebateBufferMemory(window_size=6)
    mem.add_turn("Arg", "Resp")
    mem.clear()
    assert mem.get_message_count() == 0


def test_buffer_is_near_limit():
    mem = DebateBufferMemory(window_size=4)
    assert not mem.is_near_limit()
    mem.add_turn("Arg 1", "Resp 1")
    mem.add_turn("Arg 2", "Resp 2")
    # 4 messages = 100% of window_size=4 → near limit
    assert mem.is_near_limit()


def test_buffer_pop_oldest():
    """pop_oldest removes and returns oldest messages."""
    mem = DebateBufferMemory(window_size=6)
    mem.add_turn("First", "First resp")
    mem.add_turn("Second", "Second resp")
    popped = mem.pop_oldest(2)
    assert len(popped) == 2
    assert popped[0].content == "First"
    assert mem.get_message_count() == 2  # second turn remains


# ── Vector Memory ─────────────────────────────────────────────────────────────

def test_vector_stores_argument():
    mem = DebateVectorMemory(top_k=3)
    mem.store_argument("Social media causes depression", 1, "weak", 3.0)
    assert mem.get_argument_count() == 1


def test_vector_is_empty_initially():
    mem = DebateVectorMemory(top_k=3)
    assert mem.is_empty()


def test_vector_find_similar_empty():
    """Returns empty list when no arguments stored."""
    mem = DebateVectorMemory(top_k=3)
    assert mem.find_similar("any argument") == []


def test_vector_clear():
    mem = DebateVectorMemory(top_k=3)
    mem.store_argument("Test argument", 1, "weak", 3.0)
    mem.clear()
    assert mem.is_empty()


def test_vector_stores_with_fallacy_name():
    """store_argument accepts fallacy_name metadata."""
    mem = DebateVectorMemory(top_k=3)
    mem.store_argument(
        "You only support this because you're biased.",
        turn_number=1,
        quality="fallacy",
        score=2.0,
        fallacy_name="ad_hominem",
    )
    assert mem.get_argument_count() == 1


def test_vector_find_similar_returns_list():
    """After storing, similarity search returns a list."""
    mem = DebateVectorMemory(top_k=3)
    mem.store_argument(
        "Social media harms mental health in teenagers",
        1, "weak", 4.0
    )
    mem.store_argument(
        "Apps cause anxiety and depression in young people",
        2, "weak", 3.0
    )
    results = mem.find_similar("Social media negatively affects youth wellbeing")
    assert isinstance(results, list)


def test_vector_top_k_validation():
    """top_k must be at least 1."""
    with pytest.raises(ValueError):
        DebateVectorMemory(top_k=0)
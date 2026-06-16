"""Tests for the summarizer's document normalization.

Regression guard for the bug where a ``Doc`` object (the normal shape of
``state.documents``) fell through to ``str(doc)`` and poisoned ``content`` with
the full ``Doc(...)`` repr — which then flowed into summaries, classifier
embeddings, and CSV exports.
"""

from delve.core.summarizer import _normalize_doc
from delve.state import Doc


def test_normalize_doc_from_doc_object_uses_clean_content():
    doc = Doc(id="abc-123", content="im looking for a cruise from new york")
    out = _normalize_doc(doc)
    assert out["id"] == "abc-123"
    assert out["content"] == "im looking for a cruise from new york"
    # The repr must never leak into content.
    assert "Doc(" not in out["content"]
    assert "summary=None" not in out["content"]


def test_normalize_doc_from_string():
    out = _normalize_doc("raw text")
    assert out["content"] == "raw text"
    assert out["id"]  # a uuid was assigned


def test_normalize_doc_from_dict_preserves_id_and_content():
    out = _normalize_doc({"id": "keep-me", "content": "hello"})
    assert out == {"id": "keep-me", "content": "hello"}


def test_normalize_doc_from_dict_without_id_gets_one():
    out = _normalize_doc({"content": "hello"})
    assert out["content"] == "hello"
    assert out["id"]


def test_normalize_doc_object_without_id_gets_uuid():
    doc = Doc(id="", content="text here")
    out = _normalize_doc(doc)
    assert out["content"] == "text here"
    assert out["id"]  # falsy id replaced with a fresh uuid

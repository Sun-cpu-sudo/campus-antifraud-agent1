"""SQLite 存储单元测试。"""
import tempfile
from pathlib import Path

from src.antifraud.storage import Storage


def test_log_and_review_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = Storage(Path(d) / "test.db")
        pid = s.log_prediction({
            "text_hash": "abc", "text_masked": "138****5678",
            "final_score": 0.9, "ml_prob": 0.9, "rule_score": 0.8,
            "risk_level": "高风险", "ml_type": "冒充公检法及政府机关类",
            "reasons": ["理由"],
        }, sender="10690")
        assert s.prediction_count() == 1
        pending = s.pending_reviews()
        assert len(pending) == 1
        s.add_review(pid, "spam")
        assert s.pending_reviews() == []
        stats = s.review_stats()
        assert stats["spam"] == 1


def test_clear():
    with tempfile.TemporaryDirectory() as d:
        s = Storage(Path(d) / "test.db")
        s.log_prediction({"text_hash": "x", "text_masked": "y",
                          "risk_level": "低风险", "reasons": []})
        s.clear()
        assert s.prediction_count() == 0

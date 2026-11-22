"""SQLite 持久化：检测日志与人工复核队列。"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATA_DIR


class Storage:
    """本地 SQLite 存储（默认仅保存脱敏文本与哈希，保护隐私）。"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DATA_DIR / "agent.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, sql: str, params: tuple = ()):
        conn = self._conn()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        conn = self._conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return rows
        finally:
            conn.close()

    def _init_db(self) -> None:
        self._execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text_masked TEXT NOT NULL,
                sender TEXT,
                final_score REAL,
                ml_prob REAL,
                rule_score REAL,
                risk_level TEXT,
                ml_type TEXT,
                reasons_json TEXT,
                store_raw INTEGER DEFAULT 0
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                ts TEXT NOT NULL,
                review_label TEXT,
                note TEXT,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        """)

    def log_prediction(self, result: Dict, sender: Optional[str] = None,
                       store_raw: bool = False) -> int:
        cur = self._execute("""
            INSERT INTO predictions (ts, text_hash, text_masked, sender,
                final_score, ml_prob, rule_score, risk_level, ml_type,
                reasons_json, store_raw)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            result.get("text_hash", ""),
            result.get("text_masked", ""),
            sender,
            result.get("final_score"),
            result.get("ml_prob"),
            result.get("rule_score"),
            result.get("risk_level"),
            result.get("ml_type"),
            json.dumps(result.get("reasons", []), ensure_ascii=False),
            1 if store_raw else 0,
        ))
        return int(cur.lastrowid)

    def add_review(self, prediction_id: int, review_label: str,
                   note: str = "") -> None:
        self._execute("""
            INSERT INTO reviews (prediction_id, ts, review_label, note)
            VALUES (?,?,?,?)
        """, (prediction_id, time.strftime("%Y-%m-%d %H:%M:%S"),
              review_label, note))

    def pending_reviews(self) -> List[Dict]:
        rows = self._query("""
            SELECT p.id, p.ts, p.text_masked, p.sender, p.risk_level,
                   p.final_score, p.ml_prob, p.rule_score, p.ml_type,
                   p.reasons_json
            FROM predictions p
            WHERE p.id NOT IN (SELECT prediction_id FROM reviews)
            ORDER BY p.id DESC
            LIMIT 200
        """)
        return [self._row_to_dict(r) for r in rows]

    def review_stats(self) -> Dict:
        row = self._query("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN review_label='spam' THEN 1 ELSE 0 END) AS spam,
                   SUM(CASE WHEN review_label='ham' THEN 1 ELSE 0 END) AS ham
            FROM reviews
        """)[0]
        return {"total": row["total"] or 0, "spam": row["spam"] or 0,
                "ham": row["ham"] or 0}

    def prediction_count(self) -> int:
        return int(self._query("SELECT COUNT(*) FROM predictions")[0][0])

    @staticmethod
    def _row_to_dict(r) -> Dict:
        d = dict(r)
        d["reasons"] = json.loads(d.get("reasons_json") or "[]")
        d.pop("reasons_json", None)
        return d

    def clear(self) -> None:
        self._execute("DELETE FROM reviews")
        self._execute("DELETE FROM predictions")

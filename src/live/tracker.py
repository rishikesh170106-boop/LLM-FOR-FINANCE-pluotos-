"""
src/live/tracker.py
Phase 11: Real-Time Live Forward Tracking & Performance Audit Engine.
Stores live Ploutos+ model predictions, monitors forward market outcomes,
and evaluates hit/miss rates once next-day market data settles.
"""
import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "results" / "live_tracker.db"
JSON_AUDIT_PATH = PROJECT_ROOT / "results" / "metrics" / "live_audit_summary.json"


class PredictionTracker:
    """
    Manages storage, tracking, and forward evaluation of live predictions.
    Supports both SQLite persistent storage and export to JSON audit metrics.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite schema for forward tracking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    last_close REAL NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    w_tech REAL NOT NULL,
                    w_sent REAL NOT NULL,
                    w_human REAL NOT NULL,
                    dominant_expert TEXT NOT NULL,
                    chart_pattern TEXT,
                    chart_signal TEXT,
                    rationale TEXT,
                    realized_next_close REAL,
                    realized_return REAL,
                    actual_direction TEXT,
                    is_correct INTEGER,
                    investment_verdict TEXT DEFAULT 'HOLD_CASH',
                    stop_loss REAL,
                    take_profit REAL,
                    risk_reward REAL,
                    status TEXT DEFAULT 'PENDING'
                )
            """)

            # Add migration columns dynamically if table already exists
            cursor.execute("PRAGMA table_info(predictions)")
            cols = [info[1] for info in cursor.fetchall()]
            if "investment_verdict" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN investment_verdict TEXT DEFAULT 'HOLD_CASH'")
            if "stop_loss" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN stop_loss REAL")
            if "take_profit" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN take_profit REAL")
            if "risk_reward" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN risk_reward REAL")
            conn.commit()

    def log_prediction(
        self,
        ticker: str,
        last_close: float,
        predicted_direction: str,
        predicted_confidence: float,
        weights: Dict[str, float],
        dominant_expert: str,
        chart_pattern: str = "Unknown",
        chart_signal: str = "NEUTRAL",
        rationale: str = "",
        prediction_date: Optional[str] = None,
        investment_verdict: str = "HOLD_CASH",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_reward: Optional[float] = None
    ) -> int:
        """Logs a new live prediction to the database with PENDING status."""
        ts = datetime.now().isoformat()
        pred_date = prediction_date or date.today().isoformat()

        w_human = float(weights.get("human", weights.get("human_proxy", 0.34)))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, ticker, prediction_date, last_close,
                    predicted_direction, predicted_confidence,
                    w_tech, w_sent, w_human, dominant_expert,
                    chart_pattern, chart_signal, rationale,
                    investment_verdict, stop_loss, take_profit, risk_reward,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """, (
                ts,
                ticker.upper(),
                pred_date,
                float(last_close),
                predicted_direction.upper(),
                float(predicted_confidence),
                float(weights.get("technical", 0.33)),
                float(weights.get("sentiment", 0.33)),
                w_human,
                dominant_expert,
                chart_pattern,
                chart_signal,
                rationale,
                investment_verdict,
                float(stop_loss) if stop_loss is not None else None,
                float(take_profit) if take_profit is not None else None,
                float(risk_reward) if risk_reward is not None else None
            ))
            pred_id = cursor.lastrowid
            conn.commit()
        return pred_id

    def get_pending_predictions(self) -> List[Dict[str, Any]]:
        """Retrieves all predictions currently waiting for next-day market settlement."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE status = 'PENDING'")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_all_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns historical predictions ordered by timestamp descending."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def update_realized_outcome(
        self,
        prediction_id: int,
        realized_next_close: float
    ) -> Dict[str, Any]:
        """
        Updates a pending prediction with the realized next-day close price,
        computes realized return and whether the directional forecast was correct.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Prediction ID {prediction_id} not found.")

            last_close = row["last_close"]
            pred_dir = row["predicted_direction"]

            # Realized return
            ret = (realized_next_close - last_close) / (last_close + 1e-9)
            actual_dir = "BULLISH" if ret >= 0.0 else "BEARISH"

            # Check if directional forecast matched
            # Ploutos+ predicts BULLISH (Up) or BEARISH (Down)
            is_correct = 1 if (pred_dir == actual_dir) else 0

            cursor.execute("""
                UPDATE predictions
                SET realized_next_close = ?,
                    realized_return = ?,
                    actual_direction = ?,
                    is_correct = ?,
                    status = 'SETTLED'
                WHERE id = ?
            """, (float(realized_next_close), float(ret), actual_dir, is_correct, prediction_id))
            conn.commit()

            return {
                "id": prediction_id,
                "ticker": row["ticker"],
                "predicted_direction": pred_dir,
                "actual_direction": actual_dir,
                "realized_return": round(ret, 4),
                "is_correct": bool(is_correct)
            }

    def evaluate_pending(self, mock_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Scans pending predictions and attempts to settle them with the latest market data.
        If mock_prices is provided (for tests/offline), uses those prices.
        """
        pending = self.get_pending_predictions()
        settled_count = 0
        results = []

        for p in pending:
            ticker = p["ticker"]
            next_close = None

            if mock_prices and ticker in mock_prices:
                next_close = mock_prices[ticker]
            else:
                try:
                    import yfinance as yf
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d")
                    if len(hist) >= 2:
                        # Check if latest bar is newer than prediction date
                        latest_date = hist.index[-1].strftime("%Y-%m-%d")
                        if latest_date > p["prediction_date"]:
                            next_close = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            if next_close is not None:
                res = self.update_realized_outcome(p["id"], next_close)
                results.append(res)
                settled_count += 1

        summary = self.compute_audit_summary()
        return {
            "pending_checked": len(pending),
            "settled_now": settled_count,
            "settled_results": results,
            "summary": summary
        }

    def compute_audit_summary(self) -> Dict[str, Any]:
        """Calculates cumulative hit rate, win rate, and directional accuracy."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'SETTLED'")
            total_settled = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'PENDING'")
            total_pending = cursor.fetchone()[0]

            if total_settled > 0:
                cursor.execute("SELECT SUM(is_correct) FROM predictions WHERE status = 'SETTLED'")
                correct_count = cursor.fetchone()[0] or 0
                accuracy = correct_count / total_settled

                cursor.execute("SELECT AVG(realized_return) FROM predictions WHERE status = 'SETTLED'")
                avg_ret = cursor.fetchone()[0] or 0.0

                cursor.execute("SELECT AVG(predicted_confidence) FROM predictions")
                avg_conf = cursor.fetchone()[0] or 0.0
            else:
                correct_count = 0
                accuracy = 0.0
                avg_ret = 0.0
                cursor.execute("SELECT AVG(predicted_confidence) FROM predictions")
                avg_conf = cursor.fetchone()[0] or 0.0

            summary = {
                "total_predictions": total_predictions,
                "total_settled": total_settled,
                "total_pending": total_pending,
                "settled_correct": correct_count,
                "forward_accuracy": round(float(accuracy), 4),
                "avg_realized_return": round(float(avg_ret), 4),
                "avg_predicted_confidence": round(float(avg_conf or 0.5), 4),
                "updated_at": datetime.now().isoformat()
            }

            JSON_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JSON_AUDIT_PATH, "w") as f:
                json.dump(summary, f, indent=2)

            return summary


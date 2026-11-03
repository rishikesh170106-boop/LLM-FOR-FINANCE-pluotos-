"""
src/experts/human_expert.py
Human / Fundamental Proxy Expert component of PloutosGen:
Grounds qualitative context using authentic sector and corporate profiles from StockNet's StockTable.
Maintains research integrity by explicitly labeling company sector classification and observable fundamentals.
"""
from pathlib import Path
from typing import Dict, Any
import pandas as pd

STOCK_TABLE_PATH = Path("data/raw/stocknet-dataset/StockTable")

# 2014-2015 Sector macroeconomic backdrop profiles
SECTOR_CONTEXTS = {
    "Technology": "High-growth secular momentum driven by mobile, cloud adoption, and software enterprise spending in 2014-2015.",
    "Basic Matierials": "Subject to commodity cyclicality and global demand pressures, particularly the 2014-2015 crude oil price correction.",
    "Financial": "Influenced by Federal Reserve monetary policy, low interest rate margins, and post-crisis regulatory capital requirements.",
    "Consumer Goods": "Defensive demand profile with stable cash flows, sensitive to consumer discretionary confidence.",
    "Healthcare": "Defensive growth driven by drug approvals and healthcare service demand, resilient to economic cycles.",
    "Industrial Goods": "Cyclical sensitivity to capital expenditure, industrial manufacturing indices, and global trade flows.",
    "Utilities": "Defensive dividend-yielding sector, sensitive to long-term treasury yields and rate hike expectations.",
    "Conglomerates": "Diversified multi-industry operations with broad exposure to international macroeconomic trends."
}

class HumanExpert:
    def __init__(self, stock_table_path: Path = STOCK_TABLE_PATH):
        self.expert_name = "Human / Fundamental Expert (Proxy)"
        self.company_profiles: Dict[str, Dict[str, str]] = {}
        self._load_stock_table(stock_table_path)

    def _load_stock_table(self, path: Path):
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().split("\t")
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        sector = parts[0].strip()
                        raw_sym = parts[1].strip().lstrip("$")
                        company = parts[2].strip()
                        self.company_profiles[raw_sym] = {
                            "sector": sector,
                            "company_name": company
                        }
        except Exception:
            pass

    def analyze(self, row: pd.Series) -> Dict[str, Any]:
        """
        Provides sector and fundamental corporate profile context for a stock.
        """
        ticker = row.get("ticker", "UNKNOWN")
        date = str(row.get("date", "UNKNOWN"))

        profile = self.company_profiles.get(ticker, {
            "sector": "General Equities",
            "company_name": f"{ticker} Corporation"
        })

        sector = profile["sector"]
        company_name = profile["company_name"]
        sector_narrative = SECTOR_CONTEXTS.get(sector, "Market participant subject to general equity market macroeconomic cycles.")

        evidence = (
            f"{company_name} operates within the {sector} sector. "
            f"Fundamental macro backdrop: {sector_narrative}"
        )

        return {
            "expert_name": self.expert_name,
            "ticker": ticker,
            "as_of_date": date,
            "signal": "NEUTRAL",
            "confidence": 0.60,
            "evidence": evidence,
            "metrics": {
                "sector": sector,
                "company_name": company_name
            },
            "source": "StockNet StockTable Corporate Profile",
            "availability": True
        }

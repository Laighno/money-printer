"""Local data storage with SQLite."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text


class DataStore:
    def __init__(self, db_url: str = "sqlite:///data/market.db"):
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url)
        self._init_tables()

    def _init_tables(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_bars (
                    code TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL,
                    PRIMARY KEY (code, date)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS industry_bars (
                    board_name TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    change_pct REAL, volume REAL, amount REAL, turnover REAL,
                    PRIMARY KEY (board_name, date)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS valuation (
                    code TEXT NOT NULL, date TEXT NOT NULL,
                    name TEXT, close REAL, pe_ttm REAL, pb REAL, total_mv REAL,
                    PRIMARY KEY (code, date)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS financial (
                    code TEXT NOT NULL, report_date TEXT NOT NULL,
                    publish_date TEXT,
                    roe REAL, eps REAL, bps REAL, debt_ratio REAL,
                    gross_margin REAL, net_margin REAL,
                    revenue_growth REAL, profit_growth REAL,
                    PRIMARY KEY (code, report_date)
                )
            """))
            conn.commit()

    # --- daily bars ---

    def save_bars(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["date"] = df["date"].astype(str)
        df.to_sql("daily_bars", self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Saved {len(df)} rows to daily_bars")

    @staticmethod
    def _normalize_date(d: str) -> str:
        """Normalize YYYYMMDD to YYYY-MM-DD for consistent DB queries."""
        if len(d) == 8 and "-" not in d:
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return d

    def load_bars(self, codes: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM daily_bars WHERE 1=1"
        params: dict = {}
        if codes:
            placeholders = ",".join(f":c{i}" for i in range(len(codes)))
            query += f" AND code IN ({placeholders})"
            for i, c in enumerate(codes):
                params[f"c{i}"] = c
        if start:
            query += " AND date >= :start"
            params["start"] = self._normalize_date(start)
        if end:
            query += " AND date <= :end"
            params["end"] = self._normalize_date(end)
        query += " ORDER BY code, date"
        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # --- industry bars ---

    def save_industry_bars(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["date"] = df["date"].astype(str)
        # Clear existing data and re-insert (handles duplicates on refresh)
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM industry_bars"))
            conn.commit()
        df.to_sql("industry_bars", self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Saved {len(df)} rows to industry_bars")

    def load_industry_bars(self, board_names: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM industry_bars WHERE 1=1"
        params: dict = {}
        if board_names:
            placeholders = ",".join(f":b{i}" for i in range(len(board_names)))
            query += f" AND board_name IN ({placeholders})"
            for i, b in enumerate(board_names):
                params[f"b{i}"] = b
        if start:
            query += " AND date >= :start"
            params["start"] = start
        if end:
            query += " AND date <= :end"
            params["end"] = end
        query += " ORDER BY board_name, date"
        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def has_industry_bars(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM industry_bars")).scalar() > 0

    # --- valuation ---

    def save_valuation(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["date"] = df["date"].astype(str)
        with self.engine.connect() as conn:
            for d in df["date"].unique():
                conn.execute(text("DELETE FROM valuation WHERE date = :d"), {"d": d})
            conn.commit()
        df.to_sql("valuation", self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Saved {len(df)} valuation rows")

    def load_valuation(self, codes: list[str] | None = None, date_str: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM valuation WHERE 1=1"
        params: dict = {}
        if date_str:
            query += " AND date = :date"
            params["date"] = date_str
        else:
            query += " AND date = (SELECT MAX(date) FROM valuation)"
        if codes:
            placeholders = ",".join(f":c{i}" for i in range(len(codes)))
            query += f" AND code IN ({placeholders})"
            for i, c in enumerate(codes):
                params[f"c{i}"] = c
        return pd.read_sql(text(query), self.engine, params=params)

    def has_valuation_data(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM valuation")).scalar() > 0

    # --- financial (with publish_date for time-alignment) ---

    def save_financial(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["report_date"] = df["report_date"].astype(str)
        df["publish_date"] = df["publish_date"].astype(str)
        with self.engine.connect() as conn:
            for _, row in df[["code", "report_date"]].drop_duplicates().iterrows():
                conn.execute(
                    text("DELETE FROM financial WHERE code = :code AND report_date = :rd"),
                    {"code": row["code"], "rd": row["report_date"]},
                )
            conn.commit()
        df.to_sql("financial", self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Saved {len(df)} financial rows")

    def load_financial_at(self, as_of_date: str, codes: list[str] | None = None) -> pd.DataFrame:
        """Load latest financial data as of a given date, respecting publish_date (no look-ahead).

        Only returns reports whose publish_date <= as_of_date.
        """
        query = """
            SELECT f.* FROM financial f
            INNER JOIN (
                SELECT code, MAX(report_date) AS max_rd FROM financial
                WHERE publish_date <= :as_of
                GROUP BY code
            ) latest ON f.code = latest.code AND f.report_date = latest.max_rd
            WHERE 1=1
        """
        params: dict = {"as_of": as_of_date}
        if codes:
            placeholders = ",".join(f":c{i}" for i in range(len(codes)))
            query += f" AND f.code IN ({placeholders})"
            for i, c in enumerate(codes):
                params[f"c{i}"] = c
        return pd.read_sql(text(query), self.engine, params=params)

    def load_financial(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Load latest financial report per stock (no time constraint)."""
        query = """
            SELECT f.* FROM financial f
            INNER JOIN (
                SELECT code, MAX(report_date) AS max_rd FROM financial GROUP BY code
            ) latest ON f.code = latest.code AND f.report_date = latest.max_rd
            WHERE 1=1
        """
        params: dict = {}
        if codes:
            placeholders = ",".join(f":c{i}" for i in range(len(codes)))
            query += f" AND f.code IN ({placeholders})"
            for i, c in enumerate(codes):
                params[f"c{i}"] = c
        return pd.read_sql(text(query), self.engine, params=params)

    def has_financial_data(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM financial")).scalar() > 0

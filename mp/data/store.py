"""Local data storage with SQLite."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = "sqlite:///data/market.db"


def _resolve_db_url(explicit: str | None = None) -> str:
    """Resolve the DB URL with this priority:

    1. ``explicit`` argument (highest)
    2. ``MP_DB_PATH`` environment variable (path or sqlite:/// URL)
    3. ``DEFAULT_DB_URL`` (production default)

    Tests can monkeypatch ``MP_DB_PATH`` to redirect the store to a temp file
    without modifying call sites.
    """
    if explicit:
        return explicit
    env = os.environ.get("MP_DB_PATH")
    if env:
        return env if env.startswith("sqlite") else f"sqlite:///{env}"
    return DEFAULT_DB_URL


class DataStore:
    def __init__(self, db_url: str | None = None):
        url = _resolve_db_url(db_url)
        db_path = url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url)
        self._init_tables()

    def _init_tables(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_bars (
                    code TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL,
                    turnover REAL,
                    PRIMARY KEY (code, date)
                )
            """))
            # Add turnover column to existing DBs that were created before it was added
            try:
                conn.execute(text("ALTER TABLE daily_bars ADD COLUMN turnover REAL"))
                conn.commit()
            except Exception:
                pass  # column already exists
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
            # Idempotent migration: older DBs created before publish_date was
            # added will silently drop it on insert and corrupt point-in-time
            # alignment on DB-fallback paths.  Add it if missing.
            try:
                conn.execute(text("ALTER TABLE financial ADD COLUMN publish_date TEXT"))
                conn.commit()
                logger.info("Migrated financial table: added publish_date column")
            except Exception:
                pass  # column already exists
            # Constituent snapshots: one row per (index, date, stock_code).
            # Accumulates quarterly over time; used to build a time-varying
            # backtest universe and mitigate survivorship bias.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS constituent_snapshots (
                    index_name TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    PRIMARY KEY (index_name, snapshot_date, code)
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

    def save_bars_upsert(self, df: pd.DataFrame) -> int:
        """Insert or replace bars — safe for incremental updates without duplicates.

        Returns the number of rows written.

        All schema enforcement is delegated to :func:`mp.data.schema.validate_bars`,
        which auto-normalizes obvious unit drift (e.g. turnover > 1 → /100),
        drops rows violating hard physical bounds, and rejects rows where
        amount/(volume*close) shows a unit mismatch.

        Per-stock drift detection (>50× the trailing 30-day median) is also
        applied as the final defense against single-row unit errors that
        slip through the batch-level checks.
        """
        if df.empty:
            return 0
        df = df.copy()
        df["date"] = df["date"].astype(str)
        cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]

        # Schema-enforced validation (Layer 3 of the unit moat).
        # Auto-normalizes turnover > 1 (back-stop for un-registered sources),
        # drops rows violating hard physical bounds, and drops rows with
        # amount/(volume*close) inconsistency.
        from mp.data.schema import detect_per_stock_drift, validate_bars

        df = validate_bars(df)
        if df.empty:
            return 0

        # Per-stock drift detection (Layer 4): catches single-row unit errors
        # that slip past batch checks — e.g. one stock from a flaky source
        # whose value is normal-magnitude but 100× its own historical norm.
        df = detect_per_stock_drift(
            df,
            historical_median_lookup=self._trailing_median,
            threshold=50.0,
        )

        written = 0
        with self.engine.connect() as conn:
            for chunk_start in range(0, len(df), 500):
                chunk = df.iloc[chunk_start:chunk_start + 500]
                values_placeholder = ",".join(["(?,?,?,?,?,?,?,?,?)"] * len(chunk))
                flat_values = []
                for row in chunk.itertuples(index=False):
                    flat_values.extend([
                        row.code, row.date,
                        _real(row.open), _real(row.high), _real(row.low), _real(row.close),
                        _real(row.volume), _real(row.amount), _real(row.turnover),
                    ])
                conn.exec_driver_sql(
                    f"INSERT OR REPLACE INTO daily_bars VALUES {values_placeholder}",
                    tuple(flat_values),
                )
                written += len(chunk)
            conn.commit()
        logger.debug(f"Upserted {written} bar rows")
        return written

    def get_max_bar_date(self, code: str) -> str | None:
        """Return the latest date string stored for *code*, or None if no data.

        Returned format: 'YYYY-MM-DD'.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(date) FROM daily_bars WHERE code = :code"),
                {"code": code},
            ).scalar()
        return result  # e.g. "2025-04-10" or None

    def _trailing_median(self, code: str, col: str, days: int = 30) -> float | None:
        """Return the median of *col* over the last *days* rows for *code*.

        Used by :func:`mp.data.schema.detect_per_stock_drift` to catch single
        rows whose value is >50× the stock's own historical median —
        characteristic of unit drift in a single source/day.
        Returns None if the column doesn't exist or no history is available.
        """
        if col not in {"volume", "amount", "turnover"}:
            return None
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f"SELECT {col} FROM daily_bars "
                        f"WHERE code = :code AND {col} IS NOT NULL AND {col} > 0 "
                        f"ORDER BY date DESC LIMIT :n"
                    ),
                    {"code": code, "n": days},
                ).fetchall()
            vals = [r[0] for r in rows if r[0] is not None]
            if not vals:
                return None
            import statistics
            return float(statistics.median(vals))
        except Exception as e:
            logger.debug(f"_trailing_median({code}, {col}) failed: {e}")
            return None

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
        """Upsert industry bar rows (INSERT OR REPLACE per board+date)."""
        if df.empty:
            return
        df = df.copy()
        df["date"] = df["date"].astype(str)
        cols = ["board_name", "date", "open", "high", "low", "close",
                "change_pct", "volume", "amount", "turnover"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]

        written = 0
        with self.engine.connect() as conn:
            for chunk_start in range(0, len(df), 500):
                chunk = df.iloc[chunk_start:chunk_start + 500]
                values_placeholder = ",".join(["(?,?,?,?,?,?,?,?,?,?)"] * len(chunk))
                flat_values = []
                for row in chunk.itertuples(index=False):
                    flat_values.extend([
                        row.board_name, row.date,
                        _real(row.open), _real(row.high), _real(row.low), _real(row.close),
                        _real(row.change_pct), _real(row.volume), _real(row.amount), _real(row.turnover),
                    ])
                conn.exec_driver_sql(
                    f"INSERT OR REPLACE INTO industry_bars VALUES {values_placeholder}",
                    tuple(flat_values),
                )
                written += len(chunk)
            conn.commit()
        logger.info(f"Saved {written} rows to industry_bars")

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
        """Upsert valuation rows by (code, date).

        Previously this method deleted ALL rows for each date before inserting,
        which was safe only when the caller passed a full cross-sectional
        snapshot. For per-stock historical backfills the old behaviour would
        wipe previously-saved stocks.  We now delete by (code, date) pair.
        """
        if df.empty:
            return
        df = df.copy()
        df["date"] = df["date"].astype(str)
        df["code"] = df["code"].astype(str).str.zfill(6)
        pairs = df[["code", "date"]].drop_duplicates()
        with self.engine.connect() as conn:
            # Batch delete by (code, date) — chunked to keep SQL param count sane.
            rows = list(pairs.itertuples(index=False, name=None))
            CHUNK = 500
            for i in range(0, len(rows), CHUNK):
                chunk = rows[i : i + CHUNK]
                placeholders = ",".join(
                    f"(:c{j}, :d{j})" for j in range(len(chunk))
                )
                params: dict = {}
                for j, (code, d) in enumerate(chunk):
                    params[f"c{j}"] = code
                    params[f"d{j}"] = d
                conn.execute(
                    text(
                        f"DELETE FROM valuation WHERE (code, date) IN ({placeholders})"
                    ),
                    params,
                )
            conn.commit()
        df.to_sql("valuation", self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Saved {len(df)} valuation rows ({len(pairs)} unique (code,date))")

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

    def load_valuation_history(
        self,
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load valuation rows across a date range.

        Used for training-time alignment of daily PE/PB/market-cap, in contrast
        to :meth:`load_valuation` which only returns a single date's snapshot.
        """
        query = "SELECT code, date, pe_ttm, pb, total_mv FROM valuation WHERE 1=1"
        params: dict = {}
        if start:
            query += " AND date >= :start"
            params["start"] = str(start)
        if end:
            query += " AND date <= :end"
            params["end"] = str(end)
        if codes:
            # Normalise to 6-digit strings on both sides of the IN clause.
            codes = [str(c).zfill(6) for c in codes]
            placeholders = ",".join(f":c{i}" for i in range(len(codes)))
            query += f" AND code IN ({placeholders})"
            for i, c in enumerate(codes):
                params[f"c{i}"] = c
        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty:
            df["code"] = df["code"].astype(str).str.zfill(6)
        return df

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

    # --- constituent snapshots ---

    def save_constituent_snapshot(
        self,
        index_name: str,
        codes: list[str],
        snapshot_date: str | None = None,
    ) -> None:
        """Persist a point-in-time index constituent list.

        Parameters
        ----------
        index_name:
            Short name, e.g. ``"hs300"``, ``"zz500"``.
        codes:
            List of 6-digit stock codes in the index on *snapshot_date*.
        snapshot_date:
            ISO date string ``"YYYY-MM-DD"``; defaults to today.
        """
        from datetime import date as _date
        if snapshot_date is None:
            snapshot_date = _date.today().isoformat()
        if not codes:
            return
        with self.engine.connect() as conn:
            for chunk_start in range(0, len(codes), 500):
                chunk = codes[chunk_start:chunk_start + 500]
                placeholders = ",".join(["(?,?,?)"] * len(chunk))
                flat = []
                for c in chunk:
                    flat.extend([index_name, snapshot_date, c])
                conn.exec_driver_sql(
                    f"INSERT OR REPLACE INTO constituent_snapshots VALUES {placeholders}",
                    tuple(flat),
                )
            conn.commit()
        logger.info("Saved constituent snapshot: {} {} codes for {}", index_name, len(codes), snapshot_date)

    def load_constituent_snapshot_at(
        self,
        index_name: str,
        as_of_date: str,
    ) -> list[str] | None:
        """Return the most recent constituent list whose snapshot_date ≤ as_of_date.

        Returns ``None`` if no snapshot exists on or before *as_of_date*.
        """
        with self.engine.connect() as conn:
            best_date = conn.execute(
                text("""
                    SELECT MAX(snapshot_date) FROM constituent_snapshots
                    WHERE index_name = :idx AND snapshot_date <= :asof
                """),
                {"idx": index_name, "asof": as_of_date},
            ).scalar()
            if best_date is None:
                return None
            rows = conn.execute(
                text("""
                    SELECT code FROM constituent_snapshots
                    WHERE index_name = :idx AND snapshot_date = :d
                    ORDER BY code
                """),
                {"idx": index_name, "d": best_date},
            ).fetchall()
        codes = [r[0] for r in rows]
        logger.debug("Loaded {} constituent snapshot for {}: {} codes (as_of {})",
                     index_name, best_date, len(codes), as_of_date)
        return codes if codes else None

    def list_constituent_snapshot_dates(self, index_name: str) -> list[str]:
        """Return all snapshot dates stored for *index_name*, ascending."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT DISTINCT snapshot_date FROM constituent_snapshots
                    WHERE index_name = :idx ORDER BY snapshot_date
                """),
                {"idx": index_name},
            ).fetchall()
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _real(v) -> float | None:
    """Convert a value to Python float, returning None for NaN/None."""
    try:
        f = float(v)
        import math
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

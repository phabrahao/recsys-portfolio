# 01_feature_pipeline/event_stream.py
from __future__ import annotations
import duckdb
import polars as pl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Event:
    user_id: int
    movie_id: int
    event_ts: int
    event_type: str = "rating"          # "rating" | "impression" | "feedback" | "reward"
    source: str = "historical_replay"   # "historical_replay" | "live"
    rating: float | None = None         # hot field: used by nearly every node-1 feature
    payload: str | None = None          # JSON blob: node-5 reward/propensity/action fields


class EventStream:
    """
    Append-only event log backed by DuckDB, replayable in strict timestamp order.
    Node 1 uses .replay() to feed the feature pipeline off ratings.csv.
    Node 5 later uses .append() to write bandit feedback into the SAME table,
    so the feature pipeline's read interface never changes.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                user_id     BIGINT NOT NULL,
                movie_id    BIGINT NOT NULL,
                event_ts    BIGINT NOT NULL,
                event_type  VARCHAR NOT NULL,
                source      VARCHAR NOT NULL,
                rating      DOUBLE,
                payload     VARCHAR
            )
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(event_ts)")

    def load_historical(self, ratings_csv: str | Path, force: bool = False) -> int:
        existing = self.con.execute(
            "SELECT count(*) FROM events WHERE source = 'historical_replay'"
        ).fetchone()[0]
        if existing > 0 and not force:
            print(f"events table already has {existing} historical rows — skipping "
                f"(pass force=True, or delete the .duckdb file, to reload)")
            return existing    
        ratings_csv = Path(ratings_csv)
        self.con.execute(f"""
            INSERT INTO events
            SELECT
                userId  AS user_id,
                movieId AS movie_id,
                CAST(timestamp AS BIGINT) AS event_ts,
                'rating' AS event_type,
                'historical_replay' AS source,
                rating,
                NULL AS payload
            FROM read_csv_auto('{ratings_csv.as_posix()}')
        """)
        return self.con.execute(
            "SELECT count(*) FROM events WHERE source = 'historical_replay'"
        ).fetchone()[0]

    def replay(self, batch_size: int = 100_000) -> Iterator[pl.DataFrame]:
        """
        Keyset-paginated (not OFFSET-paginated) chunked replay in timestamp order.
        OFFSET N forces DuckDB to re-walk N rows on every call — quadratic at 32M rows.
        Keyset uses the (event_ts, user_id, movie_id) tuple as a cursor instead, so
        every batch is a fresh indexed range scan regardless of how far in we are.
        """
        cursor = (-1, -1, -1)
        while True:
            batch = self.con.execute(
                """
                SELECT * FROM events
                WHERE (event_ts, user_id, movie_id) > (?, ?, ?)
                ORDER BY event_ts, user_id, movie_id
                LIMIT ?
                """,
                [*cursor, batch_size],
            ).pl()
            if batch.is_empty():
                return
            yield batch
            last = batch.tail(1)
            cursor = (last["event_ts"][0], last["user_id"][0], last["movie_id"][0])

    def append(self, event: Event) -> None:
        """Live write path — node 5's feedback loop calls this."""
        self.con.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                event.user_id, event.movie_id, event.event_ts,
                event.event_type, event.source, event.rating, event.payload,
            ],
        )

    def close(self) -> None:
        self.con.close()
"""
Node 1 pipeline script: read data/processed/events.duckdb, compute the
current feature set (see recsys_portfolio.features), write the result to
data/processed/features.parquet.

Run from repo root: uv run python 01_feature_pipeline/build_features.py
"""
import duckdb
from pathlib import Path
from recsys_portfolio.features import build_feature_table, DEFAULT_COLD_START_THRESHOLD

EVENTS_DB_PATH = Path("data/processed/events.duckdb")
FEATURES_PARQUET_PATH = Path("data/processed/features.parquet")


def main(
    cold_start_threshold: int = DEFAULT_COLD_START_THRESHOLD,
    genre_fatigue_window_days: int = 30,
    session_gap_seconds: int = 1800,
) -> None:
    if not EVENTS_DB_PATH.exists():
        raise FileNotFoundError(
            f"{EVENTS_DB_PATH} not found -- load historical data first, e.g.:\n"
            f"  uv run python -c \"from recsys_portfolio.event_stream import EventStream; "
            f"es = EventStream('{EVENTS_DB_PATH}'); "
            f"es.load_historical('data/raw/ml-32m/ratings.csv'); es.close()\""
        )

    FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # read_only=True: this script only reads events, never writes -- keeps it
    # safe to run repeatedly and safe to run while something else holds a
    # write connection to the same file (e.g. an EventStream instance).
    con = duckdb.connect(str(EVENTS_DB_PATH), read_only=True)

    rel = build_feature_table(
        con,
        cold_start_threshold=cold_start_threshold,
        genre_fatigue_window_days=genre_fatigue_window_days,
        session_gap_seconds=session_gap_seconds,
        movies_csv_path="data/raw/ml-32m/movies.csv",
    )

    # COPY runs the whole window-function query and streams the result
    # straight to Parquet -- avoids materializing all 32M rows into a
    # Python-side DataFrame first.
    rel.query("final_features", f"""
        COPY (SELECT * FROM final_features)
        TO '{FEATURES_PARQUET_PATH.as_posix()}' (FORMAT PARQUET)
    """)

    row_count = con.sql(f"SELECT count(*) FROM '{FEATURES_PARQUET_PATH.as_posix()}'").fetchone()[0]
    cold_start_share = con.sql(f"""
        SELECT avg(is_cold_start_at_this_point::INT)
        FROM '{FEATURES_PARQUET_PATH.as_posix()}'
    """).fetchone()[0]
    fatigue_stats = con.sql(f"""
        SELECT avg(genre_fatigue_score), max(genre_fatigue_score)
        FROM '{FEATURES_PARQUET_PATH.as_posix()}'
    """).fetchone()
    session_stats = con.sql(f"""
        SELECT
            avg(is_new_session::INT),
            median(seconds_since_last_event)
        FROM '{FEATURES_PARQUET_PATH.as_posix()}'
    """).fetchone()

    print(f"wrote {row_count:,} rows to {FEATURES_PARQUET_PATH}")
    print(f"cold_start_threshold={cold_start_threshold} -> "
          f"{cold_start_share:.2%} of events flagged cold-start")
    print(f"genre_fatigue_window_days={genre_fatigue_window_days} -> "
          f"mean={fatigue_stats[0]:.3f}, max={fatigue_stats[1]:.1f}")
    print(f"session_gap_seconds={session_gap_seconds} -> "
          f"{session_stats[0]:.2%} of events start a new session, "
          f"median gap (non-null)={session_stats[1]}s")

    con.close()


if __name__ == "__main__":
    main()
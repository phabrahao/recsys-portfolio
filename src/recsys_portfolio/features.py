"""
Reusable point-in-time feature functions, computed as DuckDB window functions
over the `events` table.

Every function here takes a DuckDB relation and returns a new relation with
additional columns -- they're meant to be chained, and they're meant to be
imported by node 1's pipeline script, node 1's tests, and (later) node 3's
training-data construction and node 5's live feature recomputation. Nothing
here should be copy-pasted into a script or a test; import it instead, so a
fix to the window frame here is a fix everywhere.

Leakage-safety contract every function in this file must uphold:
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
i.e. a feature at row R may use rows strictly before R (by event_ts), never
R itself and never anything after it. See 01_feature_pipeline/tests/
test_leakage.py for the regression tests that enforce this.
"""
import duckdb

DEFAULT_COLD_START_THRESHOLD = 10


def add_rolling_rating_count(
    con: duckdb.DuckDBPyConnection, source_table: str = "events"
) -> duckdb.DuckDBPyRelation:
    """
    Point-in-time count of prior ratings per movie, as of each event.

    This is both a feature in its own right (rating velocity / popularity
    momentum) and the basis for add_cold_start_flag below.
    """
    return con.sql(f"""
        SELECT
            user_id, movie_id, event_ts, rating, event_type, source,
            count(*) OVER (
                PARTITION BY movie_id ORDER BY event_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_rating_count
        FROM {source_table}
    """)


def add_cold_start_flag(
    rel: duckdb.DuckDBPyRelation, threshold: int = DEFAULT_COLD_START_THRESHOLD
) -> duckdb.DuckDBPyRelation:
    """
    Flags whether an event occurred while its movie was still cold-start,
    i.e. had fewer than `threshold` prior ratings at that point in time.

    Must be called on the output of add_rolling_rating_count (or any
    relation that already has a prior_rating_count column) -- this function
    does not compute the count itself, it only thresholds it. Threshold
    default matches the value used in the node-1 EDA (README.md); pass a
    different value here when node 3 wants to sweep it.
    """
    return rel.query(
        "feat_input",
        f"""
        SELECT *,
            prior_rating_count < {threshold} AS is_cold_start_at_this_point
        FROM feat_input
        """,
    )


def build_feature_table(
    con: duckdb.DuckDBPyConnection,
    cold_start_threshold: int = DEFAULT_COLD_START_THRESHOLD,
    source_table: str = "events",
) -> duckdb.DuckDBPyRelation:
    """
    Convenience composition of the two functions above -- the full node-1
    feature set as it stands today. Add new add_<feature>() functions above
    and chain them in here as the pipeline grows (session recency, genre
    content vectors, etc. -- see 01_feature_pipeline/README.md step 5).
    """
    rel = add_rolling_rating_count(con, source_table=source_table)
    rel = add_cold_start_flag(rel, threshold=cold_start_threshold)
    return rel

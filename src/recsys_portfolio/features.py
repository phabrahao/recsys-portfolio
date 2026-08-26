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
DEFAULT_GENRE_FATIGUE_WINDOW_DAYS = 30


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


def add_genre_fatigue_score(
    rel: duckdb.DuckDBPyRelation,
    movies_csv_path: str,
    window_days: int = DEFAULT_GENRE_FATIGUE_WINDOW_DAYS,
) -> duckdb.DuckDBPyRelation:
    """
    Fatigue precursor feature: for each event, how much has this user recently
    been exposed to the current movie's genre(s), as of that point in time.

    Design decisions (see 01_feature_pipeline/README.md for the full writeup):
      - Time-based window (last `window_days` days), not count-based (last N
        ratings) -- fatigue is about recency in real time, not interaction
        count, so a user who rates 5 movies in one sitting shouldn't look
        identical to one who spreads them across a year.
      - Multi-genre movies (e.g. "Action|Sci-Fi") are handled by averaging
        the prior-exposure count across all of the movie's genres, not
        taking the max (would overstate fatigue from the single most-repeated
        genre) or using only the first-listed genre (throws away signal from
        the rest). Average is the middle ground.
      - COUNTS DISTINCT PRIOR DAYS with genre activity, not raw prior events.
        This was NOT the first design: an earlier raw-event-count version was
        found (via a p95/max sanity check against the real 32M-row data) to
        be dominated by MovieLens bulk-import bursts -- some users have
        thousands of ratings landing on a single calendar day (max observed:
        6,456 ratings in one day for one user), which is an import artifact,
        not genuine viewing/rating fatigue. Counting distinct days instead of
        raw events means a 50-rating burst day and a 1-rating day both
        contribute exactly "1 day of exposure" to the window, which is robust
        to import volume while still capturing real recency.
      - Uses a RANGE window frame (not ROWS, unlike the other features in
        this module) because the window boundary is defined by elapsed time,
        not a fixed number of preceding rows. EXCLUDE CURRENT ROW keeps the
        same leakage-safety contract as the rest of this file: a row's score
        never includes itself.

    rel must already contain user_id, movie_id, event_ts columns (i.e. this
    is meant to be chained onto add_rolling_rating_count's output, or called
    directly on a bare events relation).
    """
    return rel.query(
        "rel_input",
        f"""
        WITH movie_genres AS (
            SELECT movieId AS movie_id, UNNEST(string_split(genres, '|')) AS genre
            FROM read_csv_auto('{movies_csv_path}')
            WHERE genres != '(no genres listed)'
        ),
        exploded AS (
            SELECT r.user_id, r.movie_id, r.event_ts, mg.genre,
                   r.event_ts // 86400 AS day_bucket
            FROM rel_input r
            JOIN movie_genres mg USING (movie_id)
        ),
        exploded_days AS (
            -- one row per (user, genre, calendar day) regardless of how many
            -- ratings happened that day -- this is what neutralizes bursts
            SELECT DISTINCT user_id, genre, day_bucket
            FROM exploded
        ),
        day_exposure AS (
            SELECT user_id, genre, day_bucket,
                count(*) OVER (
                    PARTITION BY user_id, genre ORDER BY day_bucket
                    RANGE BETWEEN {window_days} PRECEDING AND CURRENT ROW
                    EXCLUDE CURRENT ROW
                ) AS genre_prior_day_count
            FROM exploded_days
        ),
        exploded_with_count AS (
            SELECT e.user_id, e.movie_id, e.event_ts, e.genre, d.genre_prior_day_count
            FROM exploded e
            JOIN day_exposure d USING (user_id, genre, day_bucket)
        ),
        genre_agg AS (
            SELECT user_id, movie_id, event_ts, avg(genre_prior_day_count) AS genre_fatigue_score
            FROM exploded_with_count
            GROUP BY user_id, movie_id, event_ts
        )
        SELECT rel_input.*, COALESCE(genre_agg.genre_fatigue_score, 0.0) AS genre_fatigue_score
        FROM rel_input
        LEFT JOIN genre_agg USING (user_id, movie_id, event_ts)
        """,
    )


def build_feature_table(
    con: duckdb.DuckDBPyConnection,
    cold_start_threshold: int = DEFAULT_COLD_START_THRESHOLD,
    genre_fatigue_window_days: int = DEFAULT_GENRE_FATIGUE_WINDOW_DAYS,
    movies_csv_path: str = "data/raw/ml-32m/movies.csv",
    source_table: str = "events",
) -> duckdb.DuckDBPyRelation:
    """
    Convenience composition of the functions above -- the full node-1
    feature set as it stands today. Add new add_<feature>() functions above
    and chain them in here as the pipeline grows (session recency next --
    see 01_feature_pipeline/README.md step 5).
    """
    rel = add_rolling_rating_count(con, source_table=source_table)
    rel = add_cold_start_flag(rel, threshold=cold_start_threshold)
    rel = add_genre_fatigue_score(
        rel, movies_csv_path=movies_csv_path, window_days=genre_fatigue_window_days
    )
    return rel
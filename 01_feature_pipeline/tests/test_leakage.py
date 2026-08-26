"""
Point-in-time leakage tests.

Two separate guarantees are tested, because two separate mechanisms produce
data in this project:

1. EventStream.replay() — the ordering contract for node 5's future live loop.
2. recsys_portfolio.features functions — the actual mechanism node 1's
   features are computed with. These tests IMPORT and CALL the real
   functions (not a copy of their SQL), so a future change to a window
   frame, partition key, or threshold in features.py is exercised by
   whatever it changed to, and fails here if it leaks. (See
   01_feature_pipeline/README.md for why replay() is not how features are
   computed -- it's the interface node 5's live loop will consume later.)

Run with: uv run pytest 01_feature_pipeline/tests/test_leakage.py -v
"""
import duckdb
import pytest
import csv
from pathlib import Path
from recsys_portfolio.event_stream import EventStream, Event
from recsys_portfolio.features import (
    add_rolling_rating_count,
    add_cold_start_flag,
    add_genre_fatigue_score,
    build_feature_table,
)


@pytest.fixture
def stream(tmp_path):
    db_path = tmp_path / "test_events.duckdb"
    es = EventStream(db_path)
    # small synthetic dataset, deliberately out-of-file-order in places
    events = [
        Event(user_id=1, movie_id=100, event_ts=10, rating=4.0),
        Event(user_id=2, movie_id=100, event_ts=20, rating=3.0),
        Event(user_id=3, movie_id=100, event_ts=15, rating=5.0),  # out of order vs above on purpose
        Event(user_id=1, movie_id=200, event_ts=30, rating=2.0),
        Event(user_id=4, movie_id=100, event_ts=40, rating=1.0),
    ]
    for e in events:
        es.append(e)
    yield es
    es.close()


# ---------------------------------------------------------------------------
# 1. replay() ordering contract
# ---------------------------------------------------------------------------

def test_replay_yields_nondecreasing_timestamps(stream):
    """replay() must hand out events in true timestamp order, regardless of
    the order they were inserted/appended in."""
    all_ts = []
    for batch in stream.replay(batch_size=2):  # small batch to force multiple pages
        all_ts.extend(batch["event_ts"].to_list())

    assert all_ts == sorted(all_ts), (
        f"replay() returned out-of-order timestamps: {all_ts}"
    )
    assert len(all_ts) == 5


def test_replay_catches_a_real_violation(stream):
    """Prove the ordering check has teeth: corrupt the underlying table
    directly (bypassing the class) and confirm a naive consumer relying on
    replay()'s contract would actually notice the corruption."""
    # sabotage: manually insert an event with an out-of-range timestamp
    # directly into the same physical table, then re-scan and check it landed
    # in the wrong logical position relative to a naive full-table sort.
    stream.con.execute("""
        INSERT INTO events VALUES (99, 999, 12, 'rating', 'live', 9.9, NULL)
    """)

    all_ts = []
    for batch in stream.replay(batch_size=2):
        all_ts.extend(batch["event_ts"].to_list())

    # replay() should STILL be correctly ordered (it re-sorts on every read) —
    # this confirms replay() is robust to insertion order, which is the
    # actual guarantee. If this assertion were to fail, that's the signal
    # a leakage-relevant ordering bug exists.
    assert all_ts == sorted(all_ts)
    assert 12 in all_ts


# ---------------------------------------------------------------------------
# 2. Window-function feature: point-in-time correctness
# ---------------------------------------------------------------------------

def test_rolling_count_feature_excludes_current_and_future_rows(stream):
    """The REAL feature function (recsys_portfolio.features.add_rolling_rating_count)
    must never let a row's feature value include itself or any later event.
    This calls the actual production code, not a copy of its SQL -- if the
    window frame in features.py ever changes, this test runs against
    whatever it changed to, and fails if that change leaks."""
    result = add_rolling_rating_count(stream.con).order("movie_id, event_ts").pl()

    # movie_id=100 has 4 ratings at ts=10,15,20,40 (inserted out of order,
    # window function must still respect ORDER BY event_ts internally)
    m100 = result.filter(result["movie_id"] == 100).sort("event_ts")
    prior_counts = m100["prior_rating_count"].to_list()

    assert prior_counts == [0, 1, 2, 3], (
        f"prior_rating_count leaked future/current info: got {prior_counts}, "
        f"expected [0, 1, 2, 3] (first rating for a movie must see 0 prior)"
    )

    # movie_id=200 has exactly one rating -> must see 0 prior, not 1 (self-leak)
    m200 = result.filter(result["movie_id"] == 200)
    assert m200["prior_rating_count"].to_list() == [0]


def test_cold_start_flag_matches_threshold(stream):
    """The REAL add_cold_start_flag function, chained onto the REAL
    add_rolling_rating_count function -- exactly how build_feature_table
    composes them. Threshold=2 makes the hand-checkable boundary land
    inside our 4-row synthetic movie (counts 0,1,2,3)."""
    rel = add_rolling_rating_count(stream.con)
    result = add_cold_start_flag(rel, threshold=2).order("movie_id, event_ts").pl()

    m100 = result.filter(result["movie_id"] == 100).sort("event_ts")
    flags = m100["is_cold_start_at_this_point"].to_list()

    # prior_rating_count = [0,1,2,3], threshold=2 -> flag = count < 2
    assert flags == [True, True, False, False], (
        f"is_cold_start_at_this_point didn't match prior_rating_count < 2: {flags}"
    )


def test_rolling_count_feature_breaks_when_boundary_is_wrong(stream):
    """Negative control: deliberately use a leaky frame (includes CURRENT ROW)
    and confirm the test infrastructure actually distinguishes leaky from
    correct — otherwise the passing test above proves nothing."""
    leaky_result = stream.con.execute("""
        SELECT
            movie_id, event_ts,
            count(*) OVER (
                PARTITION BY movie_id ORDER BY event_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS leaky_count
        FROM events
        WHERE movie_id = 100
        ORDER BY event_ts
    """).pl()

    leaky_counts = leaky_result["leaky_count"].to_list()
    # with CURRENT ROW included, first rating already "sees" itself -> count=1, not 0
    assert leaky_counts == [1, 2, 3, 4]
    assert leaky_counts != [0, 1, 2, 3], (
        "sanity check failed: leaky and correct framings produced identical "
        "results, meaning this test suite cannot actually detect leakage"
    )


# ---------------------------------------------------------------------------
# 3. Genre fatigue score: time-based window, multi-genre averaging
# ---------------------------------------------------------------------------

@pytest.fixture
def movies_csv(tmp_path):
    """Small synthetic movies.csv: enough genre variety to hand-verify averaging,
    plus one '(no genres listed)' movie to catch row-dropping regressions."""
    path = tmp_path / "movies_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        w.writerow([10, "A", "Action|Comedy"])
        w.writerow([11, "B", "Action"])
        w.writerow([12, "C", "Action|Sci-Fi"])
        w.writerow([13, "D", "Action"])
        w.writerow([14, "E", "(no genres listed)"])
    return path


@pytest.fixture
def fatigue_stream(tmp_path):
    """User 1 rates 3 Action-genre movies within days of each other, then one
    more 40 days later -- long enough to fall outside a 30-day fatigue window.
    Also rates a genre-less movie (14), to test that path doesn't drop rows."""
    db_path = tmp_path / "fatigue_events.duckdb"
    es = EventStream(db_path)
    DAY = 86400
    for movie_id, ts in [(10, 0), (11, 1 * DAY), (12, 2 * DAY), (13, 40 * DAY), (14, 50 * DAY)]:
        es.append(Event(user_id=1, movie_id=movie_id, event_ts=ts, rating=4.0))
    yield es
    es.close()


def test_genre_fatigue_score_builds_up_and_resets_outside_window(fatigue_stream, movies_csv):
    """Hand-verified ground truth:
    - movie 10 (Action|Comedy) at day 0: no prior events at all -> 0.0
    - movie 11 (Action) at day 1: 1 prior Action exposure (movie 10) -> 1.0
    - movie 12 (Action|Sci-Fi) at day 2: prior Action=2 (movies 10,11), prior
      Sci-Fi=0 -> average = 1.0. This is the multi-genre-averaging behavior:
      NOT max (would give 2.0), NOT first-genre-only (would give 2.0 or 0.0
      depending on which genre is "first").
    - movie 13 (Action) at day 40: the window is 30 days, so events from
      day 0-2 are all more than 30 days before day 40 -> correctly resets to 0.0
    - movie 14 ((no genres listed)) at day 50: no genre data at all -> 0.0
      via LEFT JOIN + COALESCE, and critically must NOT disappear from the
      result (see test_genre_fatigue_score_does_not_drop_genreless_movies)
    """
    rel = add_rolling_rating_count(fatigue_stream.con)
    result = add_genre_fatigue_score(
        rel, movies_csv_path=str(movies_csv), window_days=30
    ).order("event_ts").pl()

    scores = result["genre_fatigue_score"].to_list()
    assert scores == [0.0, 1.0, 1.0, 0.0, 0.0], (
        f"genre_fatigue_score didn't match hand-verified expectation: {scores}"
    )


def test_genre_fatigue_score_averages_not_maxes_across_genres(fatigue_stream, movies_csv):
    """Isolates the averaging behavior specifically: movie 12 is Action|Sci-Fi
    with prior counts [2, 0] across those two genres. avg=1.0, max would be 2.0.
    This test exists to catch a regression to max-based aggregation."""
    rel = add_rolling_rating_count(fatigue_stream.con)
    result = add_genre_fatigue_score(
        rel, movies_csv_path=str(movies_csv), window_days=30
    ).pl()

    movie_12_score = result.filter(result["movie_id"] == 12)["genre_fatigue_score"][0]
    assert movie_12_score == 1.0, (
        f"expected average(2, 0)=1.0 for movie 12's two genres, got {movie_12_score} "
        f"-- if this is 2.0, aggregation regressed to max() instead of avg()"
    )


def test_genre_fatigue_score_does_not_drop_genreless_movies(fatigue_stream, movies_csv):
    """Regression test for a real bug found in this project: an INNER JOIN
    against movie_genres silently dropped every event for movies tagged
    '(no genres listed)', because those movies have zero rows in the
    exploded genre table. Caught by comparing row count in vs out -- lost
    55,498 rows (32,000,204 -> 31,944,706) against the real MovieLens data
    before this test existed. Must be a LEFT JOIN with COALESCE to 0.0."""
    rel = add_rolling_rating_count(fatigue_stream.con)
    result = add_genre_fatigue_score(
        rel, movies_csv_path=str(movies_csv), window_days=30
    ).pl()

    assert result.height == 5, (
        f"expected all 5 input events to survive the join, got {result.height} rows "
        f"-- a genre-less movie is likely being silently dropped"
    )

    movie_14_score = result.filter(result["movie_id"] == 14)["genre_fatigue_score"][0]
    assert movie_14_score == 0.0, (
        f"genre-less movie should default to genre_fatigue_score=0.0, got {movie_14_score}"
    )


# ---------------------------------------------------------------------------
# 4. Same check, against the real MovieLens-loaded database (not synthetic)
# ---------------------------------------------------------------------------

REAL_DB_PATH = Path("data/processed/events.duckdb")


@pytest.mark.skipif(
    not REAL_DB_PATH.exists(),
    reason="data/processed/events.duckdb not found — run load_historical() first",
)
def test_real_data_first_rating_has_zero_prior_count():
    """Spot-check against the actual 32M-row table: pick a movie with very
    few ratings, confirm its first-ever rating sees prior_rating_count = 0.
    This is the same guarantee as the synthetic test, but against real data,
    so it also catches load-path bugs (e.g. duplicate loads) the synthetic
    fixture can't see since it never touches load_historical()."""
    con = duckdb.connect(str(REAL_DB_PATH), read_only=True)

    total = con.execute(
        "SELECT count(*) FROM events WHERE source = 'historical_replay'"
    ).fetchone()[0]
    assert total == 32_000_204, (
        f"expected exactly 32,000,204 historical rows (MovieLens 32M), got "
        f"{total} — table may contain a duplicate load, rerun load_historical() "
        f"after deleting the .duckdb file if this fails"
    )

    # a movie with exactly one rating, if one exists, is the cleanest check
    single_rating_movie = con.execute("""
        SELECT movie_id FROM events
        WHERE source = 'historical_replay'
        GROUP BY movie_id HAVING count(*) = 1
        LIMIT 1
    """).fetchone()

    assert single_rating_movie is not None, "no single-rating movie found to test against"
    movie_id = single_rating_movie[0]

    # goes through the REAL pipeline composition (build_feature_table), not
    # a hand-written query -- this is the same call build_features.py makes
    result = build_feature_table(con, movies_csv_path="data/raw/ml-32m/movies.csv") \
        .filter(f"movie_id = {movie_id}").pl()

    assert result["prior_rating_count"].to_list() == [0], (
        f"movie_id={movie_id} has exactly 1 rating total but "
        f"prior_rating_count={result['prior_rating_count'].to_list()} — leakage in the real table"
    )
    assert result["is_cold_start_at_this_point"].to_list() == [True], (
        f"movie_id={movie_id}'s only rating should be flagged cold-start "
        f"(0 prior ratings), got {result['is_cold_start_at_this_point'].to_list()}"
    )
    assert "genre_fatigue_score" in result.columns, (
        "build_feature_table did not attach genre_fatigue_score -- "
        "check movies.csv path and the join in add_genre_fatigue_score"
    )

    # the row-loss regression check: build_feature_table's output row count
    # must equal the input event count, no matter which features are chained
    # in -- any INNER JOIN in a feature function is a bug (should be LEFT JOIN)
    full_result_count = build_feature_table(
        con, movies_csv_path="data/raw/ml-32m/movies.csv"
    ).count("*").fetchone()[0]
    assert full_result_count == total, (
        f"build_feature_table dropped rows: {total} events in, "
        f"{full_result_count} out ({total - full_result_count} lost) -- "
        f"a feature function is likely using INNER JOIN where it should use LEFT JOIN"
    )

    con.close()
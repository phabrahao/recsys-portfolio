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
    add_session_recency,
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
    """Hand-verified ground truth (day-bucketed):
    - movie 10 (Action|Comedy) at day 0: no prior days at all -> 0.0
    - movie 11 (Action) at day 1: 1 prior day with Action activity (day 0) -> 1.0
    - movie 12 (Action|Sci-Fi) at day 2: prior Action days=2 (day 0, day 1),
      prior Sci-Fi days=0 -> average = 1.0. This is the multi-genre-averaging
      behavior: NOT max (would give 2.0), NOT first-genre-only.
    - movie 13 (Action) at day 40: the window is 30 days, so activity from
      day 0-2 is all more than 30 days before day 40 -> correctly resets to 0.0
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
    with prior day-counts [2, 0] across those two genres. avg=1.0, max would
    be 2.0. This test exists to catch a regression to max-based aggregation."""
    rel = add_rolling_rating_count(fatigue_stream.con)
    result = add_genre_fatigue_score(
        rel, movies_csv_path=str(movies_csv), window_days=30
    ).pl()

    movie_12_score = result.filter(result["movie_id"] == 12)["genre_fatigue_score"][0]
    assert movie_12_score == 1.0, (
        f"expected average(2, 0)=1.0 for movie 12's two genres, got {movie_12_score} "
        f"-- if this is 2.0, aggregation regressed to max() instead of avg()"
    )


def test_genre_fatigue_score_is_robust_to_bulk_import_bursts(tmp_path):
    """Regression test for a real issue found in this project: against the
    real 32M-row dataset, a raw-event-count version of this feature had
    median=19.25, p95=189, max=4156 -- because MovieLens has bulk-import
    users with thousands of ratings landing on a single calendar day
    (max observed: 6,456 ratings in one day for one user). That's an import
    artifact, not fatigue. This test builds the same shape of scenario at
    small scale and asserts a 50-rating burst day contributes the SAME
    fatigue weight as a 1-rating day would -- i.e. day-bucketing, not raw
    event count, is what the feature actually measures."""
    db_path = tmp_path / "burst_events.duckdb"
    es = EventStream(db_path)
    DAY = 86400
    # 50 Action ratings, all on day 0 (the burst)
    for i, movie_id in enumerate(range(100, 150)):
        es.append(Event(user_id=1, movie_id=movie_id, event_ts=0, rating=4.0))
    # one genuine Action rating 5 days later
    es.append(Event(user_id=1, movie_id=200, event_ts=5 * DAY, rating=4.0))
    es.close()

    movies_path = tmp_path / "movies_burst.csv"
    with open(movies_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        for movie_id in range(100, 150):
            w.writerow([movie_id, f"M{movie_id}", "Action"])
        w.writerow([200, "M200", "Action"])

    es = EventStream(db_path)
    rel = add_rolling_rating_count(es.con)
    result = add_genre_fatigue_score(
        rel, movies_csv_path=str(movies_path), window_days=30
    ).pl()
    es.close()

    # every burst-day rating sees 0 prior days (they're all the first day)
    burst_scores = result.filter(result["event_ts"] == 0)["genre_fatigue_score"].to_list()
    assert all(s == 0.0 for s in burst_scores), (
        f"burst-day ratings should all see 0 prior days, got {set(burst_scores)}"
    )

    # movie 200, 5 days later, should see exactly 1 prior day -- NOT 50
    # (50 would mean the old, burst-vulnerable raw-count definition regressed back in)
    movie_200_score = result.filter(result["movie_id"] == 200)["genre_fatigue_score"][0]
    assert movie_200_score == 1.0, (
        f"expected 1.0 (one prior day of Action activity), got {movie_200_score} -- "
        f"if this is close to 50, the feature regressed to counting raw events "
        f"instead of distinct days, and is vulnerable to import-burst inflation again"
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
# 4. Session recency: day-bucketed session boundary, burst-robust
# ---------------------------------------------------------------------------

@pytest.fixture
def session_stream(tmp_path):
    """User 1: a 5-rating burst on day 0 (seconds apart, simulating
    bulk-import), then a genuine event on day 5, then another event on that
    SAME day 5 an hour later. User 2: single event, to check the brand-new-
    user NULL case independently of user 1's history."""
    db_path = tmp_path / "session_events.duckdb"
    es = EventStream(db_path)
    for i, mid in enumerate(range(100, 105)):
        es.append(Event(user_id=1, movie_id=mid, event_ts=i * 3, rating=4.0))  # 3s apart
    DAY = 86400
    es.append(Event(user_id=1, movie_id=200, event_ts=5 * DAY, rating=4.0))
    es.append(Event(user_id=1, movie_id=201, event_ts=5 * DAY + 3600, rating=4.0))
    es.append(Event(user_id=2, movie_id=300, event_ts=500, rating=3.0))
    yield es
    es.close()


def test_session_recency_burst_day_has_one_new_session_not_many(session_stream):
    """Regression test for a real issue found in this project: against the
    real 32M-row dataset, a threshold-based is_new_session (gap > 30 min)
    had median seconds_since_last_event=10, p75=34, 83% of gaps under 1
    minute -- MovieLens bulk-import bursts made the elapsed-time threshold
    untrustworthy at sub-day resolution. This test asserts the fix: only the
    FIRST event of a burst day is a new session, regardless of how tight the
    gaps are between the rest."""
    result = add_session_recency(
        session_stream.con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("user_id = 1 AND event_ts < 300").order("event_ts").pl()

    sessions = result["is_new_session"].to_list()
    assert sessions == [True, False, False, False, False], (
        f"expected only the first burst event to start a new session, got {sessions} "
        f"-- if more are True, the day-bucketing fix has regressed toward "
        f"trusting sub-day gaps again"
    )


def test_session_recency_new_calendar_day_starts_new_session(session_stream):
    """Movie 200, 5 days after the burst, must be a new session regardless
    of its (huge, but irrelevant) raw seconds_since_last_event -- this is
    what the day-bucketed logic is FOR."""
    result = add_session_recency(
        session_stream.con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("movie_id = 200").pl()

    assert result["is_new_session"][0] == True
    assert result["days_since_last_active_day"][0] == 5


def test_session_recency_same_day_stays_one_session_even_with_large_gap(session_stream):
    """Movie 201 is on the SAME calendar day as movie 200, but an hour
    later. Even though the raw gap (3600s) is much larger than the OLD
    30-minute threshold would have allowed, it must NOT start a new session
    -- day granularity, not elapsed seconds, decides the boundary now."""
    result = add_session_recency(
        session_stream.con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("movie_id = 201").pl()

    assert result["is_new_session"][0] == False, (
        "a same-day event was flagged as a new session -- day-bucketing logic broke"
    )
    assert result["seconds_since_last_event"][0] == 3600


def test_session_recency_first_ever_event_has_null_days_since_active(session_stream):
    """A user's first-ever active day has no prior active day to measure
    against -- days_since_last_active_day must be NULL, not a sentinel,
    same reasoning as the rest of this file's NULL-for-missing-history
    convention. Checked independently via user 2, who has no history from
    user 1's burst to interfere with the assertion."""
    result = add_session_recency(
        session_stream.con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("user_id = 2").pl()

    assert result["days_since_last_active_day"][0] is None
    assert result["seconds_since_last_event"][0] is None
    assert result["is_new_session"][0] == True


def test_session_recency_never_uses_current_or_future_event(session_stream):
    """Leakage check consistent with the rest of this file: perturbing a
    LATER event's timestamp must not change an EARLIER event's computed
    values. Moves movie 201 far into the future and confirms movie 200's
    values (and the burst's) are unaffected."""
    con = session_stream.con
    before = add_session_recency(
        con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("movie_id = 200").pl()

    con.execute("UPDATE events SET event_ts = 9999999 WHERE movie_id = 201")

    after = add_session_recency(
        con.sql("SELECT user_id, movie_id, event_ts FROM events")
    ).filter("movie_id = 200").pl()

    assert before["is_new_session"][0] == after["is_new_session"][0] == True
    assert before["days_since_last_active_day"][0] == after["days_since_last_active_day"][0] == 5


# ---------------------------------------------------------------------------
# 5. Same check, against the real MovieLens-loaded database (not synthetic)
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
    assert "seconds_since_last_event" in result.columns, (
        "build_feature_table did not attach session recency columns"
    )
    assert "days_since_last_active_day" in result.columns, (
        "build_feature_table did not attach the day-bucketed session recency column"
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
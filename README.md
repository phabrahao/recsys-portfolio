# recsys-portfolio

A real-time recommendation system, built end to end: retrieval, ranking, low-latency serving, and
a bandit-driven feedback loop — with explicit handling of cold-start, delayed reward, and fatigue.

Built on [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), entirely on free/local
resources (CPU, DuckDB, HF Spaces/Streamlit for hosting).

**Guiding principle:** each node below is independently shippable, and each one documents at least
one deliberate engineering decision — an ablation, a tradeoff, a bug found and fixed — rather than
aiming for breadth. See each node's own README for the full writeup; this page is a map, not a
substitute for reading them.

## Status

| Node | What it does | Status |
|---|---|---|
| [01 — Feature Pipeline](./01_feature_pipeline) | Point-in-time-correct feature engineering over a replayable event stream; cold-start, genre fatigue, session recency | ✅ Complete |
| [02 — Retrieval](./02_retrieval) | Two-tower embedding model + ANN index (FAISS/hnswlib) | 🚧 In progress |
| [03 — Ranking](./03_ranking) | Multi-objective ranking (CTR + watch-time proxy), explicit cold-start handling | ⬜ Not started |
| [04 — Serving](./04_serving) | FastAPI/gRPC wrapping retrieval→ranking, latency profiling | ⬜ Not started |
| [05 — Feedback Loop](./05_feedback_loop) | Bandit exploration (LinUCB/Thompson sampling), delayed reward, fatigue modeling | ⬜ Not started |
| [06 — Capstone](./06_capstone) | All nodes integrated into one live demo + dashboard | ⬜ Not started |

## Highlights so far

- **Point-in-time leakage discipline**: every feature is tested against the real production code
  (not a copy of its logic), with negative-control tests proving each test can actually detect a
  leaky version — see [`01_feature_pipeline/tests/test_leakage.py`](./01_feature_pipeline/tests/test_leakage.py).
- **Two real bugs found via distribution sanity-checks, not test failures**: a genre-fatigue feature
  and a session-recency feature both initially shipped with definitions that ran cleanly and passed
  their first tests, but were dominated by MovieLens bulk-import artifacts. Both are documented with
  before/after distributions in [node 1's README](./01_feature_pipeline/README.md#two-bugs-the-sanity-checks-caught-and-why-they-mattered).

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) · [Polars](https://pola.rs/) · [DuckDB](https://duckdb.org/)
(feature-store substitute) · PyTorch (nodes 2–3) · FastAPI/gRPC (node 4) · Streamlit/HF Spaces (hosting)
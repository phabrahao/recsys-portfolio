# Node 1 — Feature Pipeline & Data Architecture

## Dataset

[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) — 32,000,204 ratings and 2,000,072
tag applications across 87,585 movies, created by 200,948 users between 1995-01-09 and 2023-10-13.

Two constraints baked into the dataset design that shaped the decisions below:
- **Users were selected at random, but only from users who had rated at least 20 movies.**
  User-side cold-start (a brand-new user with little/no history) is structurally absent from this
  data — confirmed empirically, min ratings/user = 20. Any cold-start handling this pipeline builds
  is necessarily **item-side only**; a synthetic new-user path would need to be manufactured
  separately (see node 5) if user-side cold-start needs to be exercised.
- **No demographic information is included.** User representations are behavioral/session-based by
  necessity, not by design preference.

## Schema

| File | Columns |
|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp (unix seconds) |
| `movies.csv` | movieId, title (release year embedded in parens), genres (pipe-separated) |
| `links.csv` | movieId, imdbId, tmdbId |
| `tags.csv` | userId, movieId, tag, timestamp |

`release_year` is extracted from `title` via regex (`\((\d{4})\)\s*$`) — 617 of 87,585 titles
(0.7%) didn't match the pattern and fall back to null; low enough to trust the field for a
`movie_age` feature without a cleanup pass.

Full exploration: [`eda.ipynb`](./eda.ipynb) · charts: [`charts/`](./charts)

## Cold-start: two numbers that tell different stories

**Static, all-time (naive) view:** 3,153 of 87,585 movies (3.6%) have zero ratings, ever, as of
this dataset's 2023 export. This is what a simple groupby on `ratings.csv` alone would miss
entirely — it requires an anti-join against the full `movies.csv` catalog to surface.

**Item-population long tail:** of the movies that *do* have at least one rating, 62.1% have fewer
than 10 ratings total. By item count, cold-start isn't an edge case here — it's the majority of the
catalog.

**Time-aware, leakage-safe view (the one that matters for training/serving):** only 1.5% of
*rating events* — not distinct movies, actual traffic — occur while the movie in question has
fewer than 10 *prior* ratings at that point in time. Computed via a cumulative count per movie,
ordered by timestamp, so it never looks forward from any given event.

**Why both numbers matter together:** the gap between "62% of movies are sparse" and "1.5% of
traffic touches sparse movies" is the actual shape of the cold-start problem — it's a long tail
that's wide (most of the catalog) but thin (little of the volume). A ranking model optimized purely
for aggregate accuracy could reasonably ignore this population and barely move its overall metric,
while still failing completely on new/niche content — which is precisely the case product teams
care about (surfacing new items, not just re-serving what's already popular). This motivates node
3's design: cold-start handling needs its own evaluation slice, not just a folded-in average.

The static 3.6% figure is a ceiling artifact of using an all-time snapshot; the pipeline itself
uses the time-aware `is_cold_start_at_this_point` flag (threshold: <10 prior ratings, tunable),
computed identically to how node 3 will consume it at serving time.

## Data quality check

1,014 ratings (0.003% of all ratings) have a timestamp before their movie's parsed release year.
At this rate it's noise (re-releases, festival screenings, or a handful of title-parsing edge
cases) rather than a systemic extraction problem — `release_year` is trustworthy as a feature input
as-is.

## Next steps

- [ ] Event stream generator (replay `ratings.csv` in timestamp order)
- [ ] First DuckDB windowed feature table
- [ ] `is_cold_start_at_this_point` as a pipeline feature column
- [ ] Point-in-time leakage test
- [ ] Additional features (session recency, genre content vectors)

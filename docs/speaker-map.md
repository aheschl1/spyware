# Speaker map — PCA over voice-prints

`GET /v1/speakers/projection` + the `map` tab. A scatter plot of the corpus
the batch clusterer actually clusters, so its output can be looked at instead
of inferred from distance numbers.

## What a point is

One row of `speaker_embeddings`: a **(block, final diarization label)**
voice-print, 256-d pyannote/wespeaker.

Not per turn and not per session. The diarizer returns a vector per turn, but
`DiarizePipeline._embedding_rows` pools them — the `clean_ms`-weighted mean of
the L2-normalized turn vectors, re-normalized — and only that survives. A block
is contiguous speech, gap-joined at `diarize_block_merge_gap_ms` (30s) and
capped at 30min, so labels are namespaced `b<block_start>:SPEAKER_00`.

Plotting the stored rows is deliberate: they are exactly `cluster_corpus`'s
input, so the map debugs the real clustering rather than a parallel view of it.

## Rules the endpoint keeps

**One model per basis.** Vectors from different embedding models occupy
different spaces; a shared basis would be meaningless. The filter is in SQL
(`WHERE e.model = %s`), and the response names the model it fitted.

**Fit globally, filter locally.** The basis is always fitted on the whole
`(user, model)` corpus; `session_id` / `include_unassigned` / `limit` only
subset the output. Otherwise every remaining point would jump whenever a filter
changed. `fit_points` and `returned` report both sides.

**The response is a pure function of its input.** PCA component signs are
arbitrary, and a plot that mirrors itself between polls is unusable. Signs are
pinned by `sign(Σ vⱼ|vⱼ|)` — continuous in the loadings, so a single new row
cannot flip an axis. sklearn's `svd_flip` (sign of the largest-magnitude entry)
is discontinuous and was rejected for that reason. No convention can stop a
component from genuinely rotating, so `basis_id` (a hash of mean + components)
reports when the basis really moved; the viewer holds its pan/zoom until it
changes.

**Cluster markers are the mean of member coordinates.** Projection is affine,
so that *is* the projected centroid — exactly, with no extra query. Projecting
`speakers.centroid` instead would re-normalize it and drift, and would place a
ghost marker for a named-but-empty cluster that is only holding a stale
centroid as an identity anchor.

**Truncation is a stride, not a head slice**, so a capped response still spans
the whole corpus instead of only its oldest sessions.

`distance` on a point is the cosine distance to its centroid in the **full
256-d space** — the same number `SpeakersView` shows. It is not the on-screen
distance, which is a shadow. Two components out of 256 explain ~15% of the
variance on the current corpus, so the toolbar pill carries that figure and the
inspector gives the real one.

## Cost, measured

| n voice-prints | PCA (eigh, →3D) |
|---:|---:|
| 404 (today) | 4 ms |
| 10,000 | 6 ms |
| 100,000 | 86 ms |
| 1,000,000 | 1.2 s |

`eigh` on the 256×256 covariance beats SVD of the n×256 matrix at every n ≥ d
and the gap widens (4.0 ms vs 14.7 ms at n=404; 19 ms vs 605 ms at n=50k), so
there is no crossover to branch on. Live request today: 37 ms read + 9 ms fit.

**The projection is not what will break — and the clusterer no longer will
either.** `cluster_corpus` (`processing/clustering.py`) is now a
nearest-neighbour chain: average linkage over cosine is reducible, and a
cluster is fully described by the sum of its members' unit vectors
(`d(A,B) = 1 − (S_A·S_B)/(|A|·|B|)`), so there is no distance matrix at all —
O(n²) time, O(n) memory, identical dendrogram. Cannot-link pins survive as
tag checks in the distance function. Measured on merging data (40 centers,
256-d):

| n | old O(n³) | NN-chain |
|---:|---:|---:|
| 404 | 0.02 s | 0.04 s |
| 3,200 | 8.97 s | 1.7 s |
| 6,400 | 74.65 s | 6.8 s |
| 12,800 | — | 37 s |

Growth is now ~4× per doubling. At ~135–200 rows/day the next wall is years
out, and `tests/unit/test_speaker_cluster.py` carries a parity test against
the old implementation as an oracle.

> Benchmarking note: measure this function on data that actually *merges*.
> Random high-dimensional vectors are near-orthogonal, nothing passes the
> threshold, the loop exits immediately, and n=6,400 looks like 0.59 s.

## When to add machinery

Nothing below is needed today; the triggers are latency, not dates.

- **~5,000 points** — replace the stride with density subsampling (keep pinned
  points, cluster extremes, farthest-from-centroid members), and move the
  LATERAL clip join out of the bulk read into a per-point detail route.
- **~20,000** — cache the fitted basis behind a `(count, max(created_at))`
  fingerprint, the `api/session_audio.py` `_RepresentationCache` pattern. It
  skips the parse and fit, not the row read.
- **~100,000** — precompute coordinates in a worker tier off the `diarize-map`
  trigger.
- **Never, for this feature** — an ivfflat/hnsw index. PCA and `cluster_corpus`
  are full-corpus scans; ANN accelerates k-NN, and the only k-NN here
  (`SpeakersRepo.similar`) scans 74 rows.

## Views

Three components are always computed (`eigh` returns them regardless), shown as
three orthogonal 2-D pairs — PC1·PC2, PC1·PC3, PC2·PC3 — plus an orbitable 3-D
scene. The explained-variance pill is derived from the components currently on
screen, so switching views changes it; PC2·PC3 is ~8.7% where PC1·PC2 is ~15.3%.
The per-component breakdown is the pill's tooltip.

**The layout is width-dependent, and that is load-bearing.** Below 640px the
legend is dropped and the plot margins collapse: a fixed 176px legend gutter is
nearly half a phone's width. `.map-plot` also carries an explicit `width: 100%` —
in the mobile column layout `flex: 1` sizes height, not width, and Plotly
silently falls back to its **700×450 default** when it measures a zero-width
container at mount, which overflows the viewport and renders the points
off-screen. That failure looks exactly like "the plot didn't render", and it
reproduces only at a phone viewport.

Both modes preserve equal units, since PCA axes are commensurate: 2-D via
`scaleanchor`, 3-D via `aspectmode: "data"`. (`"cube"` would force a cube box and
stretch PC3, overstating separation along it.)

Rendering is not on the scaling ladder: `scattergl` is WebGL and handles 100k+
points, and zoom/pan, box and lasso select, hover and legend isolation all come
with Plotly rather than being hand-rolled.

`src/plotlyBundle.ts` is a custom bundle — `plotly.js/lib/core` plus only
`scattergl` and `scatter3d` — because the prebuilt dists are all-or-nothing: the
gl2d one has no `scatter3d`, and the full dist carries traces nothing renders.
It costs ~671 kB gzipped, dynamically imported so it downloads only when the tab
is opened. Those `lib/*` entry points are CommonJS and reference Node's
`global`, unlike the dists, which is why `vite.config.ts` defines
`global: "globalThis"` — without it the tab fails at runtime while the build
still succeeds.

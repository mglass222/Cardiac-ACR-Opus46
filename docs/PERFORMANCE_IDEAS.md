# Performance Improvement Ideas

Engineering / efficiency ideas for future work. Unlike `WEAKNESSES.md`
(which catalogs study-design and methodology issues that affect paper
defensibility), this file tracks implementation-level improvements:
speed, disk usage, memory, deployment ergonomics, and similar
engineering wins.

Each entry should include:
- **Motivation** — what's inefficient today and why it matters
- **Approach** — concrete sketch of the change
- **Cost / risk** — implementation effort, failure modes, what could
  regress
- **Priority** — rough effort-vs-payoff rank

Move items to a "Shipped" section at the bottom once implemented, with
a date and a pointer to the commit.

---

## 1. Streaming WSI inference (no intermediate patch PNGs)

**Motivation.** The current `wsi.diagnose` pipeline materializes every
intermediate artifact on disk: SVS → PNG → filtered PNG → scored tile
PNGs → 224×224 patch PNGs → (tissue filter deletes the bad ones) →
opened again for classification. Per slide that's 1–7 GB of
small-file IO, most of which is written and re-read seconds later.
On HDD or NFS it's a real bottleneck; on SSD it wastes write cycles
and disk space; at any storage tier it makes per-slide cleanup
necessary and complicates parallelization across slides.

**Approach.** Replace steps 4–6 of `diagnose.run()` with an
OpenSlide-backed `torch.utils.data.Dataset`:

```
PatchDataset.__getitem__(idx):
    region = openslide.read_region(x_idx, y_idx, 224, 224)
    if tissue_fraction(region) < 0.5:
        return SKIP_SENTINEL
    return transform(region)

DataLoader(num_workers=8, worker_init_fn=reopen_slide, pin_memory=True)
  → UNIBackbone.encode()
  → head → softmax → dict[(x, y) → probs]
```

Patches that fail the tissue check never leave the worker process.
The DataLoader hides OpenSlide-read latency behind GPU compute. Read
in larger chunks (e.g., 2048×2048) and slice in memory to amortize
OpenSlide's per-call overhead. OpenSlide handles don't pickle across
workers — each worker needs to `open_slide` itself in
`worker_init_fn`, which is a standard pattern.

**Cost / risk.**
- One-afternoon refactor. Scope: new
  `cardiac_acr/wsi/streaming_dataset.py`; rewrite
  `wsi.diagnose.classify_patches` to consume it; drop the
  preprocessing steps that write intermediate patches.
- Loses the "go inspect the PNG that got classified wrong"
  debugging affordance. Mitigation: add a `--dump-patches` flag that
  reinstates disk writes for error-analysis runs.
- Existing modules (`tiles.py`, `tileset_utils.py`,
  `filter_patches.py`) stay as-is for **training patch extraction**,
  which legitimately needs ImageFolder-style on-disk patches. Only
  the WSI inference loop changes.
- Annotation functions (`annotate_png`, `annotate_svs`) read from
  `Saved_Databases/` pickles, not patch files, so they continue to
  work unchanged.

**Priority.** High payoff, moderate effort. Cuts per-slide disk
usage from GB to effectively zero, removes a cleanup step, and likely
speeds up inference on non-SSD storage by a large factor. Do when
current methodology work (test-set labeling, calibration) is not
blocking.

---

## 2. Cache UNI features for test-slide patches

**Motivation.** Re-running `wsi.diagnose` to sweep `PREDICTION_THRESHOLD`
(or to try a retrained head) currently re-encodes every patch through
the ViT-H backbone — the expensive step. The UNI features themselves
don't change between runs; only the head / threshold does.

**Approach.** Mirror the training-side feature-cache pattern for
inference: write one `.pt` per slide under
`data/Backend/Features_Cache/<slide>.pt` holding `{patch_coords,
features [N, 1536]}`. `classify_patches` checks for the cache first;
if present, it loads and runs just the head.

**Cost / risk.**
- Small — maybe a couple of hours. The encoding / head split is
  already clean in `UNIBackbone.encode` + `head(...)`.
- Storage: ~6 KB per patch × tens of thousands of patches per slide
  = ~100–500 MB per slide in fp32. Fp16 halves that.
- Invalidation: if the backbone version or the transform changes,
  caches must be regenerated. Store a schema-version string alongside
  the features to fail-fast on mismatch.

**Priority.** Medium — only valuable if you expect to re-run
diagnose many times with the same backbone. Pairs well with #1
(streaming) since both touch `classify_patches`.

---

## 4. Pipeline multiple slides through the GPU simultaneously

**Motivation.** `diagnose.run()` currently processes slides serially
with a per-slide overhead (slide open, preprocess, classify,
threshold, write pickle). On a fast GPU, classification finishes
before the next slide's preprocessing is done, leaving the GPU idle.

**Approach.** Decouple with a producer/consumer queue: one thread
pool runs preprocessing → patch stream for slides N, N+1, N+2 in
background; the main thread consumes whatever patches are ready and
encodes them, tagging each with its slide ID; per-slide aggregation
happens as patches drain.

**Cost / risk.**
- Moderate — changes the control flow of `diagnose.run` and requires
  careful per-slide bookkeeping (which patch belongs to which slide
  when probabilities are dict-keyed).
- Marginal gain if per-slide preprocessing is fast relative to
  classification. On a 2070 SUPER it's probably not worth it; on a
  5090 where GPU time per patch drops 10×, preprocessing becomes the
  bottleneck and this matters.

**Priority.** Deferred. Revisit if #1 is shipped and GPU utilization
during `diagnose` is still sub-saturated.

---

## 5. fp8 inference on Blackwell (future hardware)

**Motivation.** If the project ever runs on a 5090 or H100/H200,
Transformer Engine's fp8 path gives another ~1.5–2× over bf16 on the
same hardware. Current setup auto-selects bf16 on Ampere+ and fp16 on
Turing (see `uni/backbone.py:_default_autocast_dtype`), which is right
for today's hardware.

**Approach.** Add an optional Transformer Engine integration for
sm_89+ or sm_100+ GPUs. Has to be gated by both compute capability
and the presence of `transformer_engine` in the environment, since
most users won't have it installed.

**Cost / risk.**
- Non-trivial — TE has its own autocast context and module-wrapping
  convention. UNI2-h would need to be rebuilt in the TE path, or the
  matmuls individually wrapped.
- fp8 has accuracy caveats. Would need a validation comparison
  against bf16 before enabling by default.

**Priority.** Don't-do-it-yet. Revisit only when Blackwell / H-series
hardware is actually available *and* there's a throughput need.

---

## 6. Structured experiment logging

**Motivation.** Today, training / evaluation / threshold-sweep
results print to stdout. Comparing runs (linear vs. MLP, threshold
0.9 vs. 0.99, backend A vs. B) means re-running or scrolling terminal
history. There's no versioned record of what each checkpoint
corresponds to.

**Approach.** Lightweight CSV or JSONL logger: one row per training
run with commit hash, config hash, backend, head type, best val acc,
best val loss, runtime. Drop into `data/Logs/`. Opt-in W&B / MLflow
integration is nice-to-have but not necessary for a small project.

**Cost / risk.** Small. Mostly plumbing in `train.py`, `evaluate.py`,
and `threshold_sweep.py`. No semantic changes.

**Priority.** Medium. Makes the paper's supplementary table easy to
produce. Pair with a `data/Logs/README.md` describing the schema.

---

## 7. Profile the pre-loop preprocessing phase

**Motivation.** `diagnose.run()` spends time on
`slide.multiprocess_training_slides_to_images` (SVS→PNG),
`wsi_filter.multiprocess_apply_filters_to_images` (filter PNG), and
`tiles.multiprocess_filtered_images_to_tiles` (tile scoring) *before*
the per-slide classify loop. These are multiprocess CPU pipelines
using PIL + scikit-image morphology; whether they matter depends on
how their time compares to classify.

On the post-DataLoader run classify hits ~66 img/s at 98% SM util.
If a slide's classify step takes, say, 6 minutes for 25k patches,
and the pre-loop phase takes 2 minutes per slide, the pre-loop is
~25% of total runtime and worth touching. If it's 30 seconds per
slide, it isn't.

**Approach.** Measure before optimizing:

```
python -m cProfile -o diag.prof -m cardiac_acr.wsi.diagnose \
  --backend uni  # with a 1–2 slide subset
python -m pstats diag.prof
    sort cumtime
    stats 30
```

Expected suspects once the profile is in hand:
- Morphology operations on whole-slide filtered PNG (scikit-image).
  Some ops have faster OpenCV equivalents.
- Serial tile-score image writes (saving annotated-tile visualizations
  we never look at — the diagnose path passes `save_top_tiles=False`
  but check that the other save-path isn't firing).
- NumPy array copies in the filter pipeline (`util.mask_rgb` and
  similar) that could be in-place.

**Cost / risk.** Profiling itself is free. Any optimization here
has to compete with #1 (streaming) on scope, since streaming
eliminates most of the filter-PNG and tile-scoring outputs entirely.
Don't invest here until you've decided whether #1 is happening.

**Priority.** Medium-low. Do the profile so the next engineering
decision is informed; hold off on fixes until the numbers say which
phase is the bottleneck.

---

## Shipped

### 2026-04-23 — Fuse tissue filter into the classify DataLoader

Dropped the separate `filter_patches_multiprocess` pass from
`wsi.diagnose.run()`. `_PatchFileDataset.__getitem__` now runs the same
`apply_image_filters` + `tissue_percent` check the old pass used; patches
under 50% tissue return a `(path, None)` sentinel, and a new
`_drop_empty_collate` drops them before the batch reaches the GPU.
Result: every patch is touched exactly once instead of up-to-three
times, and no patch PNGs are deleted mid-run. `filter_patches.py`
itself is unchanged — nothing else imports
`multiprocess_apply_filters_to_images`, but it stays in place in case a
future caller wants the standalone tissue-fraction dict.

### 2026-04-22 — DataLoader-fed classify (4× GPU utilization)

`wsi.diagnose.classify_patches` was CPU-bound on serial PIL decode +
transform, with the GPU idle ~75% of the time (SM util ~25% mean,
bursty 6–40%). Replaced the manual loop with a
`torch.utils.data.DataLoader` fed by a `_PatchFileDataset`:
`num_workers=8`, `pin_memory=True`, `shuffle=False`, tqdm progress
bar. Post-change SM util is flat at 97–98% — the Turing fp16 ceiling
for UNI2-h. See `DEVELOPMENT_LOG.md` 2026-04-22 entry for numbers.

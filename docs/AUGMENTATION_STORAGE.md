# How D4 Augmentation Is Stored

A common question while reading the encode pipeline: are the 8 D4 view
variants of each patch held on disk or in memory? Answer is hybrid —
the **images** are transient, the **encoded features** are persisted.

## The short version

| | Disk | RAM (during train) |
|---|---|---|
| Source patches (canonical) | ~5 KB × 11,710 PNGs (~60 MB) | one batch at a time |
| Augmented patches (8 D4 views) | **never written** | one batch at a time, discarded after encode |
| Encoded D4 features | 576 MB (`data/Features/training_views8.pt`) | 576 MB resident |

## What actually happens

### Source patches (PNGs)

Stored once on disk under
`data/Patches/Training_Sets/Training/<class>/`. Only the canonical
orientation. We never write rotated or flipped copies. Created by
`extract_patches.py` + `create_training_sets.py`; ~60 MB total for
11,710 patches.

### Augmented patches (transient)

During the encode pass (`backends/uni/encode_patches.py`), the
DataLoader builds 8 separate `torchvision.transforms.Compose` pipelines
— one per D4 symmetry. For each view:

1. A worker reads a source PNG into CPU memory.
2. Applies that view's transform (rotation + optional flip + resize +
   normalize).
3. Hands the resulting tensor to UNI2-h on the GPU.
4. The augmented tensor is discarded as soon as the encoded feature
   comes back.

The 8 versions of the image exist in worker-process memory for a
fraction of a second — long enough for one ViT-H forward pass — and
are never written to disk.

### Encoded features (persisted)

After all 8 views finish, the 93,680 (= 11,710 × 8) feature vectors
are concatenated with their (duplicated) labels and saved to
`data/Features/training_views8.pt`. The schema is unchanged from the
single-view era; only the row count grew 8×.

At head-training time, `train.py` calls `FeatureCache.load(...)`,
which `torch.load`s the entire `.pt` file into a single
`FloatTensor[93680, 1536]` and a `LongTensor[93680]`. That's ~580 MB
of RAM, resident for the whole 50-epoch training run.

## Why this design

We want **augmentation diversity** without paying the cost of
re-running the ViT-H forward pass on every head retrain.

Two alternatives we explicitly didn't take:

1. **Store 8 augmented PNGs per patch.**
   - Cost: ~14 GB of duplicate PNGs (224×224×3 ≈ 150 KB/image × 93,680
     images).
   - Benefit: nothing, since the head trains on features, not images.
   - The encode step would still need to run the ViT forward 8× per
     patch.
2. **Apply random augmentation per training batch and re-encode each
   epoch.**
   - Cost: 50 epochs × 11,710 patches × ViT-H forward = ~3 hours per
     training run instead of ~30 seconds.
   - Benefit: stochastic aug instead of fixed 8 views.
   - Tradeoff is real but expensive; we deferred it (only used in
     `finetune.py`, which is parked).

Caching at the **feature level** is the right point in the pipeline:
features are 1536 floats = 6 KB per encoding (vs. 150 KB per image),
storing 8 of them costs 576 MB total, and the head trains in seconds
because no backbone forward is needed.

It's the same tradeoff the original ResNet pipeline made when it
cached 224×224 patch PNGs — except we cache one stage further down,
where the data is much smaller.

## Implementation pointers

- View enumeration: `backends/uni/encode_patches.py:49` — `_D4_VIEWS`
  is the explicit list of 8 (rotation, flip) tuples.
- View-specific transforms: `_build_view_transforms(num_views)` at
  `encode_patches.py:70`.
- Per-view encoding loop: `_encode_split(...)` at
  `encode_patches.py:94`. Note the outer loop over views, inner loop
  over patches — each view is a fresh `ImageFolder` pass.
- Cache load: `backends/uni/features_dataset.py:33` —
  `FeatureCache.load()` reads the `.pt` once.
- Per-class growth check: every class count in the cache is exactly
  8× the source count (verified after the encode pass).

## Related

- `docs/UNI_MULTIVIEW_PLAN.md` — original design rationale, A/B
  numbers, and the 2026-04-24 ship.
- `docs/DEVELOPMENT_LOG.md` 2026-04-24 entries — full timeline,
  including the +0.58 pp lift D4 produced over the 1-view baseline.

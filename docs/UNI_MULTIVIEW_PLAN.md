# Multi-view feature encoding for the UNI head (close the val-acc gap)

## Context

The UNI head plateaus at ~94% validation accuracy; the prior ResNet
end-to-end pipeline reached ~96%. The leading cause: ResNet trained
under `ColorJitter` + `RandomRotation(180)`, while the UNI pipeline
trains the head on **pre-encoded features with zero augmentation** —
each patch is seen exactly once in its canonical orientation.

H&E histopathology is rotation- and reflection-invariant, so the
dihedral group D4 (4 rotations × 2 flips = 8 views per patch) is a
"free" augmentation that doesn't require any biological assumption
about the data. Encoding each training patch under all 8 D4 symmetries
multiplies the feature cache 8× and lets the head see meaningfully
varied versions of the same biology.

This is a minimally invasive change: only the encode step grows. The
training loop, head, loss, and downstream WSI inference are untouched.

**Out of scope** (deliberately, to keep this surgical):

- Eval-time TTA (logit-averaging over D4 at inference) — separate change
  in `wsi/diagnose.py` if v1 succeeds.
- ColorJitter as enumerable views — frozen jitter is a strict downgrade
  vs. per-batch random jitter; defer.
- End-to-end fine-tuning of UNI2-h itself.
- Feature-space augmentation (noise, mixup) in `train.py` — defer until
  D4 alone is shown insufficient.
- The 147-patch Hemorrhage class. Augmentation cannot rescue a class
  with one annotated slide; that's a data-collection problem.

## Approach

Build a list of 8 deterministic D4 transforms. Run the existing
encoder once per view over the same `ImageFolder`. Accumulate features
and labels across views into a single `.pt`. Validation stays at 1
view (canonical) so the val-acc metric remains directly comparable to
the 94% baseline. No schema changes — `FeatureCache.load()` consumes
the larger `.pt` unchanged.

## Files to modify

### `cardiac_acr/backends/uni/config.py`

- Add `NUM_TRAIN_VIEWS = 8` in the "UNI feature encoding" block (after
  line 56). Default 8 (full D4).
- Change `TRAINING_FEATURES_PATH` (line 38) to
  `os.path.join(FEATURE_DIR, f"training_views{NUM_TRAIN_VIEWS}.pt")`.
- `VALIDATION_FEATURES_PATH` stays as `validation.pt` (always N=1).

Suffixing the path lets us A/B against the existing `training.pt`
without re-encoding. Once results are confirmed, the old file can be
deleted in a follow-up.

### `cardiac_acr/backends/uni/encode_patches.py`

- Add a top-level `_apply_d4(img, k_rot, do_flip)` helper using
  `torchvision.transforms.functional.rotate` and `hflip`. Bind via
  `functools.partial` to dodge the lambda-closure-in-loop trap.
- Replace `_build_transform()` (lines 38–45) with
  `_build_view_transforms(num_views: int) -> list[transforms.Compose]`.
  - `num_views == 1` → returns `[base_transform]` (current behavior).
  - `num_views == 8` → returns 8 transforms, each with a fixed D4 op
    inserted between `CenterCrop` and `ToTensor`. View ordering:
    `(k_rot ∈ {0,90,180,270}) × (do_flip ∈ {False, True})`.
- Modify `_encode_split(backbone, split_dir, out_path, num_views)`:
  - Outer loop over views (1..N). Inner loop is the existing batch
    encoder, reused unchanged.
  - Build a fresh `ImageFolder` per view with its view-specific
    transform; reuse the `classes` list from view 0.
  - Append to shared `all_feats` / `all_labels`. Order doesn't matter
    — `train.py:91` shuffles the loader.
  - Update progress print to show `view i/N · patch j/total`. Without
    this, the percentage resets each view and looks broken.
  - Keep the empty-cache branch (lines 60–70) intact — it writes
    `torch.empty(0, EMBED_DIM)` which is correct under any N.
- `main()` (lines 114–131): pass `uni_cfg.NUM_TRAIN_VIEWS` for
  Training, hardcoded `1` for Validation.

### `cardiac_acr/backends/uni/train.py`

- No code changes. `_class_weights` is invariant under uniform
  per-class N× scaling: `weight_i = total/(K*count_i) = (N*total_old)/(K*N*count_old)`.
- Optional: add a one-line print of `len(train_loader.dataset)` so
  it's visible in logs that the cache grew 8×.

### Files explicitly **not** touched

`evaluate.py`, `head.py`, `features_dataset.py`, `wsi/diagnose.py`,
`backbone.py`, anything under `preprocessing/`. The change is
encoder-local.

## Reused components

- `torchvision.transforms.functional.rotate` + `hflip` — built-ins.
  No new aug utilities.
- `UNIBackbone.encode()` (`backends/uni/backbone.py`) — unchanged;
  reused as-is for each view's pass.
- The existing `DataLoader` config (`ENCODE_BATCH_SIZE=32`,
  `ENCODE_NUM_WORKERS=8`) — unchanged.
- `FeatureCache.load()` and `as_tensor_dataset()`
  (`features_dataset.py:33,39`) — schema-compatible with the larger
  cache.

## Verification

**Encode pass:**
```bash
cd ~/Documents/Code/Cardiac-ACR-2026 && source .venv/bin/activate
python -m cardiac_acr.backends.uni.encode_patches
```

Expected output:
- Training: `view 1/8` … `view 8/8`, each ~3 min on 2070 SUPER. Total
  encode ~24 min.
- Final `features` shape: `(93680, 1536)` for training (= 11710 × 8).
  Labels shape: `(93680,)`.
- Validation: single pass, identical to current behavior.
- File on disk: `data/Features/training_views8.pt` ≈ 550 MB;
  `validation.pt` ≈ 16 MB unchanged.

**Sanity check the cache before training:**
```python
from cardiac_acr.backends.uni.features_dataset import FeatureCache
from cardiac_acr.backends.uni import config as cfg
c = FeatureCache.load(cfg.TRAINING_FEATURES_PATH)
assert c.features.shape == (93680, 1536), c.features.shape
print(c.class_counts())  # each class count × 8
```

**Train and compare:**
```bash
python -m cardiac_acr.backends.uni.train
```

Compare best-validation accuracy line vs. the 94% baseline. With D4
augmentation working, expect 95%+. If still ~94%, augmentation isn't
the bottleneck and the next lever (per-batch feature-space noise, or
unfreezing the last UNI block) is on the table.

**Optional: regression check on WSI inference.**
After re-train, run a single-slide diagnose (slide 139) and confirm
slide-level dx is still `2R`. The head probabilities will shift
slightly but the slide-level classification should be stable.

## Decision points (defaults already chosen)

- **N = 8** (full D4). Encode cost is one-time (~24 min) and the
  training cache at 550 MB is trivially in-memory.
- **Suffixed cache** (`training_views8.pt`) instead of overwriting
  `training.pt`. Reversible without re-encoding.
- **Validation stays N=1** so val-acc remains directly comparable to
  the 94% baseline.

If you want N=4 instead (rotations only, ~12 min encode, 275 MB
cache), it's a one-line change in `config.py`:
`NUM_TRAIN_VIEWS = 4`.

## Commit plan

1. **First commit (before any code changes):** copy this plan file to
   `docs/UNI_MULTIVIEW_PLAN.md` so the design rationale is preserved
   in the repo. Plan files in `~/.claude/plans/` are transient; the
   docs/ copy is the durable record.
2. **Second commit:** `feat(uni): D4 multi-view feature encoding for
   training set`. Includes the config + encoder edits.
3. **Third commit (after re-encode + retrain):** one-paragraph note in
   `docs/DEVELOPMENT_LOG.md` recording the val-acc before/after and
   any per-class shifts seen in the evaluate output.

## Follow-ups (revisit if D4 alone doesn't close the gap)

- Eval-time D4 averaging in `wsi/diagnose.py`.
- Stochastic per-batch feature-space augmentation in `train.py`
  (Gaussian noise, dropout on features).
- Unfreezing the last block of UNI2-h and fine-tuning end-to-end.

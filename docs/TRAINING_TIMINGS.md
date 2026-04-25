# Training Timings (current setup, 2070 SUPER)

How long each step of the UNI pipeline takes on the development
hardware (RTX 2070 SUPER, 8 GB, fp16). All numbers are measured, not
estimated.

## Headline numbers

| Step | When it runs | Time | Notes |
|---|---|---|---|
| **`encode_patches.py`** (D4, 8 views) | One-time per backbone or augmentation change | **~20 min** | 77.6 img/s steady. 1207 s training + 35 s validation. Output: `training_views8.pt` (576 MB) + `validation.pt` (16 MB). |
| **`train.py`** (head on cached features) | Every head retrain | **~36 s** for 50 epochs | ~0.7 s/epoch. Full forward + backward + val pass over all 93,680 rows. |
| **`evaluate.py`** | After each retrain | **~3 s** | Just val pass through the trained head. |
| **`sweep_head.py`** (18 configs × 50 epochs) | Hyperparameter exploration | **~10 min** | Same 30-40 s/run as `train.py`, no checkpoint write per run. |
| **`finetune.py`** (LoRA, 15 epochs) | Backbone fine-tune (currently parked) | **~33 min** for 7 epochs (early-stopped) | Each epoch ~5 min — full ViT-H forward + backward through last 4 blocks. |

## Why head training is so fast

The expensive part — running 11,710 source patches × 8 D4 views =
93,680 patches through a 681M-param ViT-H — is amortized into the
encode step. Once `training_views8.pt` exists, every subsequent head
retrain operates on a `FloatTensor[93680, 1536]` already loaded into
RAM. With `TRAIN_BATCH_SIZE = 512`, that's only 184 batches per
epoch, and each batch is a couple of matrix multiplies through a tiny
MLP head (~3M params).

The 50-epoch schedule is mostly inertia — best val acc consistently
lands within the first 10 epochs, so 20 epochs would converge to the
same place. The remaining 30 epochs just confirm the cosine schedule
has decayed cleanly. Not worth shortening since the whole run is half
a minute.

## When the 20-minute encode cost actually bites

You only re-pay the encode if any of these change:

- **The backbone** (UNI version, image resolution, normalize mean/std).
- **The augmentation set** (`NUM_TRAIN_VIEWS`, the D4 list, transform
  ordering).
- **The source patch library** (new training slides, re-extracted
  patches, different train/val split).

Hyperparameter tuning on the head, comparing linear vs MLP, sweeping
LR/WD, switching head architectures — all run against the existing
cache and complete in ~30 seconds each. That's why the 18-config
sweep on 2026-04-24 took ~10 minutes total instead of hours.

## Encoding throughput per stage

From `encode_patches.py` log on 2026-04-24:

```
Training (11,710 patches × 8 views = 93,680 encodings)
  view 1/8 |  11710/93680 encodings |  69.8 img/s |  167.7 s
  view 8/8 |  93680/93680 encodings |  77.6 img/s | 1207.5 s
Validation (2,743 patches × 1 view = 2,743 encodings)
  view 1/1 |   2743/2743 encodings |  78.2 img/s |   35.1 s
```

Steady-state ~77 img/s once `torch.compile` has warmed up; the first
view runs slower because it includes the compile cost.

## What changes if hardware changes

- **5090 (Blackwell)**: ~10× faster fp16 throughput vs 2070 SUPER.
  Encode would drop from ~20 min to ~2 min. Head training is already
  IO-bound at the 30-second scale, would not change much.
- **A100/H100**: 5–8× faster on bf16. Same head-training picture.
- **CPU only**: encode pass becomes infeasible (hours per view × 8
  views). Head training is fine, ~5 minutes for 50 epochs.

The encode step dominates whenever it runs. On any GPU faster than
the 2070 SUPER, it stops being a meaningful bottleneck — the head
training is already short enough that further speedup doesn't change
the developer workflow.

## LoRA fine-tune timing (parked but documented)

For reference if anyone revisits the LoRA path:

- Per epoch: ~5 min on the 2070 SUPER (730 batches at 16 patches each,
  forward through all 24 blocks + backward through last 4 + head, fp16
  with grad checkpointing).
- 7-epoch early-stopped run: 2006 s (~33 min).
- Full 15-epoch run would be: ~75 min.
- Memory: ~5–6 GB at batch 16 with grad checkpointing on. Drops to
  batch 8 + grad accum if `target_blocks` is increased to 5+.

Compare to head-only training: 36 s. The fine-tune path is ~125×
slower per epoch, which is why it's only worth running if there's a
clear hypothesis it'll close the val gap. The 2026-04-24 attempt
showed it doesn't — see `DEVELOPMENT_LOG.md`.

## Related

- `docs/AUGMENTATION_STORAGE.md` — how the 8 D4 view variants are
  stored (hybrid: images transient, features persisted).
- `docs/UNI_MULTIVIEW_PLAN.md` — design rationale for the encode-time
  augmentation.
- `docs/UNI_LORA_PLAN.md` — LoRA fine-tune plan (negative result).

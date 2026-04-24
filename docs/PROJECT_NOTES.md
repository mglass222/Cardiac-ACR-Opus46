# Project Notes

Ongoing methodology and design-justification notes for the Cardiac-ACR
paper. Unlike `DEVELOPMENT_LOG.md` (which records *what was built*),
this file records *why choices would be defensible in a manuscript* —
decisions that need paper-level rationale, not just engineering
rationale.

---

## Defending the choice of `PREDICTION_THRESHOLD`

**Context.** `wsi.diagnose` aggregates patch-level softmax predictions
into a slide-level ISHLT rejection grade. Before aggregation, patches
with top-softmax below `PREDICTION_THRESHOLD` (default 0.99, inherited
from the legacy 2019 ResNet pipeline) are dropped. The value of this
threshold directly affects slide-level grades and therefore headline
numbers.

**The problem.** "We picked the threshold that gave us the best
numbers" is p-hacking / test-set contamination. A reviewer will reject
it. The threshold must be selected by a procedure that does not
consult the test set.

### Three defensible framings (any one suffices)

**1. Pre-specified tuning set (cleanest).**
Split slides three ways: train / tuning / test. Sweep threshold on
*tuning*, optimizing a pre-declared slide-level metric (e.g.,
slide-level grade accuracy, macro-F1, Cohen's κ vs. pathologist).
Fix that threshold. Report test-set performance at that single
threshold. Disclose tuning-set size, sweep range, and objective.

*Current repo status:* train/val split is at the patch level via
`create_training_sets.py`; test slides have no ground-truth grades in
the code path. To use this framing we need to (a) carve a tuning
split out of whatever labeled test set exists, and (b) persist slide-
level ground-truth grades somewhere the sweep can read them.

**2. Clinical operating point.**
Pick threshold to satisfy a named clinical requirement stated *before*
looking at results — e.g., "slide-level sensitivity for ≥1R rejection
must be ≥ 0.95 per ISHLT recommendations; the smallest threshold
satisfying this on the tuning set was 0.XX." This anchors the number
to a constraint, not an outcome, which is generally more persuasive in
a clinical journal.

**3. Sensitivity analysis (always include this regardless).**
Figure: slide-level headline metric (y-axis) vs. threshold from 0.5 to
0.99 (x-axis). If the curve is approximately flat across that range,
argue: "performance is robust across the operating band we
considered." This neutralizes the cherry-picking objection even if
framings 1 or 2 are used — the answer doesn't materially depend on
the exact threshold.

### What to write in Methods

- How the threshold was chosen (which slides, which objective, which
  grid of candidate thresholds).
- Explicit statement that the test set was not consulted during
  threshold selection.
- Justification of the objective (sensitivity-first, accuracy, κ,
  etc.) with a clinical citation where possible.

### Anti-patterns to avoid

- Sweeping threshold on the test set and picking the best.
- Changing threshold after seeing test-set confusion matrices.
- Reporting only the final threshold without the sweep / sensitivity
  analysis — reviewers treat unexplained constants as red flags.

### Engineering follow-ups this would require

- Promote `PREDICTION_THRESHOLD` from a config constant to a
  `--threshold` CLI flag on `wsi.diagnose`, so tuning-set runs and
  test-set runs can use different values without editing config.
- Extend `wsi/threshold_sweep.py` to compute **slide-level** grade
  accuracy against a ground-truth file, not just surviving-patch
  counts, so the sweep directly reports the objective we would
  optimize.
- Document the tuning/test slide split (or the absence of one) in
  `DEVELOPMENT_LOG.md`.

---

## Backbone: UNI2-h vs. the 2019 ResNet-50 baseline

**Expected reviewer question.** "A pathology foundation model is
overkill for a six-class rejection-grading task — did you verify the
complexity is justified?"

**Answer.** Yes, by direct head-to-head. The repo ships two
interchangeable backends behind the same `BackendClassifier` API: the
2019 paper's ResNet-50 (ImageNet-pretrained, fine-tuned end-to-end)
and UNI2-h (`MahmoodLab/UNI2-h`, ViT-H/14, pathology-pretrained on
~200M H&E patches from 300k MGB slides). Same preprocessing, same
train/val split, same slide-level aggregation — only the patch
encoder differs. Report both sets of numbers; the domain-pretraining
uplift is the argument for UNI.

**Why UNI2-h specifically.** DEVELOPMENT_LOG.md §"Backbone: UNI2-h"
has the full survey. Summary: UNI2-h is the current SOTA with clean
access, DINOv3 was rejected because arXiv 2509.06467 documents
natural-image DINOv3 degrading on WSIs, and Virchow2 / Phikon v2 /
H0-mini / GenBio-PathFM were surveyed but offered weaker track records
or access friction.

**Anticipated weakness.** UNI2-h was pretrained on MGB slides —
overlap with our heart-transplant cohort is plausible but not
quantifiable from the public model card. Disclose this as a known
limitation in Discussion. A reviewer with access to the MGB-slide
metadata could request verification.

---

## Frozen backbone + probe head (no fine-tuning)

**Expected reviewer question.** "You're leaving accuracy on the table
by not fine-tuning."

**Answer.**

- Frozen-backbone linear probe is the **canonical UNI evaluation
  protocol** — published numbers are reported this way, so our results
  are directly comparable to the UNI paper and downstream benchmarks.
- Fine-tuning a 681M-parameter ViT-H on this dataset (7.7k training
  patches, 6 classes, one slide per class in the rare tail) is a high
  overfitting risk. Frozen features + a lightweight head is the
  defensive choice on a small dataset.
- Practical: backprop through ViT-H does not fit in an 8 GB card at
  any useful batch size. Fine-tuning would require hardware we
  don't have and compute that a reviewer wouldn't expect.
- Encoding is expensive and one-shot; training the head is cheap and
  lets us iterate on hyperparameters, head architecture, and class
  weighting without re-encoding.

**If a reviewer still pushes.** Offer linear-probe fine-tuning of the
last transformer block as an ablation — preserves the compute budget
but opens the top of the backbone. Not currently implemented.

---

## Head architecture: linear first, MLP as a pre-registered alternative

**Expected reviewer question.** "Why MLP and not linear?" (or vice
versa — whichever you report as primary.)

**Answer.** Both are trained, both are evaluated, and the decision
rule is fixed *before* looking at test performance: select the head
with higher **validation** accuracy. In this run the MLP
(`Linear(1536→512) → ReLU → Dropout(0.4) → Linear(512→6)`) gave val
acc 0.9362 vs. the linear probe's 0.9311, so MLP is the headline head.

This framing makes the choice *mechanical*, not aesthetic: if the
paper is rerun with different data and linear wins, the paper reports
linear. The selection rule — not the winner — is what goes in
Methods.

Report *both* numbers in the paper. A flat linear-vs-MLP gap is a
signal that UNI features are near-linearly separable for this task,
which is itself an interesting finding.

---

## Class-imbalance handling: balanced cross-entropy

**Expected reviewer question.** "Hemorrhage has 147 patches and
Healing has 3707 — how did you prevent the majority classes from
dominating?"

**Answer.** Class-weighted cross-entropy with sklearn-style
inverse-frequency weights:
`w_i = N_total / (num_classes × N_i)`
(see `cardiac_acr/backends/uni/train.py:_class_weights`).

Alternatives considered:

| Method | Why rejected |
|---|---|
| Oversampling minorities | Risks overfitting the 147 Hemorrhage patches — same images seen many times per epoch. |
| Focal loss | Adds `γ` hyperparameter without a tuning set to justify it; weighted CE is one-knob-fewer. |
| SMOTE / feature-space synth | Not standard for deep-feature pipelines; invents examples that may not correspond to real histology. |
| Uniform weights | Reduces to accuracy optimization, which the majority classes already dominate. |

The choice is defensible as the simplest, most interpretable option
that addresses the imbalance without inventing data.

**Known limitation.** Hemorrhage weight (~13.3 in the current split)
is extreme and the model has effectively no held-out data to calibrate
against. Acknowledge this in Discussion — Hemorrhage performance
numbers should be reported with a caveat, not as a primary result.

---

## Slide-level train/val split (not patch-level)

**Expected reviewer question.** "How did you prevent train/val
leakage through patches from the same slide?"

**Answer.** The split is at the **slide** level, not the patch level.
`cardiac_acr/preprocessing/create_training_sets.py` uses a frozen
`TRAIN_SLIDES` list (preserved verbatim from the 2019 paper's
`Create_Training_Sets_V8.ipynb`); every patch from a listed slide goes
to `Training/`, everything else to `Validation/`. No slide contributes
patches to both splits.

**Why this matters.** Patches from the same slide share staining
batch, scanner calibration, patient genetics, and H&E preparation
idiosyncrasies. A patch-level random split would give inflated
validation numbers — the model partially memorizes per-slide artifacts
rather than learning transferable histology. Slide-level split is the
standard in the WSI literature (cite: Kim et al. 2022, Campanella et
al. 2019) and the correct framing for the paper.

**Reproducibility.** The frozen slide list is in-repo source (not a
random seed), so the exact split can be recreated on any machine.

**Known asymmetry.** Hemorrhage is annotated on a single training
slide, so Validation/Hemorrhage is empty (0 patches). Per-class
validation metrics for Hemorrhage are undefined. Report this in a
table footnote and in Discussion.

---

## 1R2 focus counting via separate segmentation, not patch counts

**Expected reviewer question.** "Why a separate segmentation pipeline
for 1R2 focuses instead of just counting 1R2-classified patches?"

**Answer.** The ISHLT grade rule is defined over the **number of
discrete focuses** — contiguous lymphocytic infiltrates — not the
number of patches. A single large focus can cover many 224×224
patches; counting patches inflates the focus count and pushes slides
into higher grades. `cardiac_acr/wsi/count_1r2.py` runs a dedicated
segmentation pipeline that groups adjacent 1R2-positive regions into
focuses and returns an integer count, which then drives the
1R2-vs-2R boundary in `wsi.diagnose`.

**Design consequence.** The patch-level classifier's 1R1A ↔ 1R2
confusion (the weakest point on the validation confusion matrix) is
*partly* decoupled from slide-level grading — a slide's final grade
depends on focus topology, which `count_1r2` reasons about
independently. Report slide-level accuracy/κ as the primary clinical
metric; patch-level 1R1A/1R2 confusion is secondary.

---

## Patch preprocessing: 224×224, 50% tissue floor

**Expected reviewer question.** "Why 224×224 patches and a 50% tissue
threshold?"

**Answer.**

- **224×224** matches UNI2-h's native input size (ViT-H/14 at 224
  means 16×16 = 256 attention tokens + 8 register + 1 CLS) and the
  ImageNet-pretraining convention the ResNet-50 baseline uses.
  Identical patch geometry on both backends keeps the comparison
  apples-to-apples.
- **50% tissue floor** removes near-empty patches (slide background,
  pen-mark coverage, low-information glass regions) that would
  dominate the feature distribution without contributing pathology
  signal. 50% is the threshold from the 2019 pipeline; sensitivity
  analysis on this value is a reasonable ablation but has not been
  run.

---

## Section stubs — to expand as the manuscript matures

- **Reporting protocol.** Which metrics go in the headline table
  (slide-level accuracy, Cohen's κ vs. pathologist, per-class F1),
  which go in supplementary (AUROC, confusion matrix, calibration).
  Decide before finalizing test-set results.
- **Human comparator.** Is there a pathologist-agreement number from
  the original 2019 cohort that we can use as a clinical baseline?
- **Failure-mode analysis.** Plan an appendix figure: for slides the
  model grades wrong, display the top-confidence patches and compare
  to ground-truth annotation. Anticipates "where does it fail?"
  question.
- **Ethics / IRB.** Document cohort provenance, deidentification, and
  IRB protocol number.
- **Code + data release.** State what will be public (code, trained
  head weights, patch-level predictions) and what will not (raw
  slides, patient metadata).

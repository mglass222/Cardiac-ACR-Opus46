# Known Weaknesses

Honest self-assessment of project limitations, ranked by how much each
would hurt in peer review. Companion to `PROJECT_NOTES.md` — that file
captures *defenses* for choices we've made; this file catalogs issues
we either can't defend yet or haven't addressed.

Keep this file current. When an item is fixed, move it to a
"Resolved" section at the bottom with a brief note and a date so the
history is preserved.

---

## Methodology (paper-blockers)

### 1. No held-out test slides with ground-truth grades

The "validation" split is patch-level, within the 80/20 slide
assignment used for training. `data/WSI/Test/` is unlabeled inference
input. There is no slide-level accuracy / Cohen's κ / AUROC number
backed by pathologist-adjudicated grades — which is the clinical
metric a cardiology reviewer will demand. Everything downstream
(threshold choice, backend comparison, MLP vs. linear) is currently
evaluated on signals that don't reflect clinical performance.

### 2. Single 80/20 split, no cross-validation, no confidence intervals

Every headline number is a point estimate from one slide-level split.
At ~150 slides, ~10 slides shifting between train/val could move val
accuracy several percentage points. Reviewers will ask for either
k-fold cross-validation or at least bootstrap confidence intervals on
the reported numbers.

### 3. Grade-rule magic numbers

`wsi.diagnose.diagnose` uses hardcoded aggregation rules:
`1R2 count ≥ 2 → 2R`, `≥1 1R1A patch → 1R1A`. Neither is justified
against ISHLT spec, clinical practice, or tuning-set data. The
"≥1 1R1A patch" rule is particularly fragile — one false-positive
patch in thousands flips a slide's grade.

### 4. UNI2-h pretraining-cohort overlap

UNI2-h was pretrained on ~300k Mass General Brigham slides. If our
cohort is MGB (or IRB-linked), test-set contamination is
unquantifiable. Not fatal but must be disclosed in Discussion; ideally
cross-check the cohort source against MGB metadata where possible.

### 5. No calibration analysis

Threshold discussions only make sense if softmax probabilities are
calibrated. We don't currently compute Expected Calibration Error
(ECE), reliability diagrams, or Brier score. A reviewer pointing at
"threshold 0.99" without calibration evidence has a strong objection.

---

## Data / statistical power

### 6. Hemorrhage is effectively unvalidated

147 train patches, 0 validation patches (annotated on a single
training slide). Per-class metrics for Hemorrhage are undefined.
Reporting it alongside the other classes without a caveat is
misleading.

### 7. Quilty nominal 100% accuracy is on 141 validation patches

Likely real — Quilty lesion morphology is distinctive — but the
confidence interval on 141 samples isn't narrow. Wilson or
Clopper-Pearson intervals should accompany the number.

### 8. No inter-rater reliability on the annotations

Who drew the ImageScope XML annotations? How many pathologists?
What's 1R1A-vs-1R2 agreement among the annotators? Without this, the
label noise floor is unknown — and the model's validation accuracy
may already be *at* the label-noise ceiling, in which case further
model improvements are unachievable by definition. This is the kind
of ceiling argument reviewers use to block "the model is X% accurate"
claims.

---

## Model / architecture

### 9. Patch-level classifier has no context

224×224 at full resolution is ~100 µm. 1R1A vs. 1R2 distinction
depends on focus count, which is an image-wide property. `count_1r2`
patches this with a hand-designed segmentation step, but the patch
classifier itself is blind to context that a pathologist uses
routinely (vessel topology, myocyte orientation, infiltrate
distribution).

### 10. ISHLT grade coverage is incomplete

The grade rule emits `{0R, 1R1A, 1R2, 2R}`. Real ISHLT 2004/2013
grading includes `1R1B`, `3R`, and AMR (antibody-mediated rejection)
grades. Either the dataset doesn't contain them (should be stated
explicitly) or the code silently under-reports (bug).

---

## Engineering / reproducibility

### 11. No test suite

The preprocessing chain (slide → filter → tile → patch → tissue
filter) is all multi-process file IO. No unit or integration tests. A
regression in tile scoring or tissue masking would be invisible until
downstream results shifted. Ground-truth PNG fixtures + pixel-hash
assertions on filter outputs would be cheap insurance.

### 12. No random-seed discipline

No visible `torch.manual_seed` / `numpy.random.seed` / `random.seed`
in the training path. `torch.compile` also adds nondeterminism. Two
runs probably don't match, which makes reproducibility claims
hand-wavy.

### 13. No experiment tracking

Results print to stdout. Comparing MLP vs. linear means re-running or
reading two terminal scrollbacks. W&B / MLflow / CSV logger would
make the linear-vs-MLP, threshold, and backend comparison concrete
and figurable.

### 14. ResNet backend parity is asymmetric

`python -m cardiac_acr evaluate --backend resnet` raises and tells the
user to run `stats/patch_level_stats` instead. The
"interchangeable backends behind one API" claim in the README is only
half-true — the two backends can't be compared head-to-head on
identical metrics without extra glue code.

### 15. Annotation outputs deferred

`wsi/diagnose.py` doesn't call `annotate_png` or `annotate_svs`. Hard
to do error analysis or show a pathologist "here's what the model
saw" without the overlays. This matters for both the paper's figures
and for iterative debugging. The annotation functions exist in the
package and read from the same `Saved_Databases/` pickles; wiring
them into the main loop is a TODO.

---

## Priority recommendation

If one week of effort were available, the highest-leverage fixes:

- **#1 (test slides with ground-truth grades)** — even 20
  pathologist-graded slides would enable slide-level accuracy / κ
  reporting, which unlocks the clinical numbers reviewers expect.
- **#5 (calibration analysis)** — ECE on the validation predictions
  is ~20 lines of code and makes the threshold defense concrete.

Those two together let you defend the threshold (PROJECT_NOTES.md §1),
the backend choice, and the headline clinical number. Everything else
is polish by comparison.

---

## Resolved

*(Nothing yet. Add entries as items are addressed, with a date and a
pointer to the commit / PR that fixed them.)*

# Project Decisions

Running log of design decisions, with the evidence behind them. Newest at the bottom.

## 1. Raw data is read-only, original WiSARD naming is preserved

**Date:** 2026-07-19

**Decision:** The dataset under `data/` keeps WiSARD's original folder structure and
filenames (e.g. `210417_MtErie_Enterprise_VIS_0003/..._00000200.jpeg`). No script may
rename, move, or modify raw data; all preprocessing writes to `data/derived/`.

**Why:** An earlier renaming scheme (`__ir__00000112.jpeg`) destroyed the flight-sequence
identity encoded in the filenames. Sequence identity is required for (a) the
sequence-based train/val/test split and (b) programmatic verification of RGB↔IR pairing.
Read-only raw data also keeps every preprocessing step reproducible from scratch.

## 2. Filename parsing convention

**Date:** 2026-07-19

**Decision:** WiSARD names are parsed as
`{date}_{location...}_{VIS|IR}_{camera}_{frame}.ext`, anchoring on the `VIS`/`IR` token
(the location may itself contain underscores). `sequence_id = {date}_{location}` —
modality-agnostic, so both streams of one flight always stay in the same split.

## 3. Pair manifest as single source of truth

**Date:** 2026-07-19

**Decision:** `data/derived/manifest.csv` (built by `src/utils/build_manifest.py`) defines
the synchronized pairs: one row per pair with
`sequence_id, frame_id, rgb_image, rgb_label, ir_image, ir_label`. All downstream code
(splits, loaders) consumes the manifest, never the raw folders directly.

- Paths are stored relative to the project root (portable across machines).
- Rows are sorted by `(sequence_id, frame_id)` so rebuilds are byte-identical.
- `src/utils/validate_manifest.py` enforces the contract (files exist, images readable,
  labels are valid single-class YOLO, filenames consistent, no duplicates) and exits
  non-zero on any error.

## 4. VIS↔IR pairing rule

**Date:** 2026-07-19

**Decision:** For the sampled sequence, VIS folder `*_VIS_0003` pairs with IR folder
`*_IR_0004` (camera number +1), and VIS frame `N` pairs with IR frame `N+1`.

**Evidence:** Verified visually on multiple frames (side-by-side viewer
`src/utils/view_pair.py`): same scene, same people in both modalities.

**Caveat:** Both offsets are assumptions confirmed only on this one sequence. Re-verify
on other sequences when the full dataset arrives; the manifest builder fails loudly if a
constructed IR path does not exist.

## 5. RGB↔IR registration: fixed affine map, early fusion feasible

**Date:** 2026-07-19

**Decision:** The IR image is a near-centered crop of the RGB view, described by a single
linear map per axis (IR→RGB, normalized coordinates), estimated by
`src/utils/estimate_registration.py` from 891 matched ground-truth box centers:

```bash
x_rgb = 0.6813 · x_ir + 0.1710   (IR spans RGB x ∈ [0.17, 0.85])
y_rgb = 0.9304 · y_ir + 0.0500   (IR spans RGB y ∈ [0.05, 0.98])
```

- Implied IR footprint in RGB pixels: ~2616×2010 (aspect 1.30) vs IR sensor native
  640×512 (aspect 1.25) — consistent with a fixed rigid camera mounting.
- Residuals: median 29 px, mean 47 px, p95 136 px (RGB scale; person boxes are
  ~60–100 px tall). Diagnostic plot: `data/derived/registration_fit.png`.

**Consequences:**

- The 4-channel early-fusion baseline is feasible: crop RGB to the IR footprint,
  downscale, stack IR as the 4th channel.
- The error is not uniform over time — spike clusters (frames ~45–60, ~160–220) reach
  100–460 px, from either close-standing-people mismatches or genuine time-sync/gimbal
  lag. Residual misalignment is the argument for mid-level feature fusion (tolerant to
  misalignment) as the main architecture; early fusion serves as the baseline.

**Caveats:**

Fitted on one sequence at one altitude. Re-estimate on 2–3 more sequences
from the full dataset before treating the transform as global. Spike frames not yet
inspected visually.

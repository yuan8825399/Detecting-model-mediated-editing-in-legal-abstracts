# CLAW-Edit manuscript audit

Audit date: 2026-10-26
Benchmark: CLAW-Edit v3, source-screened formal benchmark

## Verified

- 493 source groups, 493 H records, 7,888 E records, 8,381 records total.
- Every source group has 17 records: one H and 16 E records (four providers x four levels).
- Source-group splits are train 243, dev 81, calibration 83, test 86. Split intersections are zero.
- `provenance_audit.json` reports 7,888 raw responses rechecked and zero retained-response structural/hash errors.
- The saved pooled and leave-one-provider-out baseline predictions pass train/calibration/test source-group isolation: no group is shared across those stages.
- The saved v3 baseline is character 2-4-gram TF-IDF plus balanced logistic regression. Calibration FPR is 4.82%; test FPR is 16.28% for abstract-only input and 13.95% for title-plus-abstract input.
- Two 1% ROC-FPR summaries are kept separate. `audit_results.json` records conservative empirical thresholds without interpolation (the reproducibility audit); the final manuscript reports linear interpolation on the saved pooled ROC curve (`E1`/`E2`/`E3`/`E4` TPR = 0.010/0.096/0.061/0.090) and labels these as ranking summaries rather than deployment thresholds.

## Not verified or absent

- No PAL-CP/conformal implementation, saved joint-lattice predictions, q-values, or per-record prediction sets were found in the v3 package. The manuscript's PAL-CP coverage, precision, and error counts cannot be independently recalculated from the available files.
- No saved 4x4 cross-model matrix was found. Leave-one-provider-out source isolation is auditable; the manuscript's full matrix numbers are not.
- No stylometric/fused detector or ablation result is produced by the v3 baseline package. The former Table S2 values cannot be reconciled to this package without the missing code and run manifest.
- The exact E1-E4 operation prompt files and collection/record schemas are available in the v1 protocol directory and are copied under `protocol/` in this audit package. A public repository URL is not present. Local code and evaluation artifacts are listed with SHA-256 hashes in `audit_results.json`; a submission version needs a versioned repository or controlled-access archive with a stable accession.

## 1% operating-point summary

The conservative values are in `audit_results.json` under `conservative_empirical_1pct_fpr`; the manuscript's interpolation values are under `manuscript_interpolated_1pct_roc_fpr`. Both are calculated from saved pooled test scores, and neither is the calibration-selected deployment threshold.

## Reproduction inventory

The package records the source paths, commands and file hashes needed to rebuild the v3 lexical baseline. It intentionally does not copy raw provider responses or the full text corpus into this output because redistribution permission is not established. Authors should deposit the permitted data or provide a controlled-access mechanism before submission.

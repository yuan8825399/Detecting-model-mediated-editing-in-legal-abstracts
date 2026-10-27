# Data access and redaction record

This release is intentionally metadata-only for the corpus text. The source
files contain Chinese legal abstracts and model responses obtained from
third-party services. Public redistribution rights for those texts were not
confirmed during the audit.

The following fields were removed from the public corpus metadata files:

- `abstract_raw`
- `abstract_clean`
- `source_title_raw`
- `source_title_clean`

The generation queue was also redacted to remove source titles and local raw
response paths. It retains provider, model, prompt ID, operation level,
generation time, status, failure reason, and SHA-256 fields. Each record is
marked `text_release=omitted_pending_rights_review` or
`source_text_release=omitted_pending_rights_review`.

The following local files were not copied into this release because they contain
source or generated text:

- `research_benchmark_legal_edit_v3/evaluation/benchmark_records.jsonl`
- `research_benchmark_legal_edit_v3/evaluation/content_review_v1/review_index.jsonl`
- `research_benchmark_legal_edit_v3/evaluation/content_review_v1/review_sample.jsonl`
- `research_benchmark_legal_edit_v1/source_registry/sources.csv`
- `中国法学/DATA/ai_rewrite/generated_dataset.csv`
- `中国法学/DATA/ai_rewrite/generation_tasks.csv`
- provider raw-response directories and evidence JSON files

To create a controlled-access or institutionally hosted text release, obtain
permission from the relevant source and service rights holders first. Then
reconcile the released files against the hashes in the local audit package and
record the access conditions in a new versioned manifest.

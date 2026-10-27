# CLAW-Edit record schema

`source_registry/sources.csv` stores one H record per source abstract. The
`source_title_raw` and `source_title_clean` fields come from the first column
of the corresponding annual XLSX under `中国法学/`, using the year and source
row encoded in `source_id`. `title_sha256` hashes the cleaned title. The title
is preserved unchanged across H and E records; E operations do not rewrite it.
The `abstract_raw` field is preserved before normalization; `abstract_clean` is
the only text used as LLM input or detector input. `abstract_sha256` hashes
the UTF-8 cleaned H text.

`source_registry/generation_queue.csv` stores one row per provider and E1-E4
task. It repeats `source_title_raw`, `source_title_clean`, and `title_sha256`
so a collector can verify the source context without joining files.
`parent_record_id` must always be `{source_id}-H`; E records are never chained
from another E record. `input_sha256` must equal the H `abstract_sha256`.
`prompt_sha256` identifies the exact prompt template.

`processed/records.jsonl` and `processed/records.csv` contain H records and
accepted or rejected E records. `output_sha256` hashes the normalized output;
`raw_evidence_sha256` hashes the complete saved provider response.
`normalized_edit_distance`, `numeric_retention`, and `chinese_ratio` are
descriptive metrics and must not be used to select records by detector score.
`evaluation_role=pilot_only` marks records used in smoke tests; exclude their
entire source group from formal metrics even though the original fixed
`group_split` remains recorded.

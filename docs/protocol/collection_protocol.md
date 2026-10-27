# CLAW-Edit collection protocol

## Fixed operation rules

Every E task uses the original H abstract as input. Do not chain E1 into E2,
E2 into E3, or E3 into E4. Keep the legal topic, research question, core
claim, legal references, numbers, and conclusion. Do not add facts, cases,
statutes, data, citations, or external knowledge. Do not request a title.
The original title is supplied as fixed metadata and must be copied unchanged
to the processed H/E record; it is not an output target.

## Required evidence

For every attempt, preserve the exact prompt, input text, model name,
model snapshot, access channel, generation timestamp, and raw response. Store
the response under `raw/<provider>/<record_id>.json`. A response is accepted
only after the file exists and its hash is recorded.

## Quality rules

Reject only for empty output, prompt leakage, obvious truncation, low Chinese
language ratio, provider failure, or a predeclared content-drift rule. Do not
filter by detector score, writing quality preference, or whether the output
looks machine generated.

## Split rule

All records sharing `group_id` stay in one split. The split is fixed before any
model training or detector evaluation. H records are never regenerated.

## No human editing conditions

This benchmark has no human polishing, human rewriting, or human post-editing
condition. Human review is limited to source verification and fixed quality
acceptance checks; it does not create an additional text version.

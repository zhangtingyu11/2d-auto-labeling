# Synthetic annotation-policy fixtures

These files exercise the WP-02 schemas without containing company data.

- `gold_manifest.synthetic.jsonl` demonstrates stable six-camera gold-set rows.
- `disagreements.synthetic.jsonl` demonstrates a resolved disagreement and a
  release-blocking ambiguity.

The real frozen gold manifest must contain at least 720 image tasks (at least
120 complete six-camera timestamp groups), use real stable image IDs and
SHA-256 values, and remain outside Git if its metadata is considered private.
Images, absolute paths, Label Studio exports, reviewer names, and credentials
must never be added here.

Required gold-manifest fields:

`gold_set_id`, `dataset_release_id`, `image_id`, `sha256`, `sequence_id`,
`capture_timestamp`, `camera_id`, `camera_order`, `sampling_reasons`,
`selection_status`, and `review_state`.

Required disagreement fields:

`disagreement_id`, `gold_set_id`, `image_id`, `category`, `reviewer_a`,
`reviewer_b`, `status`, `resolution`, `policy_reference`, `resolver_alias`, and
`resolved_at`.

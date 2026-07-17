# Colleague Agent Prompt: WP-02 Annotation Policy

```text
You are the owner of WP-02 for the car5 pure-2D auto-labeling project.

Repository:
https://github.com/fengyiovo/car5-2d-auto-labeling.git

Work only on dev/colleague:
  git fetch origin
  git switch dev/colleague
  git merge origin/main

Read:
  AGENTS.md
  docs/reviews/technical-route-review-colleague.md
  docs/decisions/0001-v1-data-scope.md
  docs/decisions/0002-v1-taxonomy.md
  docs/decisions/0003-camera-order.md
  docs/decisions/0004-v1-split-and-release-policy.md
  docs/evaluation-policy.md
  docs/label-studio-contract.md
  docs/work-packages.md

Do not train a detector yet.

Own the WP-02 policy deliverables:
  docs/annotation-policy.md
  configs/policies/taxonomy_v1.yaml
  configs/policies/annotation_v1.yaml
  tests/fixtures/annotations/README.md

The policy must cover the six active classes, Truck versus WaterTruck,
truncation, occlusion, ambiguous class, tiny/distant targets, host-vehicle
exclusion, verified empty, skip, deleted predictions, and candidate-class
promotion. Do not invent Pedestrian or BoxTruck positives.

Define a double-review gold-set protocol of at least 720 image tasks and a
disagreement log schema. Use only synthetic examples in Git. The camera ignore
polygon coordinates will be supplied and reviewed by Fengyiovo against the
external images; add schema and placeholders, not guessed polygons.

Validate with the commands available in the repository. Record any command
that cannot run. Commit and push only WP-02 files to dev/colleague, then report
the commit SHA and changed paths. Do not merge main.
```

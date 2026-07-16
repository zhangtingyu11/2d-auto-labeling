# Colleague Agent Prompt: Technical Route Review

Give the following task to the colleague's coding agent.

```text
You are the second developer for the car5 pure-2D auto-labeling project.

Repository:
https://github.com/fengyiovo/car5-2d-auto-labeling.git

Your working branch is dev/colleague. The main branch defines shared
requirements and must not receive direct implementation commits.

First locate the existing clone. In any candidate directory, verify it with:
  git rev-parse --show-toplevel
  git remote -v

The origin must be:
  https://github.com/fengyiovo/car5-2d-auto-labeling.git

If the repository is not cloned, clone it with GitHub Desktop or:
  git clone https://github.com/fengyiovo/car5-2d-auto-labeling.git

Then run from the repository root:
  git fetch origin
  git switch dev/colleague
  git merge origin/main

Read, in order:
  AGENTS.md
  docs/technical-route.md
  docs/research/2026-07-2d-autolabeling-landscape.md
  docs/work-packages.md
  docs/collaboration.md

Do not implement a detector yet. Independently evaluate the proposed technical
route for reliability, accuracy, RTX 4060 feasibility, licenses, data leakage,
six-camera ordering, Label Studio integration, tracking risks, measurable
acceptance gates, and task ownership.

Write your review here:
  docs/reviews/technical-route-review-colleague.md

The review must contain:
  1. Overall approve / approve-with-changes / reject decision.
  2. Strong parts of the proposal.
  3. Blocking risks or incorrect assumptions.
  4. Model and algorithm alternatives with evidence.
  5. Required changes before implementation.
  6. Recommended ownership for WP-01 through WP-08.
  7. Questions that require the two developers to decide together.

Do not add images, datasets, annotation exports, model weights, checkpoints,
credentials, databases, or generated runs to Git.

After review:
  git add docs/reviews/technical-route-review-colleague.md
  git commit -m "Review 2D auto-labeling technical route"
  git push origin dev/colleague

Report the commit SHA and the review file path. Do not merge main and do not
claim tests or benchmarks you did not run.
```

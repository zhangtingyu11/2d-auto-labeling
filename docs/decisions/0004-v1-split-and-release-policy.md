# ADR 0004: v1 Split and Release Policy

Status: accepted

Date: 2026-07-16

## Context

The current data is one collection day and route. Timestamp gaps divide it into
five natural sequences, but rare classes are localized:

- `Car`, `WaterTruck`, and most `Sign` examples occur in early sequences;
- `Bulldozer` occurs only in the last sequence;
- several truck and excavator object IDs persist for long intervals.

A random image split would leak adjacent frames and identical objects. A single
70/15/15 chronological split would leave important classes absent from train,
validation, or test.

## Decision

Use five leave-one-sequence-out development folds over the natural sequences
`s00` through `s04`.

- All six cameras at one timestamp stay in the same fold.
- Non-keyframe context stays with its enclosing natural sequence.
- Near-duplicate hashes and persistent source object IDs are reported for every
  fold; they are not hidden by aggregate metrics.
- A class absent from either fold training data or fold validation data is
  reported as `not_evaluable` for that fold.
- Detector comparison uses identical folds and reports supported slices rather
  than inventing a complete per-class test score.
- After model selection, the development model may train on all five current
  sequences for review-assistance use.

The current dataset is not a production acceptance test. A second independent
day, route, or site is required as a frozen external test before any class-
camera slice can automatically accept annotations.

## Product Consequence

The first product release is review-assistance only. It may prioritize images,
draw proposals, and propagate candidates, but every task still requires human
submission. Automatic acceptance remains disabled until an independent test
release satisfies `docs/evaluation-policy.md`.

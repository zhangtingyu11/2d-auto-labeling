# ADR 0005: Single Owner and Pilot-First Training

Status: accepted

Date: 2026-07-17

## Decision

Fengyiovo owns the implementation until the first complete tool release. Work
packages formerly assigned to a colleague move to `dev/fengyiovo`; historical
reviews remain as evidence but are not an execution dependency.

Detector training uses three gates:

1. a 5 to 10 minute pipeline and overfit smoke test;
2. a 1 to 2 hour RTMDet-S pilot with held-out evaluation and visual review;
3. long training only after the pilot demonstrates finite decreasing loss,
   non-empty predictions, supported-class learning, camera-slice output, and
   acceptable host-region behavior.

The pilot and long run use identical release manifests, fold identities,
taxonomy, evaluator, and camera policies. The pilot cannot be promoted solely
because GPU utilization is high or training loss is low.

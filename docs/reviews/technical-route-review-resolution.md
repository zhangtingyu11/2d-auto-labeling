# Resolution of Independent Technical-Route Review

Date: 2026-07-16

Reviewed commit: `52d1561`

Main-branch review import: `2957348`

Overall disposition: accepted with changes. The route remains RTMDet baseline,
RF-DETR challenger, review-only temporal assistance, and Label Studio human
review. The review changed the data scope, taxonomy, split, hardware probe,
and release policy before implementation.

## Finding Resolution

| Review finding | Resolution | Binding document |
| --- | --- | --- |
| 928 keyframes versus 4,640 frames per camera | v1 supervised set is 5,568 reviewed keyframe images; non-keyframes are temporal context | ADR 0001 and data audit |
| Eight versus seven/six classes | v1 has six active classes; `Pedestrian` and `BoxTruck` are inactive candidates | ADR 0002 |
| Camera identity and order | Fixed to `front`, `front_left`, `front_right`, `back`, `back_left`, `back_right` | ADR 0003 |
| Split leakage and one-route limitation | Five natural-sequence development folds; independent collection required for acceptance | ADR 0004 |
| Acceptance gates not executable | Matching, slices, sample minima, Wilson bound, timing, and regression rules defined | Evaluation policy |
| RTX 4060 assumptions | Laptop GPU and 8,188 MiB recorded; RTMDet-S 640 is the first probe | Hardware profile |
| Label Studio semantics | Community 1.23.0 pinned; explicit synthetic round-trip contract required | Label Studio contract |
| ByteTrack ego-motion risk | Tracker output remains a proposal; resets, detector support, and forward/backward checks remain mandatory | Technical route section 9 |
| SAM 2.1 drift risk | Tiny/Small is an optional bounded offline challenger, never v1 ground truth | Technical route section 9.4 |
| Dependency licenses | Phase-one allowlist and explicit exclusions recorded | License register |

## Gate Status

Completed at decision level:

- population and timebase decision;
- final-export count, order, class, and checksum audit;
- active taxonomy decision;
- canonical camera order;
- development-fold and independent-test policy;
- evaluation specification;
- hardware capacity record;
- Label Studio version and contract specification;
- phase-one license register;
- work-package ownership.

Still required before detector training:

- WP-01 executable manifest builder, validators, fold report, and parser tests;
- WP-02 annotation policy, camera ignore polygons, gold manifest, and agreement
  review;
- executable evaluator matching tests;
- synthetic Label Studio round-trip fixture passing on Community 1.23.0.

WP-01 and WP-02 may proceed in parallel. WP-03 and WP-04 remain blocked until
the detector-start gate in `docs/work-packages.md` passes.

# car5 v1 2D Annotation Policy

Status: accepted for implementation

Dataset release: `car5_20260611_day_02_labels_v1`

## Scope

This policy governs the six-camera 1920 x 1080 car5 keyframes and every model
prediction reviewed in Label Studio. A reviewed image is either a set of valid
active-class boxes or an explicit verified empty image. Skip never means empty.

## Active Classes

| Class | Include | Exclude and confusion rule |
| --- | --- | --- |
| `Car` | Passenger cars, SUVs, vans, identifiable light utility vehicles | Do not use for mining dump trucks or water trucks. |
| `Truck` | Cargo, rigid dump, mining dump, and articulated trucks | Use `WaterTruck` when tank or spraying equipment is identifiable. Never box the camera host vehicle. |
| `Bulldozer` | Earthmover whose defining attachment is a front blade | Use `Excavator` when boom, stick, and bucket/breaker are identifiable. |
| `Excavator` | Excavator identifiable by boom, stick, and bucket/breaker | Do not label an ambiguous fragment of generic machinery. |
| `WaterTruck` | Truck identifiable by tank, spray bar, cannon, or watering equipment | If the water function is not visible, use `Truck`. |
| `Sign` | Fixed traffic, safety, regulatory, and site-operation signs | Exclude painted vehicle text, advertisements, and unreadable incidental boards. |

`Pedestrian` and `BoxTruck` remain inactive. Their presence must not be silently
mapped into an active class.

## Box Geometry

- Draw one tight axis-aligned box around the visible extent of each target.
- Do not infer the hidden full body behind another object or outside the image.
- Include visible attachments that define the class, such as an excavator boom
  and bucket or a water-truck spray assembly.
- Exclude detached shadows, dust, reflections, and background gaps.
- Adjacent objects receive separate boxes even when they overlap.

There is no hard pixel cutoff. The reviewed data contains correctly
identifiable edge-truncated targets as narrow as about four pixels. Label a
small or truncated target only when its active class is reliable; otherwise
leave it unboxed and record the ambiguity for the second audit pass.

## Truncation and Occlusion

- `truncated=true` when the visible target reaches within one pixel of an image
  boundary.
- `occluded=true` when an external object hides part of the target.
- The host vehicle and configured static camera masks are not ordinary
  occluders and are never annotation targets.
- A tiny edge fragment without enough appearance to distinguish the class is
  ignored. Temporal neighbors may be inspected to decide identity, but the box
  must cover only pixels visible in the current image.

## Host-Vehicle Exclusion

The camera platform is always excluded, including body panels, wheels,
steps, dump body, chains, and reflections of those parts. Camera-specific
polygons live in `configs/cameras/`:

- `front`, `front_left`, `front_right`: no static host mask;
- `back`: upper dump-body structure plus left and right chains;
- `back_left`: left image-edge host body;
- `back_right`: right image-edge host body.

A prediction is ignored as host-region output only when at least 80% of its
area lies inside configured ignore polygons. This avoids suppressing a real
external vehicle that is only partly behind the host structure. Every ignored
prediction count remains visible in evaluation reports.

## Empty and Ambiguous Images

An image becomes `verified_empty` only after scanning the full image at useful
zoom and finding no active-class target. A skipped, cancelled, failed-to-load,
or inactive-class-only task is not automatically verified empty. Ambiguous
objects are logged and reviewed in a second pass; uncertainty is not resolved
by inventing a class.

## Solo Review Protocol

The current implementation has one owner, so the former two-person review is
replaced by a reproducible two-pass audit:

1. First pass reviews boxes in chronological, frame-major, six-camera order.
2. Second pass reviews a shuffled list containing all rare classes, all edge
   truncations, all very small boxes, every camera ignore-region overlap, and a
   representative verified-empty sample.
3. Changes are logged by policy category and exported again through the strict
   validator.
4. The frozen gold set stores image IDs and SHA-256 hashes, never copied images.

The first release remains review-assistance only. No prediction is accepted
without a human submission.

# car5 v1 Data Audit

Audit date: 2026-07-16

Status: verified against the local source tree and the final reviewed Label
Studio export. Raw data and exports remain outside Git.

## Source Population

The source dataset contains one collection day with six camera directories:

| Camera | JPEG files | Resolution |
| --- | ---: | ---: |
| `front` | 4,640 | 1920 x 1080 |
| `front_left` | 4,640 | 1920 x 1080 |
| `front_right` | 4,640 | 1920 x 1080 |
| `back` | 4,640 | 1920 x 1080 |
| `back_left` | 4,640 | 1920 x 1080 |
| `back_right` | 4,640 | 1920 x 1080 |

Other source counts:

- 4,640 ego-pose JSON files;
- 928 3D label JSON files;
- 928 five-feature LiDAR files;
- six camera calibration JSON files.

The camera stream has a median interval of 100 ms. The labeled keyframes have
a median interval of approximately 500 ms. Every labeled frame is exactly
every fifth camera image in sorted source order:

- first keyframe camera index: 0;
- keyframe index delta: 5 for all 927 transitions;
- last keyframe camera index: 4,635;
- four high-rate camera frames follow the last keyframe.

Every one of the 928 label stems has an exact same-stem JPEG in all six camera
directories. Expected keyframe images are therefore `928 x 6 = 5,568`, with
zero missing exact matches.

## Natural Sequence Boundaries

Timestamp gaps greater than one second divide the keyframe stream into five
natural segments:

| Sequence | Keyframe indices | Keyframes | First timestamp | Last timestamp |
| --- | --- | ---: | --- | --- |
| `s00` | 0-230 | 231 | `20260611_165130_934602` | `20260611_165326_739196` |
| `s01` | 231-415 | 185 | `20260611_165336_738711` | `20260611_165509_542487` |
| `s02` | 416-591 | 176 | `20260611_165519_443419` | `20260611_165647_045014` |
| `s03` | 592-761 | 170 | `20260611_165659_644418` | `20260611_165824_145881` |
| `s04` | 762-927 | 166 | `20260611_165831_545770` | `20260611_165954_048384` |

The gaps are approximately 10.0, 9.9, 12.6, and 7.4 seconds in the keyframe
stream. They are sequence-reset boundaries for tracking.

## Final Reviewed 2D Export

External artifact name:

`car5_928frames_6cams_2d_bbox_labelstudio_final_20260716.json`

SHA-256:

`9F34B00F528B8DEF14F484C0C65CC0A5F400A6F027DFCE9CEE177197A36AFA65`

Audit results:

- 5,568 tasks;
- 5,568 unique `(camera, frame_id)` keys;
- zero duplicates;
- zero missing keyframe-camera tasks;
- zero extra tasks outside the 928 labeled stems;
- 928 tasks per camera;
- 2,784 first-half tasks and 2,784 second-half tasks;
- 928 valid six-camera groups in canonical camera order;
- every task has exactly one reviewed annotation record;
- 2,790 explicit reviewed-empty tasks;
- 2,778 non-empty tasks;
- 4,249 reviewed boxes;
- zero cancelled tasks.

Reviewed 2D box counts:

| Class | Boxes |
| --- | ---: |
| `Car` | 62 |
| `Truck` | 3,143 |
| `Bulldozer` | 35 |
| `Excavator` | 846 |
| `WaterTruck` | 84 |
| `Sign` | 79 |

`Pedestrian` and `BoxTruck` have zero reviewed boxes and are not v1 training
classes.

## Consequences

1. The v1 supervised population is the 5,568 reviewed keyframe images.
2. The other 22,272 high-rate images are context-only in v1. They may support
   tracking and propagation but are not human ground truth.
3. Context frames inherit the sequence and evaluation fold of their enclosing
   keyframe segment. They must never cross a train/evaluation boundary.
4. The source is one day and one route. It cannot establish independent-scene
   generalization or authorize automatic acceptance by itself.
5. Rare classes are concentrated in particular segments. Aggregate random
   image splitting would leak near-duplicate scenes and produce misleading
   metrics.

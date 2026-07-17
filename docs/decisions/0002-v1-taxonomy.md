# ADR 0002: v1 Active Taxonomy

Status: accepted

Date: 2026-07-16

## Decision

The immutable v1 active class IDs are:

| ID | Class |
| ---: | --- |
| 0 | `Car` |
| 1 | `Truck` |
| 2 | `Bulldozer` |
| 3 | `Excavator` |
| 4 | `WaterTruck` |
| 5 | `Sign` |

`Pedestrian` and `BoxTruck` are inactive candidate classes. They are not model
outputs, training categories, background aliases, or zero-positive active
classes in v1.

## Reason

Both the source 3D labels and the final reviewed 2D export contain the six
active classes. The final 2D export contains zero `Pedestrian` and zero
`BoxTruck` boxes. Training active heads without positive examples would create
undefined evaluation and misleading product behavior.

## Visual Decision Rules

- `Car`: passenger and light road vehicle that is visually identifiable as a
  car rather than site heavy equipment.
- `Truck`: default heavy road truck, including dump trucks, when no more
  specific active class is visually certain.
- `Bulldozer`: tracked or wheeled earth mover whose blade and body identify a
  bulldozer.
- `Excavator`: machine identifiable by an excavator boom, dipper, and bucket.
- `WaterTruck`: use only when the water tank or spraying equipment is visually
  identifiable. Otherwise use `Truck`.
- `Sign`: external traffic or worksite sign. Text, logos, or markings on the
  host vehicle are not signs.

Annotate an external object when a tight visible box and active class can be
assigned consistently. Truncated external targets remain valid. The visible
host vehicle is excluded by versioned camera ignore polygons, not by deleting
all border boxes.

## Candidate-Class Promotion

Promoting `Pedestrian` or `BoxTruck` requires:

1. an annotation-policy update with positive and confusion examples;
2. a reviewed positive sample set covering relevant cameras and sizes;
3. a new taxonomy version and explicit class-ID migration;
4. revalidation of exports, metrics, and model heads.

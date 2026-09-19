"""Generate the complete BasePlate catalog as JSON.

The output uses the spelling of the Animal and Landscape enum member names in
the game code.  Mixed-landscape pairs are stored once only: their order does
not imply a direction or a separate reversed plate.
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("base_plates.json")

# Five single-animal plates for each landscape.
PURE_PLATES = {
    "MOUNTAIN": ["HAWK", "HAWK", "DEER", "DEER", "GRIZZLY"],
    "RIVER": ["HAWK", "HAWK", "SALMON", "GRIZZLY", "GRIZZLY"],
    "PLAINS": ["SALMON", "SALMON", "DEER", "DEER", "FOX"],
    "WETLAND": ["SALMON", "SALMON", "FOX", "FOX", "HAWK"],
    "FOREST": ["DEER", "GRIZZLY", "GRIZZLY", "FOX", "FOX"],
}

# Each unordered landscape pair has exactly six animal combinations.
# "Bear" from the source list is represented as the Animal enum member GRIZZLY.
MIXED_PLATES = {
    ("MOUNTAIN", "PLAINS"): [
        ("HAWK", "DEER"), ("HAWK", "FOX"), ("DEER", "SALMON"),
        ("GRIZZLY", "SALMON"), ("SALMON", "GRIZZLY", "FOX"),
        ("FOX", "DEER", "GRIZZLY"),
    ],
    ("MOUNTAIN", "RIVER"): [
        ("SALMON", "HAWK"), ("SALMON", "GRIZZLY"), ("HAWK", "DEER"),
        ("HAWK", "GRIZZLY"), ("GRIZZLY", "DEER"),
        ("SALMON", "HAWK", "GRIZZLY"),
    ],
    ("WETLAND", "RIVER"): [
        ("FOX", "SALMON"), ("HAWK", "GRIZZLY"), ("SALMON", "GRIZZLY"),
        ("FOX", "HAWK"), ("SALMON", "HAWK"),
        ("SALMON", "HAWK", "GRIZZLY"),
    ],
    ("MOUNTAIN", "WETLAND"): [
        ("DEER", "FOX"), ("SALMON", "HAWK"), ("HAWK", "DEER"),
        ("GRIZZLY", "SALMON"), ("GRIZZLY", "SALMON", "DEER"),
        ("HAWK", "GRIZZLY", "FOX"),
    ],
    ("FOREST", "RIVER"): [
        ("DEER", "HAWK"), ("DEER", "GRIZZLY"), ("FOX", "SALMON"),
        ("FOX", "GRIZZLY"), ("GRIZZLY", "SALMON"),
        ("HAWK", "DEER", "FOX"),
    ],
    ("FOREST", "WETLAND"): [
        ("DEER", "HAWK"), ("FOX", "HAWK"), ("GRIZZLY", "SALMON"),
        ("DEER", "SALMON"), ("GRIZZLY", "FOX"),
        ("SALMON", "DEER", "HAWK"),
    ],
    ("PLAINS", "WETLAND"): [
        ("DEER", "SALMON"), ("FOX", "HAWK"), ("SALMON", "HAWK"),
        ("DEER", "FOX"), ("SALMON", "HAWK", "FOX"),
        ("SALMON", "DEER", "FOX"),
    ],
    ("MOUNTAIN", "FOREST"): [
        ("DEER", "FOX"), ("HAWK", "DEER"), ("GRIZZLY", "FOX"),
        ("HAWK", "GRIZZLY"), ("FOX", "DEER", "GRIZZLY"),
        ("HAWK", "DEER", "GRIZZLY"),
    ],
    ("FOREST", "PLAINS"): [
        ("DEER", "SALMON"), ("GRIZZLY", "DEER"), ("GRIZZLY", "FOX"),
        ("DEER", "FOX"), ("FOX", "SALMON"),
        ("SALMON", "DEER", "FOX"),
    ],
    ("PLAINS", "RIVER"): [
        ("FOX", "GRIZZLY"), ("DEER", "SALMON"), ("DEER", "HAWK"),
        ("FOX", "HAWK"), ("FOX", "GRIZZLY", "HAWK"),
        ("SALMON", "GRIZZLY", "FOX"),
    ],
}


def make_catalog() -> dict:
    plates = []
    for landscape, animals in PURE_PLATES.items():
        plates.extend(
            {
                "animals": [animal],
                "left_landscape": landscape,
                "right_landscape": landscape,
            }
            for animal in animals
        )

    for (left_landscape, right_landscape), combinations in MIXED_PLATES.items():
        plates.extend(
            {
                "animals": list(animals),
                "left_landscape": left_landscape,
                "right_landscape": right_landscape,
            }
            for animals in combinations
        )

    return {"base_plates": plates}


def validate(catalog: dict) -> None:
    valid_animals = {"GRIZZLY", "DEER", "SALMON", "HAWK", "FOX"}
    valid_landscapes = {"MOUNTAIN", "FOREST", "PLAINS", "WETLAND", "RIVER"}
    plates = catalog["base_plates"]

    assert len(PURE_PLATES) == 5
    assert all(len(animals) == 5 for animals in PURE_PLATES.values())
    assert len(MIXED_PLATES) == 10
    assert all(len(combinations) == 6 for combinations in MIXED_PLATES.values())
    assert len(plates) == 85  # 5 landscapes * 5, plus 10 pairs * 6.

    mixed_pair_counts = {}
    for plate in plates:
        assert set(plate["animals"]) <= valid_animals
        assert {plate["left_landscape"], plate["right_landscape"]} <= valid_landscapes
        if plate["left_landscape"] == plate["right_landscape"]:
            assert len(plate["animals"]) == 1
        else:
            assert len(plate["animals"]) in (2, 3)
            pair = frozenset((plate["left_landscape"], plate["right_landscape"]))
            mixed_pair_counts[pair] = mixed_pair_counts.get(pair, 0) + 1
    assert len(mixed_pair_counts) == 10
    assert all(count == 6 for count in mixed_pair_counts.values())


def main() -> None:
    catalog = make_catalog()
    validate(catalog)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog['base_plates'])} base plates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

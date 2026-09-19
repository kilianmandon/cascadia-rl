"""Generate the separate catalog of three-plate starting tiles as JSON.

Starter plates are intentionally defined independently from the main base-plate
pool: a starter plate may have an animal combination that is not in the pool.
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("starting_tiles.json")

# Animals that can appear on each landscape.  GRIZZLY is the enum spelling for
# the "Bear" used in the source list.
LANDSCAPE_ANIMALS = {
    "MOUNTAIN": {"HAWK", "DEER", "GRIZZLY"},
    "FOREST": {"DEER", "GRIZZLY", "FOX"},
    "PLAINS": {"SALMON", "DEER", "FOX"},
    "WETLAND": {"SALMON", "FOX", "HAWK"},
    "RIVER": {"HAWK", "SALMON", "GRIZZLY"},
}

# Each item is the three BasePlates belonging to one starting tile.
STARTING_TILES = [
    [
        ("WETLAND", "WETLAND", ("HAWK",)),
        ("FOREST", "RIVER", ("SALMON", "DEER", "HAWK")),
        ("MOUNTAIN", "PLAINS", ("GRIZZLY", "FOX")),
    ],
    [
        ("FOREST", "FOREST", ("DEER",)),
        ("RIVER", "MOUNTAIN", ("HAWK", "DEER", "GRIZZLY")),
        ("PLAINS", "WETLAND", ("FOX", "SALMON")),
    ],
    [
        ("PLAINS", "PLAINS", ("FOX",)),
        ("RIVER", "WETLAND", ("SALMON", "HAWK", "FOX")),
        ("MOUNTAIN", "FOREST", ("GRIZZLY", "DEER")),
    ],
    [
        ("MOUNTAIN", "MOUNTAIN", ("GRIZZLY",)),
        ("WETLAND", "FOREST", ("HAWK", "DEER", "FOX")),
        ("PLAINS", "RIVER", ("SALMON", "GRIZZLY")),
    ],
    [
        ("RIVER", "RIVER", ("SALMON",)),
        ("FOREST", "PLAINS", ("SALMON", "DEER", "GRIZZLY")),
        ("WETLAND", "MOUNTAIN", ("FOX", "HAWK")),
    ],
]


def make_catalog() -> dict:
    return {
        "starting_tiles": [
            {
                "id": f"START_{number}",
                "base_plates": [
                    {
                        "animals": list(animals),
                        "left_landscape": left_landscape,
                        "right_landscape": right_landscape,
                    }
                    for left_landscape, right_landscape, animals in base_plates
                ],
            }
            for number, base_plates in enumerate(STARTING_TILES, start=1)
        ]
    }


def validate(catalog: dict) -> None:
    tiles = catalog["starting_tiles"]
    assert len(tiles) == 5
    assert all(len(tile["base_plates"]) == 3 for tile in tiles)

    for tile in tiles:
        for plate in tile["base_plates"]:
            left = plate["left_landscape"]
            right = plate["right_landscape"]
            assert left in LANDSCAPE_ANIMALS and right in LANDSCAPE_ANIMALS
            assert len(plate["animals"]) in (1, 2, 3)
            valid_animals = LANDSCAPE_ANIMALS[left] | LANDSCAPE_ANIMALS[right]
            assert set(plate["animals"]) <= valid_animals
            if left == right:
                assert len(plate["animals"]) == 1


def main() -> None:
    catalog = make_catalog()
    validate(catalog)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog['starting_tiles'])} starting tiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

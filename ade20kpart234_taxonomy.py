"""
ADE20K-Part-234 taxonomy.

The semantic label order is obtained by flattening the official OV-PARTS
ADE20K-Part-234 parent-object -> parts mapping in insertion order.

Part labels:
    0 .. 233

Ignore / background in the released PNG masks:
    65535

Important:
    PART_CLASSES contains object-aware part names, e.g.
        "person's head"
        "car's wheel"
        "chair's leg"

    This keeps duplicated part concepts from different parent objects
    semantically distinct, as required by part semantic segmentation.
"""

from __future__ import annotations


IGNORE_LABEL = 65535
NUM_PARTS = 234
NUM_OBJECTS = 44


# ---------------------------------------------------------------------
# Official OV-PARTS order.
#
# Each tuple:
#   (
#       canonical parent-object name used by our pipeline,
#       official mapping key,
#       ordered part names,
#   )
# ---------------------------------------------------------------------

GROUP_SPECS = (
    (
        "person",
        "person, individual, someone, somebody, mortal, soul",
        (
            "head",
            "arm",
            "foot",
            "leg",
            "hand",
            "neck",
            "torso",
            "back",
            "gaze",
        ),
    ),
    (
        "door",
        "door",
        (
            "panel",
            "door frame",
            "handle",
            "knob",
        ),
    ),
    (
        "clock",
        "clock",
        (
            "face",
            "frame",
        ),
    ),
    (
        "toilet",
        "toilet, can, commode, crapper, pot, potty, stool, throne",
        (
            "bowl",
            "cistern",
            "lid",
        ),
    ),
    (
        "cabinet",
        "cabinet",
        (
            "door",
            "drawer",
            "front",
            "side",
            "skirt",
            "top",
            "shelf",
        ),
    ),
    (
        "sink",
        "sink",
        (
            "faucet",
            "tap",
            "bowl",
            "pedestal",
            "top",
        ),
    ),
    (
        "lamp",
        "lamp",
        (
            "canopy",
            "light source",
            "shade",
            "tube",
            "base",
            "arm",
            "cord",
            "column",
            "highlight",
        ),
    ),
    (
        "sconce",
        "sconce",
        (
            "arm",
            "light source",
            "shade",
            "backplate",
            "highlight",
        ),
    ),
    (
        "chair",
        "chair",
        (
            "leg",
            "seat",
            "arm",
            "stretcher",
            "base",
            "back",
            "skirt",
            "apron",
            "seat cushion",
        ),
    ),
    (
        "chest of drawers",
        "chest of drawers, chest, bureau, dresser",
        (
            "apron",
            "drawer",
            "front",
            "leg",
            "door",
        ),
    ),
    (
        "chandelier",
        "chandelier, pendant, pendent",
        (
            "canopy",
            "light source",
            "shade",
            "bulb",
            "arm",
            "chain",
            "cord",
            "highlight",
        ),
    ),
    (
        "bed",
        "bed",
        (
            "footboard",
            "headboard",
            "leg",
            "side rail",
        ),
    ),
    (
        "table",
        "table",
        (
            "drawer",
            "top",
            "apron",
            "leg",
            "shelf",
            "wheel",
        ),
    ),
    (
        "armchair",
        "armchair",
        (
            "arm",
            "back",
            "seat",
            "seat base",
            "seat cushion",
            "leg",
            "apron",
            "back pillow",
        ),
    ),
    (
        "ottoman",
        "ottoman, pouf, pouffe, puff, hassock",
        (
            "seat",
            "leg",
            "back",
        ),
    ),
    (
        "shelf",
        "shelf",
        (
            "door",
            "drawer",
            "front",
            "shelf",
        ),
    ),
    (
        "swivel chair",
        "swivel chair",
        (
            "back",
            "seat",
            "wheel",
            "base",
        ),
    ),
    (
        "fan",
        "fan",
        (
            "blade",
            "canopy",
            "tube",
        ),
    ),
    (
        "coffee table",
        "coffee table, cocktail table",
        (
            "top",
            "leg",
        ),
    ),
    (
        "stool",
        "stool",
        (
            "leg",
            "seat",
        ),
    ),
    (
        "sofa",
        "sofa, couch, lounge",
        (
            "arm",
            "seat base",
            "seat cushion",
            "back pillow",
            "leg",
            "back",
            "skirt",
        ),
    ),
    (
        "computer",
        (
            "computer, computing machine, computing device, data processor, "
            "electronic computer, information processing system"
        ),
        (
            "keyboard",
            "monitor",
            "computer case",
            "mouse",
        ),
    ),
    (
        "desk",
        "desk",
        (
            "leg",
            "top",
            "drawer",
            "apron",
            "door",
            "shelf",
        ),
    ),
    (
        "wardrobe",
        "wardrobe, closet, press",
        (
            "door",
            "drawer",
            "top",
            "front",
            "leg",
            "mirror",
        ),
    ),
    (
        "car",
        "car, auto, automobile, machine, motorcar",
        (
            "bumper",
            "door",
            "headlight",
            "license plate",
            "mirror",
            "wheel",
            "window",
            "hood",
            "logo",
            "wiper",
        ),
    ),
    (
        "bus",
        (
            "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
            "motorcoach, omnibus, passenger vehicle"
        ),
        (
            "headlight",
            "door",
            "license plate",
            "mirror",
            "window",
            "wiper",
            "wheel",
            "logo",
            "bumper",
        ),
    ),
    (
        "oven",
        "oven",
        (
            "button panel",
            "door",
            "drawer",
            "top",
        ),
    ),
    (
        "cooking stove",
        "cooking stove, stove, kitchen stove, range, kitchen range",
        (
            "burner",
            "stove",
            "oven",
            "button panel",
            "drawer",
            "door",
        ),
    ),
    (
        "microwave",
        "microwave, microwave oven",
        (
            "button panel",
            "front",
            "side",
            "top",
            "door",
            "window",
        ),
    ),
    (
        "refrigerator",
        "refrigerator, icebox",
        (
            "door",
            "side",
            "button panel",
            "drawer",
        ),
    ),
    (
        "kitchen island",
        "kitchen island",
        (
            "drawer",
            "side",
            "top",
            "door",
            "front",
        ),
    ),
    (
        "dishwasher",
        "dishwasher, dish washer, dishwashing machine",
        (
            "button panel",
            "skirt",
            "handle",
        ),
    ),
    (
        "bookcase",
        "bookcase",
        (
            "door",
            "front",
            "drawer",
            "side",
        ),
    ),
    (
        "television",
        (
            "television receiver, television, television set, tv, tv set, "
            "idiot box, boob tube, telly, goggle box"
        ),
        (
            "screen",
            "base",
            "frame",
            "keys",
            "speaker",
            "buttons",
        ),
    ),
    (
        "glass",
        "glass, drinking glass",
        (
            "base",
            "bowl",
            "opening",
            "stem",
        ),
    ),
    (
        "pool table",
        "pool table, billiard table, snooker table",
        (
            "leg",
            "bed",
            "pocket",
        ),
    ),
    (
        "van",
        "van",
        (
            "bumper",
            "door",
            "headlight",
            "taillight",
            "license plate",
            "mirror",
            "wheel",
            "window",
            "logo",
            "wiper",
        ),
    ),
    (
        "airplane",
        "airplane, aeroplane, plane",
        (
            "fuselage",
            "stabilizer",
            "wing",
            "landing gear",
            "turbine engine",
            "propeller",
            "door",
        ),
    ),
    (
        "truck",
        "truck, motortruck",
        (
            "door",
            "headlight",
            "license plate",
            "wheel",
            "window",
            "mirror",
            "bumper",
            "logo",
        ),
    ),
    (
        "minibike",
        "minibike, motorbike",
        (
            "mirror",
            "wheel",
            "license plate",
            "seat",
        ),
    ),
    (
        "washer",
        "washer, automatic washer, washing machine",
        (
            "door",
            "button panel",
            "front",
            "side",
        ),
    ),
    (
        "bench",
        "bench",
        (
            "leg",
            "seat",
            "arm",
            "back",
        ),
    ),
    (
        "traffic light",
        "traffic light",
        (
            "housing",
            "pole",
        ),
    ),
    (
        "light",
        "light",
        (
            "shade",
            "highlight",
            "light source",
            "aperture",
            "canopy",
            "diffusor",
        ),
    ),
)


# ---------------------------------------------------------------------
# Flatten taxonomy.
# ---------------------------------------------------------------------

OBJECT_CLASSES = tuple(
    spec[0]
    for spec in GROUP_SPECS
)

OBJECT_OFFICIAL_NAMES = tuple(
    spec[1]
    for spec in GROUP_SPECS
)

OBJECT_NAME_TO_ID = {
    name: i
    for i, name in enumerate(OBJECT_CLASSES)
}


PART_CLASSES = []
PART_RAW_NAMES = []
OBJECT_GROUPS = {}
PART_TO_OBJECT = {}
PART_TO_OBJECT_NAME = {}


_pid = 0

for object_id, (
    object_name,
    official_name,
    part_names,
) in enumerate(GROUP_SPECS):

    ids = []

    for part_name in part_names:

        pid = _pid
        _pid += 1

        # Object-aware text class.
        class_name = f"{object_name}'s {part_name}"

        PART_CLASSES.append(class_name)
        PART_RAW_NAMES.append(part_name)

        ids.append(pid)

        PART_TO_OBJECT[pid] = object_id
        PART_TO_OBJECT_NAME[pid] = object_name

    OBJECT_GROUPS[object_name] = ids


PART_CLASSES = tuple(PART_CLASSES)
PART_RAW_NAMES = tuple(PART_RAW_NAMES)


# ---------------------------------------------------------------------
# Strong contract checks.
# ---------------------------------------------------------------------

assert len(GROUP_SPECS) == NUM_OBJECTS
assert len(OBJECT_CLASSES) == NUM_OBJECTS

assert len(PART_CLASSES) == NUM_PARTS
assert len(PART_RAW_NAMES) == NUM_PARTS

assert sorted(PART_TO_OBJECT) == list(range(NUM_PARTS))

_flat = []

for object_name in OBJECT_CLASSES:
    ids = OBJECT_GROUPS[object_name]

    assert len(ids) > 0
    assert ids == list(
        range(ids[0], ids[0] + len(ids))
    )

    _flat.extend(ids)

assert _flat == list(range(NUM_PARTS))
assert len(_flat) == len(set(_flat))


# Known flatten-order anchors.
assert PART_CLASSES[0] == "person's head"
assert PART_CLASSES[8] == "person's gaze"

assert PART_CLASSES[9] == "door's panel"
assert PART_CLASSES[11] == "door's handle"

assert PART_CLASSES[125] == "car's bumper"
assert PART_CLASSES[134] == "car's wiper"

assert PART_CLASSES[189] == "van's bumper"
assert PART_CLASSES[193] == "van's license plate"

assert PART_CLASSES[199] == "airplane's fuselage"

assert PART_CLASSES[233] == "light's diffusor"


if __name__ == "__main__":

    print("objects:", len(OBJECT_CLASSES))
    print("parts  :", len(PART_CLASSES))

    print()

    for oid, name in enumerate(OBJECT_CLASSES):
        ids = OBJECT_GROUPS[name]

        print(
            f"{oid:02d} "
            f"{name:20s} "
            f"{ids[0]:3d}..{ids[-1]:3d} "
            f"n={len(ids)}"
        )

    print()
    print("11 :", PART_CLASSES[11])
    print("193:", PART_CLASSES[193])

    print()
    print("ADE20K-PART-234 TAXONOMY: PASS")

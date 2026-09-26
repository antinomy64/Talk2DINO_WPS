PART_CLASSES = [
    "Quadruped Head",      # 0
    "Quadruped Body",      # 1
    "Quadruped Foot",      # 2
    "Quadruped Tail",      # 3

    "Biped Head",          # 4
    "Biped Body",          # 5
    "Biped Hand",          # 6
    "Biped Foot",          # 7
    "Biped Tail",          # 8

    "Fish Head",           # 9
    "Fish Body",           # 10
    "Fish Fin",            # 11
    "Fish Tail",           # 12

    "Bird Head",           # 13
    "Bird Body",           # 14
    "Bird Wing",           # 15
    "Bird Foot",           # 16
    "Bird Tail",           # 17

    "Snake Head",          # 18
    "Snake Body",          # 19

    "Reptile Head",        # 20
    "Reptile Body",        # 21
    "Reptile Foot",        # 22
    "Reptile Tail",        # 23

    "Car Body",            # 24
    "Car Tire",            # 25
    "Car Side Mirror",     # 26

    "Bicycle Body",        # 27
    "Bicycle Head",        # 28
    "Bicycle Seat",        # 29
    "Bicycle Tire",        # 30

    "Boat Body",           # 31
    "Boat Sail",           # 32

    "Aeroplane Head",      # 33
    "Aeroplane Body",      # 34
    "Aeroplane Engine",    # 35
    "Aeroplane Wing",      # 36
    "Aeroplane Tail",      # 37

    "Bottle Mouth",        # 38
    "Bottle Body",         # 39
]

OBJECT_GROUPS = {
    "quadruped": [0, 1, 2, 3],
    "biped": [4, 5, 6, 7, 8],
    "fish": [9, 10, 11, 12],
    "bird": [13, 14, 15, 16, 17],
    "snake": [18, 19],
    "reptile": [20, 21, 22, 23],
    "car": [24, 25, 26],
    "bicycle": [27, 28, 29, 30],
    "boat": [31, 32],
    "aeroplane": [33, 34, 35, 36, 37],
    "bottle": [38, 39],
}

OBJECT_CLASSES = list(OBJECT_GROUPS.keys())

PART_TO_OBJECT = {}
for obj_id, (_, part_ids) in enumerate(OBJECT_GROUPS.items()):
    for p in part_ids:
        assert p not in PART_TO_OBJECT
        PART_TO_OBJECT[p] = obj_id

assert len(PART_CLASSES) == 40
assert len(OBJECT_GROUPS) == 11
assert set(PART_TO_OBJECT.keys()) == set(range(40))

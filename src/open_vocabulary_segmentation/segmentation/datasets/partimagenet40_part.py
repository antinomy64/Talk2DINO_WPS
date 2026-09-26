from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


PART_CLASSES = (
    "Quadruped Head",
    "Quadruped Body",
    "Quadruped Foot",
    "Quadruped Tail",

    "Biped Head",
    "Biped Body",
    "Biped Hand",
    "Biped Foot",
    "Biped Tail",

    "Fish Head",
    "Fish Body",
    "Fish Fin",
    "Fish Tail",

    "Bird Head",
    "Bird Body",
    "Bird Wing",
    "Bird Foot",
    "Bird Tail",

    "Snake Head",
    "Snake Body",

    "Reptile Head",
    "Reptile Body",
    "Reptile Foot",
    "Reptile Tail",

    "Car Body",
    "Car Tire",
    "Car Side Mirror",

    "Bicycle Body",
    "Bicycle Head",
    "Bicycle Seat",
    "Bicycle Tire",

    "Boat Body",
    "Boat Sail",

    "Aeroplane Head",
    "Aeroplane Body",
    "Aeroplane Engine",
    "Aeroplane Wing",
    "Aeroplane Tail",

    "Bottle Mouth",
    "Bottle Body",
)


def _palette(n):
    return [
        [
            (37 * i) % 256,
            (67 * i) % 256,
            (97 * i) % 256,
        ]
        for i in range(n)
    ]


@DATASETS.register_module()
class PartImageNet40PartDataset(CustomDataset):
    """
    PartImageNet 40-way semantic part evaluation.

    GT protocol:
      0..39 : valid semantic part IDs
      255   : ignore/background

    Therefore class 0 is VALID and reduce_zero_label must stay False.
    """

    CLASSES = PART_CLASSES
    PALETTE = _palette(len(PART_CLASSES))

    def __init__(self, **kwargs):
        kwargs.setdefault("img_suffix", ".JPEG")
        kwargs.setdefault("seg_map_suffix", ".png")
        kwargs.setdefault("reduce_zero_label", False)
        kwargs.setdefault("ignore_index", 255)

        super().__init__(**kwargs)

        assert len(self.CLASSES) == 40
        assert self.reduce_zero_label is False
        assert self.ignore_index == 255

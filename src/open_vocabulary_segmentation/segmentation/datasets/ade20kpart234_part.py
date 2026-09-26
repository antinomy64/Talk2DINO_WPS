#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from mmseg.datasets import DATASETS, CustomDataset

# 0..233 valid semantic parts; 65535 background/unlabeled.
PART_CLASSES = ("person's head", "person's arm", "person's foot", "person's leg", "person's hand", "person's neck", "person's torso", "person's back", "person's gaze", "door's panel", "door's door frame", "door's handle", "door's knob", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's side", "cabinet's skirt", "cabinet's top", "cabinet's shelf", "sink's faucet", "sink's tap", "sink's bowl", "sink's pedestal", "sink's top", "lamp's canopy", "lamp's light source", "lamp's shade", "lamp's tube", "lamp's base", "lamp's arm", "lamp's cord", "lamp's column", "lamp's highlight", "sconce's arm", "sconce's light source", "sconce's shade", "sconce's backplate", "sconce's highlight", "chair's leg", "chair's seat", "chair's arm", "chair's stretcher", "chair's base", "chair's back", "chair's skirt", "chair's apron", "chair's seat cushion", "chest of drawers's apron", "chest of drawers's drawer", "chest of drawers's front", "chest of drawers's leg", "chest of drawers's door", "chandelier's canopy", "chandelier's light source", "chandelier's shade", "chandelier's bulb", "chandelier's arm", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's drawer", "table's top", "table's apron", "table's leg", "table's shelf", "table's wheel", "armchair's arm", "armchair's back", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "armchair's leg", "armchair's apron", "armchair's back pillow", "ottoman's seat", "ottoman's leg", "ottoman's back", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", "swivel chair's back", "swivel chair's seat", "swivel chair's wheel", "swivel chair's base", "fan's blade", "fan's canopy", "fan's tube", "coffee table's top", "coffee table's leg", "stool's leg", "stool's seat", "sofa's arm", "sofa's seat base", "sofa's seat cushion", "sofa's back pillow", "sofa's leg", "sofa's back", "sofa's skirt", "computer's keyboard", "computer's monitor", "computer's computer case", "computer's mouse", "desk's leg", "desk's top", "desk's drawer", "desk's apron", "desk's door", "desk's shelf", "wardrobe's door", "wardrobe's drawer", "wardrobe's top", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "car's bumper", "car's door", "car's headlight", "car's license plate", "car's mirror", "car's wheel", "car's window", "car's hood", "car's logo", "car's wiper", "bus's headlight", "bus's door", "bus's license plate", "bus's mirror", "bus's window", "bus's wiper", "bus's wheel", "bus's logo", "bus's bumper", "oven's button panel", "oven's door", "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's stove", "cooking stove's oven", "cooking stove's button panel", "cooking stove's drawer", "cooking stove's door", "microwave's button panel", "microwave's front", "microwave's side", "microwave's top", "microwave's door", "microwave's window", "refrigerator's door", "refrigerator's side", "refrigerator's button panel", "refrigerator's drawer", "kitchen island's drawer", "kitchen island's side", "kitchen island's top", "kitchen island's door", "kitchen island's front", "dishwasher's button panel", "dishwasher's skirt", "dishwasher's handle", "bookcase's door", "bookcase's front", "bookcase's drawer", "bookcase's side", "television's screen", "television's base", "television's frame", "television's keys", "television's speaker", "television's buttons", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's leg", "pool table's bed", "pool table's pocket", "van's bumper", "van's door", "van's headlight", "van's taillight", "van's license plate", "van's mirror", "van's wheel", "van's window", "van's logo", "van's wiper", "airplane's fuselage", "airplane's stabilizer", "airplane's wing", "airplane's landing gear", "airplane's turbine engine", "airplane's propeller", "airplane's door", "truck's door", "truck's headlight", "truck's license plate", "truck's wheel", "truck's window", "truck's mirror", "truck's bumper", "truck's logo", "minibike's mirror", "minibike's wheel", "minibike's license plate", "minibike's seat", "washer's door", "washer's button panel", "washer's front", "washer's side", "bench's leg", "bench's seat", "bench's arm", "bench's back", "traffic light's housing", "traffic light's pole", "light's shade", "light's highlight", "light's light source", "light's aperture", "light's canopy", "light's diffusor")


def _voc_palette(n: int):
    palette = []
    for j in range(n):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
        palette.append([r, g, b])
    return palette


@DATASETS.register_module(force=True)
class ADE20KPart234Dataset(CustomDataset):
    CLASSES = PART_CLASSES
    PALETTE = _voc_palette(len(PART_CLASSES))

    def __init__(self, **kwargs):
        requested_ignore = int(kwargs.pop("ignore_index", 65535))
        requested_reduce = bool(kwargs.pop("reduce_zero_label", False))

        if requested_ignore != 65535:
            raise ValueError(
                f"ADE20K-Part-234 requires ignore_index=65535, "
                f"got {requested_ignore}"
            )
        if requested_reduce:
            raise ValueError(
                "ADE20K-Part-234 class 0 is valid; reduce_zero_label must be False"
            )

        super().__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            ignore_index=65535,
            reduce_zero_label=False,
            **kwargs,
        )

        assert len(self.CLASSES) == 234
        assert self.CLASSES[0] == "person's head"
        assert self.ignore_index == 65535
        assert self.reduce_zero_label is False

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        if self.ann_dir is None or not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(self.ann_dir)

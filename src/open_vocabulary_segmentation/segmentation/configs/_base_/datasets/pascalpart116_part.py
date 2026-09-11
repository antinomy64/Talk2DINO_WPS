# Pascal-Part-116 fine part evaluation.
#
# Critical label semantics:
#   0..115 = valid semantic part IDs
#   255    = background/ignore
#
# Therefore:
#   ignore_index=255
#   reduce_zero_label=False
#
# Do NOT use reduce_zero_label=True: it would erase real part class 0 and
# shift all remaining part IDs by -1.

custom_imports = dict(
    imports=["segmentation.datasets.pascalpart116_part"],
    allow_failed_imports=False,
)

dataset_type = "PascalPart116PartDataset"
data_root = "./data/PascalPart116"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="FloatImage"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="annotations_detectron2_part/val",
        split="val.txt",
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=255,
        reduce_zero_label=False,
    )
)

# Match the official Talk2DINO ViT-B Pascal-VOC inference protocol.
test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))

# ADE20K-Part-234 full-image semantic part evaluation.
# 0..233 valid semantic parts; 65535 ignore; no label shift.

custom_imports = dict(
    imports=["segmentation.datasets.ade20kpart234_part"],
    allow_failed_imports=False,
)

dataset_type = "ADE20KPart234Dataset"
data_root = "/home/master/dataset/lyx/ADE20KPart234"

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
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=65535,
        reduce_zero_label=False,
    )
)

test_cfg = dict(
    mode="slide",
    stride=(224, 224),
    crop_size=(448, 448),
)

custom_imports = dict(
    imports=["segmentation.datasets.partimagenet40_part"],
    allow_failed_imports=False,
)

dataset_type = "PartImageNet40PartDataset"

data_root = (
    "/home/master/dataset/lyx/PartImageNet/PartImageNet"
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="annotations_val_ignore255",
        pipeline=test_pipeline,
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="annotations_val_ignore255",
        pipeline=test_pipeline,
    ),
)

# MMSeg inference configuration consumed by Talk2DINO's
# build_dinotext_seg_inference().
test_cfg = dict(
    mode="slide",
    crop_size=(448, 448),
    stride=(224, 224),
)

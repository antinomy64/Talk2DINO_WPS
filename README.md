# Talk2DINO-WPS: RelProto + Orthogonal Part Alignment

This repository implements a weakly-supervised part semantic segmentation pipeline built on top of Talk2DINO.

The final method does **not** use the PartStruct projector fine-tuning stage. Instead, it keeps the object-level Talk2DINO projector frozen and learns a single shared orthogonal transformation \(W\) over projected part-text prototypes. The learned transformation is optimized using image-specific visual RelProto targets induced from predicted object foreground regions.

## Final Pipeline

```text
COCO image-caption supervision
        |
        v
Talk2DINO object-level CLIP -> DINO projector
        |
        |  frozen
        v
raw CLIP part text bank [116, 512]
        |
        v
frozen object-level projector
        |
        v
base part text bank T0 [116, 768]
        |
        |                    Pascal-Part training images
        |                              |
        |                              v
        |                    frozen Talk2DINO object prediction
        |                              |
        |                              v
        |                    predicted object foreground + crop
        |                              |
        |                              v
        |                    DINOv2 crop patch tokens [1024, 768]
        |                              |
        +---------------> RelProto induction <---------------+
                               |
                               v
                    learn one shared orthogonal W
                               |
                               v
                     aligned part text prototypes
                               |
                               v
                     bake W into projector
                               |
                               v
                    PascalPart116 dense evaluation
```

The core trainable component of the final method is the shared orthogonal matrix

\[
W \in \mathbb{R}^{768\times768}, \qquad W^\top W = I.
\]

The original Talk2DINO projector is frozen throughout RelProto alignment.

---

## Supervision Protocol

The final RelProto stage uses only **non-spatial semantic supervision**:

- image-level present object labels;
- per-object present part labels.

The current preprocessing script can derive these weak labels from Pascal-Part object/part masks for reproducibility. Importantly, the masks are used only to recover semantic presence labels. Their spatial information is discarded before predicted-object segmentation, cropping, RelProto induction, and \(W\) optimization.

All spatial support used by the final RelProto stage comes from the frozen Talk2DINO prediction:

- predicted semantic object mask;
- predicted object crop;
- predicted foreground patch mask;
- DINOv2 crop patch tokens.

Therefore the final method does **not** train with GT part masks, GT part boxes, GT part points, SAM masks, or other part-level spatial supervision.

---

## 1. Frozen Object-Level Projector

We use the original Talk2DINO-style object-level CLIP-to-DINO projector trained from image-caption supervision.

The final mainline projector is:

```text
weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth
```

This projector is used directly as the frozen base projector.

**PartStruct fine-tuning is not part of the final pipeline.**

For all commands below, explicitly pass this checkpoint instead of relying on legacy script defaults.

---

## 2. Part Text Bank

PascalPart116 part names are encoded with CLIP ViT-B/16 using the Talk2DINO `sub_imagenet_template` prompt set.

The text bank is saved **before** the Talk2DINO projector and is intentionally not L2-normalized:

```text
raw CLIP prompt mean: [116, 512]
normalized: False
stage: pre_projector_prompt_mean
```

Generate it with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python extract_clip_text_bank.py \
  --classes_source \
    src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py:PART_CLASSES \
  --out_path \
    feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt \
  --clip_model ViT-B/16 \
  --template_set sub_imagenet_template \
  --device cuda
```

During RelProto training, the frozen projector produces

\[
T_0
=
\operatorname{Norm}
\left(
P_{\mathrm{obj}}
\left(
E_{\mathrm{CLIP}}(\text{part})
\right)
\right)
\in
\mathbb{R}^{116\times768}.
\]

No PartStruct projector fine-tuning is performed.

---

## 3. Predicted-Object Crop Cache

`extract_predobj_cropaug.py` converts weak semantic labels into the spatial inputs used by RelProto.

For each training image:

1. recover image-level present object labels and per-object present part labels;
2. run the **frozen initial Talk2DINO projector** over the present object classes;
3. obtain predicted semantic object masks with background competition and PAMR;
4. build a square object crop from the predicted mask;
5. expand the crop by `1.2`;
6. resize the RGB crop to `448 x 448`;
7. extract DINOv2 ViT-B/14-reg patch tokens;
8. rasterize the predicted object mask onto the `32 x 32` patch grid.

Each valid object annotation stores:

```text
part_category_id      weak part-presence labels
part_class_name       semantic part names
cropaug_box_xyxy      crop from predicted object mask
pred_obj_mask_patch   predicted foreground mask [1024]
cropaug_patch_tokens  DINOv2 crop tokens [1024, 768]
```

Mainline cache command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python extract_predobj_cropaug.py \
  --repo_root . \
  --image_root data/PascalPart116/images/train \
  --obj_mask_root data/PascalPart116/annotations_detectron2_obj/train \
  --part_mask_root data/PascalPart116/annotations_detectron2_part/train \
  --projector_weight \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --bg_thresh 0.54 \
  --lambda_bg 0.2 \
  --pamr \
  --output_pth \
    feature/pascalpart116_predobj_clip_struct_initial/train_predobj_cropaug_initial_bg054.pth \
  --device cuda
```

The cache stores the SHA256 of the projector used to produce it. RelProto training can enforce that the cache and the frozen projector are identical.

---

## 4. RelProto Induction

For one object annotation, let

- \(C\) be the set of present part IDs;
- \(X \in \mathbb{R}^{P\times768}\) be unit DINO patch features;
- \(\Omega\) be the predicted foreground patch set;
- \(W\) be the shared trainable transform.

Current text queries are

\[
\hat T
=
\operatorname{Norm}
\left(
T_0[C]W
\right).
\]

### Relative evidence

For part \(c\) and foreground patch \(p\),

\[
S_{c,p}
=
\hat t_c^\top x_p.
\]

When more than one part is present,

\[
R_{c,p}
=
S_{c,p}
-
\max_{q\neq c}S_{q,p}.
\]

For a single-part object, \(R=S\).

This relative score prefers patches that support one semantic part more strongly than the other parts known to be present in the same object.

### Anchor selection

Each part independently selects its strongest foreground patch:

\[
a_c
=
\arg\max_{p\in\Omega} R_{c,p}.
\]

Anchor collisions are allowed. Different parts are not forced into a global one-to-one assignment.

### Prototype support

For every part:

1. always include its own anchor;
2. exclude only its own anchor from its supplementary candidates;
3. add the highest-scoring patches with **strictly positive** relative margin;
4. stop at `prototype_max_patches`;
5. do not force-fill invalid support.

The main setting is

```text
prototype_max_patches = 4
```

The image-specific visual RelProto is the normalized equal mean of its selected unit DINO patch features:

\[
v_c
=
\operatorname{Norm}
\left(
\frac{1}{|\mathcal P_c|}
\sum_{p\in\mathcal P_c}x_p
\right).
\]

The induced visual prototypes are detached before optimizing \(W\).

---

## 5. Shared Orthogonal Alignment

The only optimized parameter in the final RelProto stage is one global matrix

\[
W\in\mathbb{R}^{768\times768}.
\]

The same \(W\) is shared across:

- all 116 semantic parts;
- all object categories;
- all training images.

For each annotation, text row \(c\) directly matches the RelProto induced for the same semantic row:

\[
\mathcal L_{\mathrm{ann}}
=
\frac{1}{|C|}
\sum_{c\in C}
\left(
1-\cos(\hat t_c,v_c)
\right).
\]

The optimizer batch loss is the equal mean over annotations.

Main optimization setting:

```text
optimizer = Adam
lr = 1e-3
batch_size = 16
epochs = 10
W initialization = identity
weight_decay = 0
```

After every Adam step, \(W\) is retracted to the orthogonal group with SVD:

\[
W = U\Sigma V^\top
\quad\Longrightarrow\quad
W \leftarrow UV^\top.
\]

This gives

\[
W^\top W \approx I.
\]

Because \(W\) is orthogonal, it preserves the pairwise cosine geometry of the normalized text bank:

\[
(t_iW)^\top(t_jW)
=
t_i^\top t_j.
\]

Thus \(W\) can globally rotate/refelect the language-derived part structure toward the visual representation without freely distorting the relational geometry among part text prototypes.

---

## 6. Train \(W\)

The current training entry point in this repository is:

```text
train_relproto_alignemt.py
```

Note the filename spelling `alignemt`.

Run the final mainline setting with the **initial, non-PartStruct projector**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python train_relproto_alignemt.py \
  --project_root . \
  --train_dataset \
    feature/pascalpart116_predobj_clip_struct_initial/train_predobj_cropaug_initial_bg054.pth \
  --text_bank \
    feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt \
  --model_config configs/vitb_mlp_infonce.yaml \
  --weights \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --out_dir \
    final_exp/predobj_clip_w_initial/relproto4_orth_seed123 \
  --prototype_max_patches 4 \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --seed 123 \
  --expected_bg_thresh 0.54 \
  --require_pamr \
  --require_cache_projector_match \
  --device cuda
```

Typical outputs include:

```text
W_epoch_001.pt
...
W_epoch_010.pt
W_last.pt
training_history.csv
training_samples.csv
preflight.json
text_bank.json
projector.json
summary.json
```

The W checkpoint records the source-projector SHA and cache metadata for reproducibility.

---

## 7. Bake \(W\) into the Projector

For normal Talk2DINO dense evaluation, the learned right-side transform can be folded into the final affine layer of the projector.

The training relation is

\[
\operatorname{Norm}
\left(
\operatorname{Norm}(P_{\mathrm{obj}}(x))W
\right).
\]

For the current two-layer projector, `bake_predobj_relproto_w_into_projector_final.py` folds \(W\) into `hidden_layers.0` and verifies numerical equivalence on all 116 real CLIP part features.

Example:

```bash
python bake_predobj_relproto_w_into_projector_final.py \
  --project_root . \
  --projector \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --w_checkpoint \
    final_exp/predobj_clip_w_initial/relproto4_orth_seed123/W_last.pt \
  --text_bank \
    feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt \
  --model_config configs/vitb_mlp_infonce.yaml \
  --output \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean_relproto4_orth_s123.pth \
  --device cuda
```

The script also writes a `.bake_report.json` containing the source SHA, W SHA, orthogonality error, and pre/post-save equivalence checks.

---

## 8. PascalPart116 Dense Evaluation

After baking \(W\), evaluate with the standard PascalPart116 full-image segmentation pipeline.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=30101 \
  src/open_vocabulary_segmentation/main.py \
    --eval \
    --output output/voc116_relproto4_orth_s123 \
    --eval_cfg \
      src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml \
    --eval_base_cfg \
      src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml \
    --opts \
      model.proj_name=vitb_mlp_infonce_coco2014_reproduce_clean_relproto4_orth_s123 \
      evaluate.pamr=false \
      evaluate.bg_thresh=0.4
```

The reported `voc116_part` value is the PascalPart116 part mIoU used by the current experiments.

---

## 9. Why Orthogonal \(W\)?

A useful controlled ablation replaces the orthogonal transform by an unconstrained shared `768 x 768` linear matrix while keeping:

- the same frozen initial projector;
- the same predicted-object cache;
- the same RelProto induction;
- the same identity initialization;
- the same parameter count;
- the same optimizer and learning rate;
- the same seeds;
- the same dense evaluation.

The only removed operation is the SVD/polar retraction.

This isolates whether preserving text-side relational geometry is useful, rather than merely testing whether an additional trainable linear layer helps.

---

## What Is Not Part of the Final Method

The repository contains several historical and diagnostic experiments. They are **not** part of the final mainline pipeline:

- `train_partstruct_ft.py`: historical PartStruct projector fine-tuning;
- LLaMA3 text-bank / teacher experiments;
- `train_textstruct_zoo.py`: text-structure screening experiments;
- `train_gtproto_oracle.py`: GT visual-prototype oracle;
- QAP experiments;
- unconstrained Linear transform: ablation only;
- GT part spatial masks for RelProto support: not used by the final W stage.

The final method is:

```text
frozen Talk2DINO object-level projector
+ weak semantic part-presence labels
+ predicted object foreground
+ RelProto induction
+ one shared orthogonal W
```

---

## Important Files

```text
extract_clip_text_bank.py
    Build the raw [116,512] CLIP part text bank.

extract_predobj_cropaug.py
    Convert weak semantic labels into predicted-object crop/foreground
    and DINOv2 patch-token caches.

train_relproto_alignemt.py
    Main final RelProto + orthogonal-W trainer.

bake_predobj_relproto_w_into_projector_final.py
    Exact W-to-projector bake with SHA and numerical-equivalence checks.

export_aligned_part_bank.py
    Export an aligned [116,768] part bank directly from a projected bank + W.

src/open_vocabulary_segmentation/
    Talk2DINO dense segmentation/evaluation implementation.

train_partstruct_ft.py
    Historical/ablation code only; not part of the final method.
```

---

## Self Tests

```bash
python extract_predobj_cropaug.py --self_test
python train_relproto_alignemt.py --self_test
```

Before a long run, it is also recommended to use the trainer preflight:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_relproto_alignemt.py \
  --project_root . \
  --train_dataset \
    feature/pascalpart116_predobj_clip_struct_initial/train_predobj_cropaug_initial_bg054.pth \
  --text_bank \
    feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt \
  --model_config configs/vitb_mlp_infonce.yaml \
  --weights \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --expected_bg_thresh 0.54 \
  --require_pamr \
  --require_cache_projector_match \
  --preflight_only \
  --device cuda
```

---

## Method Summary

The final method does not attempt to learn part correspondence by independently fitting each semantic part, nor does it fine-tune the Talk2DINO projector with a second PartStruct stage.

Instead, it keeps the object-level cross-modal projector frozen and learns a **single shared orthogonal transformation \(W\)**. Image-specific RelProto targets are induced from predicted object foregrounds using relative part evidence. The orthogonality constraint lets the learned transform adapt the global orientation of part-text prototypes toward the visual DINO space while preserving their pairwise language-side relational structure.

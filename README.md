# Clean RelProto + Orthogonal Alignment

This package is the cleaned implementation of the final No-QAP RelProto stage.
It does not import or monkey-patch the old QAP experiment code.

## Method

For each training annotation with present part IDs C and an allowed foreground
support Omega:

1. Current semantic queries: `T_hat = normalize(T[C] @ W)`.
2. Patch cosine scores: `S = T_hat @ X.T`.
3. Relative margin: `R[c,p] = S[c,p] - max_{q != c} S[q,p]`; K=1 uses R=S.
4. Select one globally-greedy distinct anchor for every present part.
5. Reserve all anchors, then add up to `max_patches-1` non-anchor patches with
   strictly positive relative margin. The cap includes the anchor.
6. Normalize the equal mean of unit patch features to obtain an image-specific
   visual prototype. Prototype targets are detached.
7. Minimize per-annotation mean `1-cos(T_hat, prototype)`, then mean over
   annotations in the optimizer batch.
8. Take one Adam step and project W back to O(768) by SVD `U @ Vh`.

The base text bank and the projector producing it are frozen. One global W is
shared across all images and parts.

## Layout

- `train_relproto_alignment.py`: training entry point.
- `src/part_alignment/relproto.py`: relative-margin prototype induction.
- `src/part_alignment/alignment.py`: orthogonal text-space alignment and loss.
- `src/part_alignment/data.py`: training input boundary (presence + foreground).
- `src/part_alignment/text_bank.py`: fixed 116-part bank loading/projecting.
- `configs/part_alignment/vitb_relproto_orth.yaml`: main no-GT-spatial template.
- `configs/part_alignment/vitb_relproto_orth_oracle_regression.yaml`: historical
  regression template only.
- `bake_relproto_w_into_projector.py`: folds right-side W into the final text
  projection Linear for normal Talk2DINO evaluation.
- `export_aligned_part_bank.py`: alternatively exports an aligned [116,768] bank.

## Install into a Talk2DINO checkout

Copy the files while preserving the directory structure. Do not replace the
existing `src/model.py` or old `final_exp/v9` files.

## Self tests

```bash
python train_relproto_alignment.py --self_test
python bake_relproto_w_into_projector.py --self_test
```

## Main no-GT-spatial run

The PTH must contain the image-level presence key configured under
`data.presence.key` and a predicted foreground mask configured under
`data.foreground.key`. The predicted mask must be aligned to the same crop/token
grid as `cropaug_patch_tokens`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python train_relproto_alignment.py \
  --config configs/part_alignment/vitb_relproto_orth.yaml \
  --device cuda
```

## Historical regression run

The oracle config is only for checking that the refactor preserves the old
mathematics. It intentionally derives visibility/object support from GT.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python train_relproto_alignment.py \
  --config configs/part_alignment/vitb_relproto_orth_oracle_regression.yaml \
  --device cuda
```

## Bake W

```bash
python bake_relproto_w_into_projector.py \
  --projector weights/<source_projector>.pth \
  --w_checkpoint outputs/relproto_alignment/vitb_relproto_orth/W_last.pt \
  --output weights/<source_projector>_relproto_orth.pth
```

The bake implements `projected_text_new = projected_text_old @ W` exactly by
left-multiplying the final Linear weight with `W.T` and right-multiplying its
bias with W.

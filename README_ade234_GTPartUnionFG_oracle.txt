ADE234 PRIVILEGED FOREGROUND ORACLE FROM GT PART UNION

The ADE20KPart234 folder has pixel-level part-GT masks but no independent object-GT masks.
For each crop belonging to object category O, convert GT Part mask into a binary
foreground by unioning ALL ADE234 part class IDs with taxonomy parent O.
THIS USES PART-LEVEL SPATIAL GT, though only a binary union is supplied to training.
It is NOT comparable to independent object-GT masks, nor a weakly supervised result.
Does not change original crop boxes, DINO features, part-presence labels or trainer losses.

File placements: all 3 scripts in Talk2DINO_official_bg repo root, alongside
build_ade234_gtobj_valid_masks.py and train_relproto_soft_structure_ade234.py.

cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
python build_ade234_gtpartunion_valid_masks.py --self-test
python train_ade234_gtpartunion_fg_oracle.py --self-test-oracle
export ADE_GT_PART_ROOT="/home/master/dataset/lyx/ADE20KPart234/annotations_detectron2_part/train"
export ADE_PART_ID_OFFSET=0  # inspect actual GT encoding; if labels are 1..234 use 1
bash run_ade234_gtpartunion_fg_oracle.sh preflight
bash run_ade234_gtpartunion_fg_oracle.sh build
bash run_ade234_gtpartunion_fg_oracle.sh train
bash run_ade234_gtpartunion_fg_oracle.sh bake
bash run_ade234_gtpartunion_fg_oracle.sh eval

Default empty-mask behavior: fail. If nonempty masks verified and a few are genuinely empty,
explicitly export ADE_EMPTY_POLICY=drop BEFORE BUILD, record crop-count changes.

The baseline to compare against MUST be ADE predicted-foreground 100ep with
identical train_relproto_soft_structure_ade234.py, including K=1 hard top-L=8.

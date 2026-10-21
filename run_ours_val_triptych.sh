#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
if [[ "${1:-}" != pp116 && "${1:-}" != ade234 ]]; then
  echo 'Usage: bash run_ours_val_triptych.sh pp116|ade234' >&2;exit 2
fi
DS="$1"
if [[ "$DS" == pp116 ]];then
  export OVPS_VIS_EXPECTED_MIOU=36.75 OVPS_VIS_EXPECTED_COUNT=850
  ECFG=src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml
  BCFG=src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml
else
  export OVPS_VIS_EXPECTED_MIOU=8.98 OVPS_VIS_EXPECTED_COUNT=1016
  ECFG=src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml
  BCFG=src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml
fi
export OVPS_VIS_DATASET="$DS"
export OVPS_VIS_OUTPUT="$(pwd)/output/ours_e100_val_triptych_${DS}"
export OVPS_VIS_ALPHA="${OVPS_VIS_ALPHA:-1.0}"
if [[ -d "$OVPS_VIS_OUTPUT" ]] && [[ -n "$(ls -A "$OVPS_VIS_OUTPUT")" ]];then
  echo "[REFUSE TO OVERWRITE] $OVPS_VIS_OUTPUT" >&2; exit 3
fi
# Examine original evaluation result JSONs; accept only checkpoint with matching metric.
PROJ="$(python - "$DS" "$OVPS_VIS_EXPECTED_MIOU" "${OVPS_VIS_PROJ_NAME:-}" <<'PY'
import json, math, sys
from pathlib import Path
DS=sys.argv[1]; expected=float(sys.argv[2]); manual=sys.argv[3]; found=[]
for p in Path('segmentation_results').glob('*.json'):
    name=p.stem.lower()
    if 'gtvisual' in name or 'oracle' in name or 'gtproto' in name:continue
    if DS=='ade234' and 'ade234' not in name:continue
    if DS=='pp116' and 'ade234' in name:continue
    if not any(t in name for t in ('soft_structure','softrelative_structure')):continue
    wt=Path('weights')/(p.stem+'.pth')
    if not wt.is_file():continue
    try: data=json.loads(p.read_text())
    except Exception:continue
    def seek(o):
        if isinstance(o,dict):
            for k,v in o.items():
                kl=str(k).lower()
                if ('miou' in kl or kl=='mIoU') and isinstance(v,(int,float)):
                    f=float(v)
                    yield f*100 if 0<=f<=1 else f
                if isinstance(v,(dict,list)):
                    yield from seek(v)
        elif isinstance(o,list):
            for v in o:yield from seek(v)
    vals=list(seek(data))
    if any(abs(v-expected)<.011 for v in vals):found.append((p.stem,p,vals))
if manual:
    candidates=[x for x in found if x[0]==manual]
    if len(candidates)!=1:
        print('Requested OVPS_VIS_PROJ_NAME has no matching JSON+weights with exact expected mIoU:',manual,file=sys.stderr)
        raise SystemExit(5)
    found=candidates
if not found:
    print(f'No {DS} original JSON + weights match mIoU={expected}.',file=sys.stderr)
    print('Matching name candidate checkpoints:',file=sys.stderr)
    for p in Path('segmentation_results').glob('*.json'):
        if ('ade234' in p.stem)==(DS=='ade234') and ('soft' in p.stem):
            print(' ',p, 'weight?', (Path('weights')/(p.stem+'.pth')).is_file(),file=sys.stderr)
    raise SystemExit(3)
if len(found)>1:
    preferred = ('vitb_mlp_infonce_coco2014_reproduce_clean_softrelative_structure_e100_reproduce'
                 if DS=='pp116' else
                 'vitb_mlp_infonce_coco2014_reproduce_clean_ade234_soft_structure_lambda4p0_100ep')
    selected=[x for x in found if x[0]==preferred]
    if len(selected)==1:
        found=selected
if len(found)>1:
    # Do not silently switch between multiple plausible E100 runs.
    print('Multiple matching E100 checkpoint candidates:',file=sys.stderr)
    for x in found:print(' ',x[0],file=sys.stderr)
    raise SystemExit(4)
print('[CHECKPOINT MATCH] '+str(found[0][1])+' metric='+str(found[0][2]),file=sys.stderr)
print(found[0][0])
PY
)"
echo "[RUN] dataset=$DS proj=$PROJ expected_mIoU=$OVPS_VIS_EXPECTED_MIOU"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/ours_triptych_diagnostic_json
python -m torch.distributed.run --nproc_per_node=1 --master_port="$([[ "$DS" == pp116 ]] && echo 33961 || echo 33962)" \
  src/open_vocabulary_segmentation/main_ours_val_triptych.py \
  --eval --output "output/ours_triptych_eval_${DS}" \
  --eval_cfg "$ECFG" --eval_base_cfg "$BCFG" \
  --opts "model.proj_name=$PROJ" evaluate.pamr=false

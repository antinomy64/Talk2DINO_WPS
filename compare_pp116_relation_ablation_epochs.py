#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare matched PP116 lambda_struct=0 vs 4, at E0/10/.../100.

Ground truth is used ONLY for offline analysis (GT prototype TV retrieval, and
validation IoU from existing evaluation JSON); never for training.
Uses the same frozen projector + W checkpoints as original training. No model edits.
Run from Talk2DINO_official_bg repo root, after run_pp116_relation_ablation.sh eval.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import train_relproto_alignemt as original
from analyze_pp116_within_object_structure import canonical_classes,verify_names,read_pt

PFX='vitb_mlp_infonce_coco2014_DIAGREL'

def save(path, rows):
    if not rows: raise RuntimeError(f'No output rows for {path}')
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

def check_ckpt(path,lambda_expected):
    d=torch.load(path,map_location='cpu',weights_only=False) # Only own local trusted .pt
    if not isinstance(d,dict) or 'W' not in d: raise ValueError(f'Invalid checkpoint {path}')
    ar=d.get('args')
    v=(ar.get('lambda_struct') if isinstance(ar,dict) else getattr(ar,'lambda_struct',None)) if ar is not None else None
    if v is None or not math.isclose(float(v),float(lambda_expected),abs_tol=1e-7):
        raise ValueError(f'Mismatched lambda in {path}: {v}, expected {lambda_expected}')
    W=torch.as_tensor(d['W'],dtype=torch.float64)
    if tuple(W.shape)!=(768,768):raise ValueError(f'Invalid W shape {W.shape}')
    if float((W.T@W-torch.eye(768,dtype=torch.float64)).abs().max())>1e-2:raise ValueError(f'W not orthogonal: {path}')
    return W


def get_iou(path, names):
    import re

    data = json.loads(path.read_text(encoding="utf-8"))

    # This repository stores ONLY overall mIoU in segmentation_results/*.json
    if not isinstance(data, dict) or "voc116_part" not in data:
        raise ValueError(f"Invalid evaluation JSON: {path}")

    value = data["voc116_part"]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric mIoU, got {value!r}")

    miou = float(value)
    if 0 <= miou <= 1:
        miou *= 100.0

    # Per-class IoU and Acc are stored in output/.../eval.log
    match = re.search(r"(lam[04]_e\d{3})$", path.stem)
    if match is None:
        raise ValueError(f"Cannot identify lambda/epoch: {path}")

    log_path = Path("output") / (
        "pp116_DIAGREL_" + match.group(1)
    ) / "eval.log"

    if not log_path.is_file():
        raise FileNotFoundError(
            f"Missing per-class evaluation log: {log_path}"
        )

    wanted = set(names)
    records = {}
    pattern = re.compile(
        r"^\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$"
    )

    for line in log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = pattern.match(line)
        if m is None:
            continue

        part, iou_str, acc_str = (
            x.strip() for x in m.groups()
        )

        if part not in wanted:
            continue

        try:
            iou_val = float(iou_str)
            acc_val = float(acc_str)
        except ValueError:
            continue

        if not math.isfinite(iou_val) or not 0 <= iou_val <= 100:
            raise ValueError(f"Invalid IoU: {part}={iou_val}")

        if part in records:
            old_iou, old_acc = records[part]
            if not (
                math.isclose(old_iou, iou_val, abs_tol=1e-6)
                and (
                    (math.isnan(old_acc) and math.isnan(acc_val))
                    or math.isclose(old_acc, acc_val, abs_tol=1e-6)
                )
            ):
                raise ValueError(f"Conflicting duplicate row: {part}")

        records[part] = (iou_val, acc_val)

    missing = [name for name in names if name not in records]

    if missing:
        raise ValueError(
            f"{log_path}: found {len(records)}/{len(names)} classes; "
            f"missing={missing[:10]}"
        )

    iou = {name: records[name][0] for name in names}
    valid = {
        name: math.isfinite(records[name][1])
        for name in names
    }

    calculated_miou = sum(iou.values()) / len(names)

    if abs(calculated_miou - miou) > 0.15:
        raise ValueError(
            f"mIoU mismatch: JSON={miou:.4f}%, "
            f"eval.log={calculated_miou:.4f}%"
        )

    print(
        f"[PASS] {match.group(1)} "
        f"mIoU={miou:.2f}% "
        f"classes={len(records)} "
        f"valid={sum(valid.values())}"
    )

    return miou, iou, valid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run0',type=Path,default=Path('final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep'))
    p.add_argument('--run4',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--gt-bank',type=Path,default=Path('feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt'))
    p.add_argument('--groups-csv',type=Path,default=Path(''),help='Optional pp116_six_groups_per_part(1).csv')
    p.add_argument('--eval-prefix',default=PFX)
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_DIAGREL_comparison'))
    a=p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise FileExistsError(f'Output already exists, refusing overwrite: {a.out_dir}')
    names=canonical_classes(Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    raw,raw_names,_=original.load_raw_clip_bank(a.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    verify_names(raw_names,names,'text bank')
    dev=torch.device('cpu')
    projector,_=original.load_frozen_projector(project_root=Path.cwd(),config_path=a.model_config,weight_path=a.projector,device=dev)
    t=original.project_raw_clip_bank(raw,projector,device=dev,batch_size=128).to(dtype=torch.float64)
    t=F.normalize(t,dim=-1)
    del projector
    gd=read_pt(a.gt_bank)
    verify_names(gd['classes'],names,'GT bank')
    gt=F.normalize(torch.as_tensor(gd['prototypes'],dtype=torch.float64),dim=-1)
    gt_counts=np.asarray(gd['patch_counts'],dtype=int)
    assert gt.shape==(116,768) and gt_counts.shape==(116,)
    groups={}
    if str(a.groups_csv) not in ('','.','None') and a.groups_csv.is_file():
        import pandas as pd
        df=pd.read_csv(a.groups_csv,encoding='utf-8-sig')
        if not {'part','类别编码','六类轨迹'}.issubset(df.columns):raise ValueError('Six-group CSV schema mismatch')
        groups={str(row['part']):str(row['类别编码'])+' '+str(row['六类轨迹']) for _,row in df.iterrows()}
    epochs=list(range(0,101,10))
    records=[];epoch_summary=[]
    for ep in epochs:
        per={}
        for tag,run,lamb in [('lam0',a.run0,0.0),('lam4',a.run4,4.0)]:
            W=torch.eye(768,dtype=torch.float64) if ep==0 else check_ckpt(run/f'W_epoch_{ep:03d}.pt',lamb)
            tx=F.normalize(t@W,dim=-1)
            C=(tx@gt.T).numpy()
            name=f'{a.eval_prefix}_{tag}_e{ep:03d}'
            path=Path('segmentation_results')/(name+'.json')
            if not path.exists():raise FileNotFoundError(f'Missing evaluation; run bash ... eval first: {path}')
            mIoU,iou,valid=get_iou(path,names)
            item={}
            for pid,part in enumerate(names):
                scores=C[:,pid].copy()
                correct=float(scores[pid]);scores[pid]=-np.inf
                wrong=int(scores.argmax());best=float(scores[wrong])
                item[part]={'iou':iou[part] if valid[part] else math.nan,'valid':int(valid[part]),
                            'margin':correct-best,'correct_cos':correct,'wrong_cos':best,
                            'top_wrong':names[wrong], 'top1':int(correct>best),'gt_patches':int(gt_counts[pid])}
            per[tag]=item
            vals=np.asarray([item[n]['margin'] for n in names if gt_counts[names.index(n)]>0],dtype=float)
            s={'epoch':ep,'arm':tag,'lambda_struct':lamb,'official_mIoU_pct':mIoU,
               'GT_proto_TV_top1_count':sum(x['top1'] for x in item.values()),
               'GT_proto_margin_mean':float(vals.mean())}
            epoch_summary.append(s)
            print(f'E{ep:03d} {tag}: mIoU={mIoU:.2f}, proto_top1={s["GT_proto_TV_top1_count"]}/116, TV_margin={s["GT_proto_margin_mean"]:+.4f}')
        for pid,name in enumerate(names):
            x=per['lam0'][name];y=per['lam4'][name]
            records.append({'epoch':ep,'class_id':pid,'part':name,'group':groups.get(name,''),
              'gt_patches':x['gt_patches'],'valid_val_gt':x['valid'],
              'lam0_iou':x['iou'],'lam4_iou':y['iou'],'relation_iou_gain_lam4_minus_lam0':y['iou']-x['iou'],
              'lam0_gt_proto_margin':x['margin'],'lam4_gt_proto_margin':y['margin'],
              'relation_margin_gain_lam4_minus_lam0':y['margin']-x['margin'],
              'lam0_top1':x['top1'],'lam4_top1':y['top1'],
              'lam0_strongest_wrong':x['top_wrong'],'lam4_strongest_wrong':y['top_wrong']})
    a.out_dir.mkdir(parents=True)
    save(a.out_dir/'per_part_epoch_lambda0_vs_lambda4.csv',records)
    save(a.out_dir/'epoch_summary.csv',epoch_summary)
    group_summ=[]
    keys=[(r['epoch'],r['group']) for r in records]
    for ep,group in sorted(set(keys)):
        subset=[r for r in records if r['epoch']==ep and r['group']==group and r['valid_val_gt']]
        if not subset:continue
        deltas=np.asarray([r['relation_iou_gain_lam4_minus_lam0'] for r in subset],float)
        md=np.asarray([r['relation_margin_gain_lam4_minus_lam0'] for r in subset],float)
        group_summ.append({'epoch':ep,'group':group,'n':len(subset),'lambda4_minus_lambda0_mean_iou_gain':float(np.mean(deltas)),
                           'lambda4_minus_lambda0_mean_margin_gain':float(np.mean(md)),
                           'lambda4_iou_better_n':int((deltas>0.01).sum()),'lambda4_iou_worse_n':int((deltas< -0.01).sum())})
    save(a.out_dir/'group_effects.csv',group_summ)
    final=sorted((r for r in records if r['epoch']==100 and r['valid_val_gt']),key=lambda r:r['relation_iou_gain_lam4_minus_lam0'])
    save(a.out_dir/'e100_worst_relation_effect.csv',final)
    report={'input':'same frozen initial text, same GT full-image bank, two independently trained orthogonal Ws',
      'lambda0':str(a.run0),'lambda4':str(a.run4),'eval_prefix':a.eval_prefix,
      'iou':'official 116-class validation IoU (Acc nan -> invalid), NO GT used during training',
      'relation_gain':'lambda4 IoU minus lambda0 IoU; positive = structure constraint helps',
      'gt_proto_mapping':'diagnostic only, train full-image GT pooled prototype',
      'note':'same architecture/hyperparams/data/seed required; checkpoint provenance must be checked'}
    (a.out_dir/'protocol.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE]',a.out_dir)

if __name__=='__main__':main()

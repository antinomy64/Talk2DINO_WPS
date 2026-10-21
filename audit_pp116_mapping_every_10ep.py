#!/usr/bin/env python3
"""Read-only PP116: GT visual-prototype -> full-116 text correspondence every 10 epochs.

Runs from Talk2DINO_official_bg root, beside analyze_pp116_within_object_structure.py.
NO GT used in training; NO weights or official code modified.
Reads base COCO projector + saved orthogonal W_epoch_XXX.pt, NOT a different projector.
"""
from __future__ import annotations
import argparse, csv, math, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from analyze_pp116_within_object_structure import canonical_classes, verify_names, read_pt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-dir', default='final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep')
    p.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--gt-bank', default='feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt')
    p.add_argument('--classes-file', default='src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py')
    p.add_argument('--iou-csv', default='output/pp116_noabs_iou_curve/per_part_iou.csv')
    p.add_argument('--epochs', nargs='+', type=int, default=list(range(0,101,10)))
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--out-dir', default='output/pp116_mapping_all_epochs')
    return p.parse_args()


def save(path, rows):
    if not rows:
        raise ValueError(f'No rows: {path}')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def load_iou(path, names, epochs):
    """Expected existing wide per_part_iou.csv: part, GT_valid, E000,...,E100, values in %."""
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        columns=set(reader.fieldnames or [])
        if not {'part','GT_valid'}.issubset(columns):
            raise ValueError('IoU CSV needs part and GT_valid: '+str(sorted(columns)))
        expected=[f'E{e:03d}' for e in epochs]
        missing=[e for e in expected if e not in columns]
        if missing: raise ValueError('IoU CSV missing columns '+str(missing))
        records={}
        for r in reader:
            name=r['part']
            if name in records:raise ValueError('Duplicate IoU part: '+name)
            valid=r['GT_valid'].strip().lower() in ('true','1','yes')
            records[name]={e:float(r[f'E{e:03d}']) if valid else math.nan for e in epochs}
        verify_names(sorted(records),sorted(names),'IoU CSV class-name inventory')
        return records


def spearman(x,y):
    from scipy.stats import spearmanr
    a=np.asarray(x,dtype=float);b=np.asarray(y,dtype=float)
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<3: return math.nan
    return float(spearmanr(a[mask],b[mask]).statistic)


def main():
    a=parse_args()
    epochs=sorted(set(a.epochs))
    if any(e<0 or e>100 for e in epochs): raise ValueError('Unexpected epoch outside 0..100')
    if 0 not in epochs or 100 not in epochs:raise ValueError('Include epochs 0 and 100 to cross-check')
    import train_relproto_alignemt as original
    names=canonical_classes(Path(a.classes_file)); n=len(names)
    iou=load_iou(Path(a.iou_csv),names,epochs)
    raw,raw_names,_=original.load_raw_clip_bank(Path(a.text_bank),expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    verify_names(raw_names,names,'CLIP text')
    dev=torch.device(a.device)
    projector,_=original.load_frozen_projector(project_root=Path.cwd(),config_path=Path(a.model_config),weight_path=Path(a.projector),device=dev)
    base=original.project_raw_clip_bank(raw,projector,device=dev,batch_size=128).to(dtype=torch.float64)
    base=F.normalize(base,dim=-1)
    del projector
    gt=read_pt(Path(a.gt_bank))
    if not isinstance(gt,dict):raise TypeError('GT bank must be mapping')
    verify_names(gt['classes'],names,'GT bank classes')
    patches=np.asarray(gt['patch_counts'],dtype=np.int64).reshape(-1)
    images=np.asarray(gt['image_counts'],dtype=np.int64).reshape(-1)
    proto=torch.as_tensor(gt['prototypes'],dtype=torch.float64)
    if proto.shape != (n,768) or patches.shape != (n,) or images.shape != (n,): raise ValueError('Invalid bank shape')
    valid=np.isfinite(proto.numpy()).all(axis=1)&(np.linalg.norm(proto.numpy(),axis=1)>1e-12)&(patches>0)&(images>0)
    if not valid.all():
        print(f'[WARN] {np.count_nonzero(~valid)} GT prototypes invalid, exclude their GT queries')
    v=F.normalize(proto,dim=-1)
    out=Path(a.out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Refusing to overwrite outputs: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; summaries=[]; prev=None; transitions=[]
    for ep in epochs:
        if ep==0:
            W=torch.eye(768,dtype=torch.float64)
        else:
            path=Path(a.weights_dir)/f'W_epoch_{ep:03d}.pt'
            if not path.is_file():raise FileNotFoundError(f'Checkpoint absent: {path}')
            d=read_pt(path)
            if not isinstance(d,dict) or 'W' not in d:raise ValueError(f'Checkpoint missing W: {path}')
            if d.get('selector','relative')!='relative' or d.get('mapping','orthogonal')!='orthogonal':
                raise ValueError(f'Expected relative + orthogonal: {path}, metadata={d.get("selector"),d.get("mapping")}')
            W=torch.as_tensor(d['W'],dtype=torch.float64)
            if W.shape!=(768,768):raise ValueError(f'Invalid W shape: {W.shape}')
            ortho_error=float((W.T@W-torch.eye(768,dtype=torch.float64)).abs().max())
            if ortho_error>1e-2:raise ValueError(f'W not orthogonal at E{ep}: error={ortho_error}')
        text=F.normalize(base@W,dim=-1)
        C=(text@v.T).numpy()  # [text_class, visual_GT_class]
        current=[]
        for p,name in enumerate(names):
            if not valid[p]:continue
            scores=C[:,p].copy()
            correct=float(scores[p]);scores[p]=-np.inf
            competitor=int(np.argmax(scores)); margin=correct-float(scores[competitor])
            prediction=p if margin>=0 and correct>=float(scores[competitor]) else competitor
            result={
                'epoch':ep,'class_id':p,'class_name':name,'object':name.split("'s ",1)[0],
                'gt_patches':int(patches[p]),'gt_images':int(images[p]),
                'gt_correct_text_cos':correct,'top_wrong_text':names[competitor],
                'max_wrong_cos':float(scores[competitor]),'gt_margin':margin,
                'gt_proto_predicted_text':names[prediction],
                'mapping_correct':int(prediction==p),
                'validation_iou_pct':iou[name][ep],
            }
            rows.append(result);current.append(result)
        iv=np.asarray([r['validation_iou_pct'] for r in current],float)
        mv=np.asarray([r['gt_margin'] for r in current],float)
        cv=np.asarray([r['mapping_correct'] for r in current],float)
        s={'epoch':ep,'valid_prototypes':len(current),'top1_correct_all_prototypes':int(cv.sum()),'top1_rate_pct':100*cv.mean(),
           'valid_val_iou_parts':int(np.isfinite(iv).sum()),'mean_per_part_iou_pct':float(np.nanmean(iv)),
           'mean_gt_margin':float(mv.mean()),'spearman_margin_iou':spearman(mv,iv)}
        if prev is not None:
            old={r['class_id']:r for r in prev}
            newgood=0;lostgood=0
            for r in current:
                p=r['class_id'];a0=old[p];y0=a0['validation_iou_pct'];y1=r['validation_iou_pct']
                category=('correct_to_correct' if a0['mapping_correct'] and r['mapping_correct'] else
                          'correct_to_wrong' if a0['mapping_correct'] else
                          'wrong_to_correct' if r['mapping_correct'] else 'wrong_to_wrong')
                if category=='wrong_to_correct':newgood+=1
                if category=='correct_to_wrong':lostgood+=1
                transitions.append({'epoch_from':prev[0]['epoch'],'epoch_to':ep,'class_name':r['class_name'],
                                    'status':category,'iou_from_pct':y0,'iou_to_pct':y1,
                                    'delta_iou_pct':y1-y0,'margin_from':a0['gt_margin'],
                                    'margin_to':r['gt_margin'],'delta_margin':r['gt_margin']-a0['gt_margin']})
            s['newly_correct_from_prev']=newgood;s['lost_correct_from_prev']=lostgood
            s['delta_mean_iou_pct']=s['mean_per_part_iou_pct']-summaries[-1]['mean_per_part_iou_pct']
        else:
            s['newly_correct_from_prev']=0;s['lost_correct_from_prev']=0;s['delta_mean_iou_pct']=math.nan
        summaries.append(s);prev=current
        print(f'E{ep:03d}: correct={int(cv.sum()):3d}/{len(current)}  valid_val_part_meanIoU={float(np.nanmean(iv)):.2f}%  '
              f'margin_mean={float(mv.mean()):+.4f}  margin-vs-IoU-rho={s["spearman_margin_iou"]:+.3f}')
    save(out/'per_part_per_epoch.csv',rows)
    save(out/'epoch_summary.csv',summaries)
    save(out/'per_part_transitions.csv',transitions)
    group_rows=[]
    by=defaultdict(list)
    for r in transitions: by[(r['epoch_from'],r['epoch_to'],r['status'])].append(r)
    for (e0,e1,cat),grp in sorted(by.items()):
        arr=np.asarray([r['delta_iou_pct'] for r in grp],float)
        arr=arr[np.isfinite(arr)]
        group_rows.append({'epoch_from':e0,'epoch_to':e1,'status':cat,'num_parts':len(grp),
                           'num_valid_iou':len(arr), 'mean_iou_change_pct':float(arr.mean()) if len(arr) else math.nan,
                           'median_iou_change_pct':float(np.median(arr)) if len(arr) else math.nan,
                           'num_iou_improved':int((arr>0.01).sum()),'num_iou_declined':int((arr < -0.01).sum()),
                           'num_iou_unchanged':int((abs(arr)<=0.01).sum())})
    save(out/'transition_group_summary.csv',group_rows)
    meta={'source':'Saved W checkpoints applied to frozen projected text; GT full-image prototypes only for audit',
          'epochs':epochs,'weights_dir':a.weights_dir,'iou_csv':a.iou_csv,
          'mean_iou_note':'classwise mean over valid parts from provided IoU CSV; NOT official evaluation mIoU',
          'mapping_note':'GT mean-prototype retrieval under FULL 116-class text vocabulary, NOT dense per-patch accuracy'}
    (out/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[OK] Saved:',out)

if __name__=='__main__':main()

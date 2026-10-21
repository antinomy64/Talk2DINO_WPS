#!/usr/bin/env python3
"""Direct Text-to-GT-prototype correspondence diagnostic for PascalPart-116.

READ-ONLY. No training and no GT use in training. Run in repository root beside
analyze_pp116_within_object_structure.py and train_relproto_alignemt.py.

Compare the original object projector (E0) and an E100 baked projector.
Rows of C = T @ V_GT.T are text queries, columns are visual GT prototypes.
The model's inference direction is VISUAL -> TEXT (columns).

Outputs are diagnostic of GLOBAL GT PROTOTYPES, not dense prediction quality.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from analyze_pp116_within_object_structure import (
    canonical_classes, verify_names, object_of, read_pt, read_iou_csv,
    file_sha256, require_file,
)

TEXT = 'feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
INIT = 'weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
E100 = 'weights/vitb_mlp_infonce_coco2014_reproduce_clean_softrelative_structure_e100_reproduce.pth'
GT = 'feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt'
CONFIG = 'configs/vitb_mlp_infonce.yaml'
CLASSES = 'src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'
IOU = 'output/pp116_noabs_iou_curve/per_part_iou.csv'


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-bank', default=TEXT)
    p.add_argument('--init-projector', default=INIT)
    p.add_argument('--e100-projector', default=E100)
    p.add_argument('--gt-bank', default=GT)
    p.add_argument('--model-config', default=CONFIG)
    p.add_argument('--classes-file', default=CLASSES)
    p.add_argument('--iou-csv', default=IOU)
    p.add_argument('--out-dir', default='output/pp116_direct_text_gt_e0_e100')
    p.add_argument('--device', default='cpu')
    p.add_argument('--no-figures', action='store_true')
    return p.parse_args()


def write(path, records):
    if not records:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def desc(names, ids, vals, limit=3):
    return ' | '.join(f'{names[j]}={vals[k]:.4f}' for k,j in list(zip(range(len(ids)),ids))[:limit])


def get_top_ranks(candidates, similarities, correct_id):
    # Lexical ID tie-breaking gives deterministic output if scores are exactly equal.
    ranked = sorted(candidates, key=lambda j: (-float(similarities[j]), j))
    correct_score = float(similarities[correct_id])
    rank = 1 + sum(float(similarities[j]) > correct_score + 1e-12 for j in candidates if j != correct_id)
    top_id = ranked[0]
    wrong_best = max((float(similarities[j]) for j in candidates if j != correct_id), default=math.nan)
    return top_id, rank, correct_score - wrong_best if len(candidates) > 1 else math.nan, ranked[:3]


def text_bank_with_projector(raw, path, device, tr, repo, cfg):
    projector,_ = tr.load_frozen_projector(project_root=repo,
        config_path=cfg,weight_path=path,device=device)
    projected = tr.project_raw_clip_bank(raw, projector, device=device, batch_size=128)
    projected = tr.normalize_last(projected.float()).cpu().double()
    del projector
    return projected


def main():
    a=args()
    repo=Path.cwd()
    paths={
        'text':require_file(a.text_bank),
        'initial_projector':require_file(a.init_projector),
        'e100_projector':require_file(a.e100_projector),
        'GT':require_file(a.gt_bank),
        'config':require_file(a.model_config),
        'classes':require_file(a.classes_file),
    }
    names=canonical_classes(paths['classes'])
    import train_relproto_alignemt as tr
    raw, raw_names, _=tr.load_raw_clip_bank(paths['text'],
        expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    verify_names(raw_names,names,'CLIP text bank')
    device=torch.device(a.device)
    T={
        'E000':text_bank_with_projector(raw,paths['initial_projector'],device,tr,repo,paths['config']),
        'E100':text_bank_with_projector(raw,paths['e100_projector'],device,tr,repo,paths['config']),
    }
    gt=read_pt(paths['GT'])
    if not isinstance(gt,dict):
        raise TypeError('GT bank must be dict with prototypes, classes, patch_counts, image_counts')
    for key in ('prototypes','patch_counts','image_counts','classes'):
        if key not in gt: raise KeyError(f'GT bank missing {key}')
    verify_names(gt['classes'],names,'GT visual bank')
    V=torch.as_tensor(gt['prototypes'],dtype=torch.float32)
    patch_count=np.asarray(gt['patch_counts'],dtype=np.int64).reshape(-1)
    image_count=np.asarray(gt['image_counts'],dtype=np.int64).reshape(-1)
    if V.shape!=(116,768) or patch_count.shape!=(116,) or image_count.shape!=(116,):
        raise ValueError(f'Bad GT dimensions: {V.shape} {patch_count.shape} {image_count.shape}')
    valid=(patch_count>0)&(image_count>0)&torch.isfinite(V).all(dim=1).numpy()&(V.norm(dim=1).numpy()>1e-12)
    if valid.sum()!=116:
        print(f'[INFO] {valid.sum()}/116 valid GT prototypes; invalid classes excluded from candidate visual prototypes')
    VN=torch.zeros((116,768),dtype=torch.float64)
    VN[valid]=F.normalize(V[valid].double(),dim=-1)
    by_object=defaultdict(list)
    for i,name in enumerate(names):
        by_object[object_of(name)].append(i)
    stats={}
    iou_path=Path(a.iou_csv)
    if a.iou_csv and iou_path.exists():
        stats,_=read_iou_csv(iou_path,names)
    elif a.iou_csv:
        print('[INFO] IoU CSV absent, proceeding without IoU:',iou_path)
    out=Path(a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory not empty; choose a new --out-dir: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    scores={ep:(T[ep]@VN.T).numpy() for ep in T}
    # No numerical manipulation of cosine scores: these already use unit vectors.
    rows=[]; pairs=[]; objs=[]
    valid_ids=[i for i in range(116) if valid[i]]
    for obj, ids in by_object.items():
        vis_ids=[i for i in ids if valid[i]]
        for ep,C in scores.items():
            summary=[]
            for p in ids:
                base={
                    'epoch':ep,'class_id':p,'class_name':names[p],'object':obj,
                    'gt_patches':int(patch_count[p]),'gt_images':int(image_count[p]),
                    'valid_gt':int(valid[p]),
                }
                for key in ('iou_ep0_pct','iou_ep100_pct','iou_peak_pct',
                            'iou_peak_to_last_drop_pct'):
                    base[key]=stats.get(names[p],{}).get(key,math.nan)
                if not valid[p]:
                    rows.append(base); continue
                # text -> visual row: how the prompt's direction matches prototypes
                rowids=vis_ids
                rt,rr,rm,rlist=get_top_ranks(rowids,C[p,:],p)
                # visual -> text column: matches the dense segmentation decision direction
                ct,cr,cm,clist=get_top_ranks(ids,C[:,p],p)
                # visual -> full 116-way text vocabulary (same competition as inference)
                at,ar,am,alist=get_top_ranks(list(range(116)),C[:,p],p)
                base.update({
                    'correct_text_gt_cos':float(C[p,p]),
                    'text_to_visual_top1':names[rt],
                    'text_to_visual_correct':int(rt==p),
                    'text_to_visual_gt_rank':int(rr),
                    'text_to_visual_gt_margin':rm,
                    'text_to_visual_top3':' | '.join(f'{names[j]}:{C[p,j]:.4f}' for j in rlist),
                    'visual_to_text_top1':names[ct],
                    'visual_to_text_correct':int(ct==p),
                    'visual_to_text_gt_rank':int(cr),
                    'visual_to_text_gt_margin':cm,
                    'visual_to_text_top3':' | '.join(f'{names[j]}:{C[j,p]:.4f}' for j in clist),
                    'full116_visual_to_text_top1':names[at],
                    'full116_visual_to_text_correct':int(at==p),
                    'full116_visual_to_text_gt_rank':int(ar),
                    'full116_visual_to_text_gt_margin':am,
                    'full116_visual_to_text_top3':' | '.join(f'{names[j]}:{C[j,p]:.4f}' for j in alist),
                    'mutual_top1_within_object':int(rt==p and ct==p),
                })
                rows.append(base)
                summary.append(base)
            if summary:
                objs.append({
                    'epoch':ep,'object':obj,'valid_gt_parts':len(summary),
                    'text_to_visual_top1_correct':sum(r['text_to_visual_correct'] for r in summary),
                    'visual_to_text_top1_correct':sum(r['visual_to_text_correct'] for r in summary),
                    'full116_visual_to_text_top1_correct':sum(r['full116_visual_to_text_correct'] for r in summary),
                    'mutual_top1':sum(r['mutual_top1_within_object'] for r in summary),
                    'text_to_visual_acc':np.mean([r['text_to_visual_correct'] for r in summary]),
                    'visual_to_text_acc':np.mean([r['visual_to_text_correct'] for r in summary]),
                    'full116_visual_to_text_acc':np.mean([r['full116_visual_to_text_correct'] for r in summary]),
                })
            for i in ids:
                for j in vis_ids:
                    pairs.append({
                        'epoch':ep,'object':obj,'text_part_id':i,'text_part':names[i],
                        'visual_part_id':j,'visual_part':names[j],
                        'cosine':float(C[i,j]),'diagonal_correct_pair':int(i==j),
                    })
    write(out/'part_text_gt_mappings.csv',rows)
    write(out/'object_text_gt_retrieval.csv',objs)
    write(out/'within_object_text_gt_cross_cosines.csv',pairs)
    # Per-class epoch0->epoch100 comparison incl IoU
    records={(r['epoch'],r['class_id']):r for r in rows}
    comparison=[]
    for i,name in enumerate(names):
        r0=records.get(('E000',i));r1=records.get(('E100',i))
        if r0 is None or r1 is None or not valid[i]: continue
        d={
            'class_id':i,'class_name':name,'object':object_of(name),
            'gt_patches':int(patch_count[i]), 'gt_images':int(image_count[i]),
            'iou_e0':r1.get('iou_ep0_pct',math.nan),
            'iou_e100':r1.get('iou_ep100_pct',math.nan),
            'iou_drop_from_peak':r1.get('iou_peak_to_last_drop_pct',math.nan),
        }
        for key in ('correct_text_gt_cos','text_to_visual_top1','text_to_visual_gt_rank',
                    'text_to_visual_gt_margin','visual_to_text_top1','visual_to_text_gt_rank',
                    'visual_to_text_gt_margin','full116_visual_to_text_top1',
                    'full116_visual_to_text_gt_rank','full116_visual_to_text_gt_margin'):
            d[f'{key}_e0']=r0.get(key)
            d[f'{key}_e100']=r1.get(key)
        d['delta_correct_cos_e100_e0']=r1['correct_text_gt_cos']-r0['correct_text_gt_cos']
        d['delta_within_visual_to_text_margin_e100_e0']=(
            r1['visual_to_text_gt_margin']-r0['visual_to_text_gt_margin'])
        d['delta_full116_margin_e100_e0']=(
            r1['full116_visual_to_text_gt_margin']-r0['full116_visual_to_text_gt_margin'])
        d['visual_to_text_change']=f"{r0['visual_to_text_top1']} -> {r1['visual_to_text_top1']}"
        d['full116_change']=f"{r0['full116_visual_to_text_top1']} -> {r1['full116_visual_to_text_top1']}"
        d['gt_support_warning']=int(patch_count[i]<100 or image_count[i]<10)
        # Diagnosis categories are hypotheses, not causal labels.
        low=math.isfinite(d['iou_e100']) and d['iou_e100']<10
        if low and r1['visual_to_text_correct'] and r1['full116_visual_to_text_correct']:
            diagnosis='PROTO_CORRECT_BUT_LOW_IOU__CHECK_PATCH_LOCALIZATION'
        elif low and r1['visual_to_text_correct'] and not r1['full116_visual_to_text_correct']:
            diagnosis='LOCAL_CORRECT_GLOBAL_COMPETITOR__CHECK_VOCAB'
        elif not r1['visual_to_text_correct'] and r0['visual_to_text_correct']:
            diagnosis='CORRESPONDENCE_LOST_E0_TO_E100'
        elif not r1['visual_to_text_correct']:
            diagnosis='WITHIN_OBJECT_PROTO_MATCH_WRONG_AT_E100'
        else:
            diagnosis='WITHIN_OBJECT_PROTO_MATCH_CORRECT'
        d['diagnostic_group']=diagnosis
        comparison.append(d)
    write(out/'epoch0_vs_100_correspondence.csv',comparison)
    meta={
        'scope':'within-object direct text-vs-full-image GT prototype cross-modal cosine, and additional full116 text candidate retrieval (not global relational structure)',
        'direction_note':'VISUAL->TEXT column retrieval approximates inference vocabulary competition at the GT *mean prototype* level. It is NOT pixel segmentation accuracy.',
        'text_pipeline':'Raw CLIP unnormalized prompt mean -> frozen projector -> L2, both E0 and E100 baked projector',
        'visual_pipeline':'Full-image DINOv2 GT patchpool prototype -> L2',
        'gt_warning':'GT is used ONLY offline; prototype averaging hides spatial/multi-modal distributions',
        'input_checksums':{k:file_sha256(v) for k,v in paths.items()},
        'gt_metadata':{str(k):v for k,v in gt.get('metadata',{}).items()
                       if isinstance(v,(str,int,float,bool,list))},
        'visual_valid_count':int(valid.sum()),
        'iou_csv':str(iou_path) if stats else None,
    }
    (out/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if not a.no_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figdir=out/'object_heatmaps'
        figdir.mkdir()
        for obj,ids in by_object.items():
            ids=[i for i in ids if valid[i]]
            if not ids:continue
            A=scores['E000'][np.ix_(ids,ids)]
            B=scores['E100'][np.ix_(ids,ids)]
            lo=min(float(A.min()),float(B.min()))
            hi=max(float(A.max()),float(B.max()))
            if lo==hi:hi=lo+1e-6
            size=max(4.5,0.58*len(ids))
            fig,axs=plt.subplots(1,2,figsize=(2*size+1,size+1))
            partnames=[names[i].split("'s ",1)[-1] for i in ids]
            for ax,M,ep in zip(axs,(A,B),('E000','E100')):
                im=ax.imshow(M,vmin=lo,vmax=hi,cmap='viridis')
                ax.set_title(f'{obj.upper()} | {ep} | Text (rows) -> GT Visual (cols)')
                ax.set_xticks(range(len(ids)),partnames,rotation=65,ha='right',fontsize=8)
                ax.set_yticks(range(len(ids)),partnames,fontsize=8)
                ax.set_xlabel('GT visual prototype')
                ax.set_ylabel('Text part')
                if len(ids)<=10:
                    for i in range(len(ids)):
                        for j in range(len(ids)):
                            ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=5.8,
                                    color='white' if M[i,j] < (lo+hi)/2 else 'black')
                # Outline diagonal (correct correspondence). The maximum in each column
                # tells which TEXT LABEL the visual prototype would select.
                for k in range(len(ids)):
                    ax.add_patch(plt.Rectangle((k-.5,k-.5),1,1,fill=False,
                                               edgecolor='red',linewidth=1.1))
            fig.colorbar(im,ax=axs,shrink=.78,label='cross-modal cosine')
            fig.subplots_adjust(left=.12,right=.9,bottom=.2,top=.85,wspace=.45)
            fig.savefig(figdir/(obj+'.png'),dpi=170)
            plt.close(fig)
    print('\n[FINISHED] Direct cross-modal correspondence audit')
    for ep in ('E000','E100'):
        curr=[r for r in rows if r['epoch']==ep and r['valid_gt']]
        print(ep,'valid parts',len(curr),
              'within visual->text correct',sum(r['visual_to_text_correct'] for r in curr),
              'within text->visual correct',sum(r['text_to_visual_correct'] for r in curr),
              'full116 visual->text correct',sum(r['full116_visual_to_text_correct'] for r in curr))
    print('\nHighest priority:')
    for d in comparison:
        if 'PROTO_CORRECT_BUT_LOW_IOU' in d['diagnostic_group'] and d['gt_patches']>=500:
            print('  PROTOTYPE OK but low IoU:',d['class_name'],d['iou_e100'])
    print('Results:',out)
    for file in sorted(out.glob('*.csv')):
        print('  ',file.name)

if __name__=='__main__':
    main()

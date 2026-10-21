#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual, read-only PP116 GT/cached-patch geometry audit.

Run from Talk2DINO_official_bg, with audit_pp116_gt_weighted_purity.py present.
For each chosen crop+part, save a 3x4 qualitative sheet comparing:
  original RGB, reconstructed object crop, native-resolution part GT, 32x32 GT,
  cached predicted foreground, its overlap with the selected GT part,
  no-flip versus flip GT assumptions, and lambda0/lambda4 Soft Relative weights.

The weights are recomputed from the same frozen Text/COCO projector and the
same W checkpoints as the earlier purity audit (GT never enters scoring).
Checks against per_crop_focus.csv fail fast if the original Purity result cannot
be reproduced. This does NOT independently prove that the *cached DINO tokens*
come from the displayed reconstructed image; that requires checking the cache
extraction transforms (or re-extracting DINO features under matched preprocess).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import torch

import audit_pp116_gt_weighted_purity as audit

SIZE = 448
FOCUS = ["bicycle's saddle", "cow's horn", "person's upper arm",
         "cat's leg", "aeroplane's body", "train's front", "sheep's horn"]


def safe(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def to_448(img, *, nearest=False):
    im = Image.fromarray(np.asarray(img))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    return np.asarray(im.resize((SIZE, SIZE), resample=method))


def crop_geo(record, rgb, full_gt, flip_mode):
    if rgb.shape[:2] != full_gt.shape:
        raise ValueError(f"RGB dimensions {rgb.shape[:2]} != part GT {full_gt.shape}; cannot align by pixel XYXY")
    h, w = full_gt.shape
    box = audit.get_box(record)
    x0, y0, x1, y1 = [int(round(float(q))) for q in box]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box {box} for {w}x{h}")
    rgb_orig = np.array(rgb[y0:y1, x0:x1], copy=True)
    gt_orig = np.array(full_gt[y0:y1, x0:x1], copy=True)
    flipped = audit.get_flip(record, flip_mode)
    rgb_crop = np.ascontiguousarray(rgb_orig[:, ::-1] if flipped else rgb_orig)
    gt_crop = np.ascontiguousarray(gt_orig[:, ::-1] if flipped else gt_orig)
    gt32 = to_448(gt_crop.astype(np.int32), nearest=True)  # display native GT on 448 canvas
    # To reproduce audit: resize directly from original pixel crop -> 32x32
    grid = np.asarray(Image.fromarray(gt_crop.astype(np.int32), mode='I').resize(
        (32, 32), Image.Resampling.NEAREST), dtype=np.int32)
    return {
        'box': (x0, y0, x1, y1), 'flipped': flipped,
        'rgb_orig_crop': rgb_orig,
        'rgb_crop': to_448(rgb_crop),
        'gt_crop_448': gt32,
        'gt32': grid,
        'gt32_no_flip': np.asarray(Image.fromarray(gt_orig.astype(np.int32), mode='I').resize(
            (32, 32), Image.Resampling.NEAREST), dtype=np.int32),
        'gt32_flip': np.asarray(Image.fromarray(np.ascontiguousarray(gt_orig[:, ::-1]).astype(np.int32), mode='I').resize(
            (32, 32), Image.Resampling.NEAREST), dtype=np.int32),
    }


def indexed_gt_overlay(rgb, gt, pid, *, is32=False):
    labels = np.asarray(gt)
    if is32:
        labels = to_448(labels.astype(np.int32), nearest=True)
    if labels.shape != rgb.shape[:2]:
        labels = to_448(labels.astype(np.int32), nearest=True)
    base = rgb.astype(np.float32) / 255.0
    # green: target part, orange: other annotated part, gray: ignore (NOT necessarily background)
    result = base.copy()
    for idx, color, alpha in ((255, [0.40, 0.40, 0.44], 0.35),
                               (-1, [1.00, 0.60, 0.03], 0.47),
                               (pid, [0.02, 0.96, 0.27], 0.67)):
        if idx == -1:
            m = (labels != 255) & (labels != pid)
        else:
            m = labels == idx
        result[m] = result[m] * (1.0 - alpha) + np.array(color) * alpha
    return np.clip(result, 0, 1)


def fg_overlay(rgb, fg):
    fg448 = to_448(fg.astype('uint8'), nearest=True).astype(bool)
    b = rgb.astype(float)/255.0
    result = b.copy()
    result[~fg448] = result[~fg448] * .35
    result[fg448] = result[fg448]*.62 + np.array([0.12, .96, .90])*.38
    return np.clip(result,0,1)


def overlap_img(rgb, gt32, fg, pid):
    # green = target & foreground; red = target dropped by foreground;
    # blue = foreground that is NOT target (not necessarily false positive).
    lab = to_448(gt32.astype(np.int32), nearest=True)
    f = to_448(fg.astype('uint8'), nearest=True).astype(bool)
    b = rgb.astype(float)/255.0 * 0.38
    for m, color in [((lab==pid)&f, [0.02, 1., 0.23]),
                     ((lab==pid)&~f, [1., 0.08, 0.14]),
                     ((lab!=pid)&f, [0.20, 0.52, 1.])]:
        b[m] = b[m]*0.16 + np.asarray(color)*0.84
    return np.clip(b,0,1)


def heatmap(w32, high):
    a = to_448(w32.astype('float32'), nearest=True)
    scale = np.clip(a/max(high,1e-12),0,1)**0.42
    return plt.get_cmap('magma')(scale)[:,:,:3]


def heat_overlay(rgb, w32, high):
    hm = heatmap(w32, high)
    w = to_448(w32.astype('float32'), nearest=True)
    strength = .85 * np.clip((w/max(high,1e-12))**0.42,0,1)
    return np.clip(rgb.astype(float)/255.0*(1-strength[:,:,None])
                   + hm*strength[:,:,None], 0,1)


def read_rows(p, epoch, parts):
    if not p.is_file():
        raise FileNotFoundError(f"Audit CSV absent: {p}. Use --indices together with --parts to choose cases manually, or point --audit-csv to existing per_crop_focus.csv")
    rows={}
    with p.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['epoch'])!=epoch or row['part'] not in parts: continue
            key=(int(row['crop_index']), int(row['class_id']))
            arm=row['arm']
            if arm not in ('lam0','lam4'):continue
            rows.setdefault(key,{})[arm]=row
    return rows


def pick_samples(rows, names, per_part, min_fg):
    bypart=defaultdict(list)
    for (idx,pid), pair in rows.items():
        if 'lam0' not in pair or 'lam4' not in pair: continue
        r=pair['lam4']
        if int(r['gt_fg_patch_count']) < min_fg: continue
        bypart[pid].append((idx,pair))
    chosen=[]
    for pid,name in enumerate(names):
        if name not in FOCUS and name not in {pair['lam4']['part'] for xs in bypart.values() for _,pair in xs}:
            continue
        candidates=bypart.get(pid,[])
        if not candidates: continue
        low=sorted(candidates, key=lambda x:(float(x[1]['lam4']['weighted_gt_purity']),-int(x[1]['lam4']['gt_fg_patch_count'])))
        contrast=sorted(candidates, key=lambda x: abs(float(x[1]['lam4']['weighted_gt_purity']) - float(x[1]['lam0']['weighted_gt_purity'])),reverse=True)
        high=sorted(candidates, key=lambda x:float(x[1]['lam4']['weighted_gt_purity']),reverse=True)
        selected=[]
        sources=(low,contrast,high)
        while len(selected)<per_part:
            made=False
            for source in sources:
                next_one=next((z for z in source if z[0] not in selected),None)
                if next_one is not None:
                    selected.append(next_one[0]);made=True
                    if len(selected)>=per_part:break
            if not made:break
        for idx in selected:chosen.append((idx,pid))
    return chosen


def label_panel(ax, image, name):
    ax.imshow(image,interpolation='nearest')
    ax.set_title(name,fontsize=10,pad=6)
    ax.axis('off')


def create_figure(out, image_id, part_name, pid, epoch, full_rgb, geo, fg, maps, stats, record):
    fig,axes=plt.subplots(3,4,figsize=(18,13),constrained_layout=False)
    fig.subplots_adjust(left=.035,right=.985,top=.89,bottom=.095,wspace=.055,hspace=.23)
    axes=axes.reshape((3,4))
    a=axes[0,0];a.imshow(full_rgb)
    x0,y0,x1,y1=geo['box']
    a.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='lime',lw=2.5))
    a.set_title('Original RGB + cached crop box',fontsize=10);a.axis('off')
    rgb=geo['rgb_crop']
    label_panel(axes[0,1],rgb, 'Reconstructed 448 crop RGB')
    label_panel(axes[0,2],indexed_gt_overlay(rgb,geo['gt_crop_448'],pid),'Original-pixel GT crop overlay')
    label_panel(axes[0,3],indexed_gt_overlay(rgb,geo['gt32'],pid,is32=True),'GT resized NEAREST to 32x32')

    label_panel(axes[1,0],fg_overlay(rgb,fg),'Cached predicted FG (32x32)')
    label_panel(axes[1,1],overlap_img(rgb,geo['gt32'],fg,pid),
                'GT target vs FG: green kept, red missed')
    label_panel(axes[1,2],indexed_gt_overlay(rgb,geo['gt32_no_flip'],pid,is32=True),
                'GT alternate: NO FLIP')
    label_panel(axes[1,3],indexed_gt_overlay(rgb,geo['gt32_flip'],pid,is32=True),
                'GT alternate: H-FLIP')
    if maps:
        w0=maps['lam0']['weights'].reshape((32,32));w4=maps['lam4']['weights'].reshape((32,32))
        high=max(float(w0.max()),float(w4.max()),1e-12)
        label_panel(axes[2,0],heatmap(w0,high),f"lambda=0 weights; purity={stats['lam0']['purity']:.3f}")
        label_panel(axes[2,1],heatmap(w4,high),f"lambda=4 weights; purity={stats['lam4']['purity']:.3f}")
        label_panel(axes[2,2],heat_overlay(rgb,w0,high),
                    f"lambda=0 heat overlay | support={maps['lam0']['n_support']}")
        label_panel(axes[2,3],heat_overlay(rgb,w4,high),
                    f"lambda=4 heat overlay | support={maps['lam4']['n_support']}")
    else:
        for a in axes[2]:a.axis('off')
        axes[2,0].text(.1,.5,'Geometry-only mode\n(no text weights loaded)',fontsize=14)

    orig_flip = audit.get_flip(record,'auto')
    recorded_flip=any(k in record for k in ('cropaug_hflip','cropaug_flipped','cropaug_flip','hflip','is_flipped','flipped','flip','flip_direction'))
    title=(f"PP116 GT/Crop/Patch spatial audit | {part_name} | {image_id} | cache row {record.get('_audit_index','?')} | E{epoch:03d}\n"
           f"xyxy={geo['box']}, selected flip={geo['flipped']}, recorded flip metadata={recorded_flip} "
           f"(auto={orig_flip}), GT target patches={int((geo['gt32']==pid).sum())}, "
           f"retained={int(((geo['gt32']==pid)&fg).sum())}, FG={int(fg.sum())}/1024")
    fig.suptitle(title,fontsize=12,fontweight='medium',y=.986)
    fig.text(.03,.058,
       "GT legend: GREEN=target part; ORANGE=other annotated parts; GRAY=IGNORE 255 (not necessarily background).\n"
       "Foreground overlap: GREEN=GT target retained; RED=GT target dropped; BLUE=predicted FG outside target (may be other valid parts).\n"
       "Lambda heatmaps: shared within-page intensity scale; weights from cached DINO tokens, GT read ONLY afterward.\n"
       "Visual inspection is necessary but not sufficient to prove cached token orientation; inspect extraction source for hidden augmentations.",
       fontsize=10,va='center')
    fig.savefig(out,dpi=125,facecolor='white')
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--geometry-only',action='store_true',help='Skip loading frozen projector and W; plot GT/FG geometry only')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--images-root',type=Path,default=Path('data/PascalPart116/images/train'))
    p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--audit-csv',type=Path,default=Path('output/pp116_purity_full/per_crop_focus.csv'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--run0',type=Path,default=Path('final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep'))
    p.add_argument('--run4',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--parts',nargs='+',default=FOCUS)
    p.add_argument('--indices',nargs='*',type=int,default=None,help='Manual 0-based record indices. Matched with each listed --parts present at that index.')
    p.add_argument('--samples-per-part',type=int,default=2,help='Automatic samples: low-purity, large delta-purity, optionally high-purity')
    p.add_argument('--min-gt-fg',type=int,default=10,help='Automatic selection: minimum target GT patches inside cached predicted FG')
    p.add_argument('--epoch',type=int,default=100)
    p.add_argument('--flip-mode',choices=('auto','none','all'),default='auto')
    p.add_argument('--soft-temperature',type=float,default=0.05)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_GT_alignment_viz_E100'))
    return p


def self_test():
    from tempfile import TemporaryDirectory
    h,w=120,180
    rgb=np.full((h,w,3),235,dtype=np.uint8)
    rgb[30:100,40:140]=[150,165,190]
    gt=np.full((h,w),255,dtype=np.int32)
    gt[50:85,70:95]=17
    gt[35:65,100:130]=23
    record={'cropaug_box_xyxy':[20,20,160,110], 'image_id':'mock_0001','part_category_id':[17,23]}
    geo=crop_geo(record,rgb,gt,'none')
    assert geo['gt32'].shape==(32,32)
    assert int((geo['gt32']==17).sum())>0
    assert np.array_equal(geo['gt32'],geo['gt32_no_flip'])
    assert np.array_equal(geo['gt32_flip'],geo['gt32_no_flip'][:,::-1])
    with TemporaryDirectory() as d:
        fp=Path(d)/'test.png'
        fg=np.zeros((32,32),dtype=bool);fg[5:27,5:27]=True
        create_figure(fp,'mock_0001','mock part',17,100,rgb,geo,fg,{}, {},record)
        assert fp.exists() and fp.stat().st_size>10000
    print('SELF_TEST_PASS: xyxy crop, nearest 32 GT, flip alternatives, image output')


def main():
    args=parser().parse_args()
    if args.self_test:return self_test()
    if args.samples_per_part<1 or not 0<=args.epoch<=100:raise ValueError('Invalid sampling/epoch')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Non-empty output directory (refuse to overwrite): {args.out_dir}')
    for required in (args.cache,args.classes_file):
        if not required.is_file():raise FileNotFoundError(required)
    for root in (args.images_root,args.gt_root):
        if not root.is_dir():raise FileNotFoundError(root)
    names=audit.canonical_names(args.classes_file)
    requested={names.index(n) for n in args.parts if n in names}
    missing=set(args.parts)-set(names)
    if missing:raise ValueError(f'Unknown names: {missing}')
    rows=read_rows(args.audit_csv,args.epoch,set(args.parts)) if args.audit_csv.is_file() else {}
    anns=audit.annotations(audit.load_pt(args.cache))
    if args.indices:
        chosen=[]
        for i in args.indices:
            if not 0<=i<len(anns):raise IndexError(f'Index {i} out of range: {len(anns)}')
            ids=audit.part_ids(anns[i])
            chosen.extend((i,p) for p in ids if p in requested)
    else:
        if not rows:raise ValueError('No audit CSV entries matching selected --parts and --epoch; pass --indices manually')
        chosen=pick_samples(rows,names,args.samples_per_part,args.min_gt_fg)
        chosen=[(i,p) for i,p in chosen if p in requested]
    if not chosen:raise RuntimeError('No samples selected (lower --min-gt-fg or pass --indices)')
    if len(chosen)>40:raise ValueError('Limit selection to <=40 figures per call')

    # Load exactly the same representation computation as the previous GT purity audit.
    text={}
    device=torch.device(args.device)
    if not args.geometry_only:
        if args.soft_temperature<=0:raise ValueError('Invalid soft temperature')
        import train_relproto_alignemt as src
        import train_pp116_softsingle_dual as train
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
        raw,raw_names,_=src.load_raw_clip_bank(args.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
        if list(raw_names)!=names:raise ValueError('Text-class ordering mismatch')
        project,_=src.load_frozen_projector(project_root=Path.cwd(),config_path=args.model_config,
                                           weight_path=args.projector,device=device)
        base=src.normalize_last(src.project_raw_clip_bank(raw,project,device=device,batch_size=128)
                                .detach().to(device=device,dtype=torch.float32))
        del project,raw
        for arm,folder,lam in (('lam0',args.run0,0),('lam4',args.run4,4)):
            W=audit.checkpoint_W(folder,args.epoch,lam).to(device)
            text[arm]=src.normalize_last(base@W).detach()
        if args.epoch==0 and not torch.equal(text['lam0'],text['lam4']):raise AssertionError('E0 arm mismatch')

    lookup=audit.GTLookup(args.gt_root)
    args.out_dir.mkdir(parents=True)
    checks=[]
    for i,pid in chosen:
        a=anns[i]
        image_id=str(a['image_id'])
        ip=args.images_root/(Path(image_id).stem+'.jpg')
        if not ip.is_file():
            tries=[args.images_root/(Path(image_id).stem+ext) for ext in ('.jpeg','.png','.JPG')]
            matches=[q for q in tries if q.is_file()]
            if len(matches)!=1:raise FileNotFoundError(f'Cannot find RGB {image_id} under {args.images_root}')
            ip=matches[0]
        with Image.open(ip) as im: full_rgb=np.asarray(im.convert('RGB'))
        gt=lookup.full(image_id)
        geo=crop_geo(a,full_rgb,gt,args.flip_mode)
        # Strong internal consistency: exactly the same indexed GT grid as prior purity implementation.
        old_gt, old_flip, shape, old_box = lookup.crop32(a,args.flip_mode)
        if not np.array_equal(old_gt,geo['gt32'].reshape(-1)) or old_flip!=geo['flipped']:
            raise AssertionError(f'GT geometry disagrees with prior audit at crop#{i}')
        fg=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool).reshape(-1).numpy().reshape(32,32)
        ids=audit.part_ids(a)
        maps={};stats={}
        if not args.geometry_only:
            with torch.no_grad():
                tokens=src.normalize_last(torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32,device=device))
                fgt=torch.as_tensor(fg.reshape(-1),device=device,dtype=torch.bool)
                if tokens.shape!=(1024,768) or not fgt.any():raise ValueError(f'Invalid tokens/foreground for row {i}')
                for arm in ('lam0','lam4'):
                    ww,cand,fall=audit.assignment(text[arm][ids],tokens,fgt,args.soft_temperature,src.relative_scores)
                    j=ids.index(pid)
                    w=ww[j].cpu().numpy()
                    gtflat=geo['gt32'].reshape(-1)
                    pur=float(w[gtflat==pid].sum())
                    ign=float(w[gtflat==255].sum())
                    # Cross-check weights with source implementation, never assume a reimplemented formula is identical.
                    if len(ids)>=2:
                        expected,extra=train.soft_relative_prototypes(text[arm][ids][None],tokens[None],fgt[None],temperature=args.soft_temperature)
                        own=torch.nn.functional.normalize(ww@tokens,dim=-1,eps=1e-12)
                        if (expected[0]-own).abs().max().item()>2e-5:
                            raise AssertionError(f'Soft Relative prototype mismatch at crop={i} arm={arm}')
                    maps[arm]={'weights':w,'n_support':int(cand[j].sum().item()),'fallback':bool(fall[j])}
                    stats[arm]={'purity':pur,'ignore_mass':ign}
                    existing=rows.get((i,pid),{}).get(arm)
                    if existing is not None:
                        old=float(existing['weighted_gt_purity'])
                        if abs(old-pur)>0.0003:
                            raise AssertionError(f'Prior purity mismatch: crop {i}, part {names[pid]}, {arm}: recomputed {pur} vs CSV {old}')
                        if int(existing['gt_fg_patch_count'])!=int(((geo['gt32']==pid)&fg).sum()):
                            raise AssertionError(f'GT/FG patch count mismatch at crop {i}, arm={arm}')
        px_count=int((geo['gt32']==pid).sum())
        retained=int(((geo['gt32']==pid)&fg).sum())
        px0=geo['gt32_no_flip']==pid
        px1=geo['gt32_flip']==pid
        def recall(target):
            den=int(target.sum())
            return float((target&fg).sum()/den) if den else math.nan
        fname=f'crop_{i:05d}_{safe(names[pid])}_E{args.epoch:03d}.png'
        a['_audit_index']=i  # local copy only; cache is never written
        create_figure(args.out_dir/fname,image_id,names[pid],pid,args.epoch,full_rgb,geo,fg,maps,stats,a)
        a.pop('_audit_index',None)
        checks.append({'crop_index':i,'image_id':image_id,'part':names[pid],'class_id':pid,
                       'crop_box':json.dumps(geo['box']),'flip_used':int(geo['flipped']),
                       'gt32_target_patches':px_count,'gt_target_in_fg':retained,
                       'fg_target_recall':retained/px_count if px_count else math.nan,
                       'no_flip_target_fg_recall':recall(px0),'flip_target_fg_recall':recall(px1),
                       'lam0_purity':stats.get('lam0',{}).get('purity',math.nan),
                       'lam4_purity':stats.get('lam4',{}).get('purity',math.nan),
                       'lam0_ignore_mass':stats.get('lam0',{}).get('ignore_mass',math.nan),
                       'lam4_ignore_mass':stats.get('lam4',{}).get('ignore_mass',math.nan),
                       'image_file':fname})
        print(f"[SAVED] {fname} | targetGT={px_count}, FGtarget={retained}, "
              f"P0={stats.get('lam0',{}).get('purity',float('nan')):.3f}, "
              f"P4={stats.get('lam4',{}).get('purity',float('nan')):.3f}",flush=True)
    with (args.out_dir/'alignment_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        ww=csv.DictWriter(f,fieldnames=list(checks[0].keys()));ww.writeheader();ww.writerows(checks)
    doc={'cache':str(args.cache),'gt_root':str(args.gt_root),'images_root':str(args.images_root),
         'audit_csv':str(args.audit_csv),'output_figures':len(checks),'epoch':args.epoch,'flip_mode':args.flip_mode,
         'method':'xyxy original-pixel crop; optional recorded H-flip; GT nearest 32x32; cached foreground and Soft Relative weights',
         'strict_checks':['RGB/GT size equal','prior audit gt32 exact-match','prior per-crop purity numerical match when available',
                          'actual trainer prototype numerical equivalence when weights enabled'],
         'caution':'No claim that recorded crop coordinates and flip fully reproduce cached DINO token extraction; unrecorded transforms cannot be inferred from GT alone. Alternative flip displays are visual diagnostics, not proofs.',
         'GT_only_for_offline_audit':True}
    (args.out_dir/'protocol.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[DONE] {len(checks)} figures -> {args.out_dir}',flush=True)


if __name__=='__main__':
    main()

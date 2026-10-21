#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full PP116 crop x candidate-part x epoch visual & quantitative *offline* GT audit.

REQUIRES in same repo:
  audit_pp116_gt_weighted_purity.py  (GT mapping, projected text, W loading)
  train_relproto_alignemt.py, train_pp116_softsingle_dual.py  (actual training code)

Exactly one contact sheet per cached object crop and requested visualization epoch;
all candidate parts in that crop appear as individual rows comparing lam0 vs lam4.
All candidate crops/parts/epochs are recorded in CSV, not only focus classes.
Uses GT ONLY AFTER computing patch weights; does not train/change files.

IMPORTANT: crop-box+recorded-flip reconstruction cannot rule out unrecorded
spatial transforms in cached DINO tokens. Image panels diagnose, not prove,
image-to-feature alignment. GT ignore 255 is NOT synonymous with background.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from html import escape as html_escape
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import audit_pp116_gt_weighted_purity as au

C, N, D = 116, 1024, 768

def write_csv(path, columns):
    f = open(path, 'w', newline='', encoding='utf-8-sig')
    w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
    w.writeheader()
    return f, w


def rgb_file(root, image_id):
    stem = Path(str(image_id)).stem
    matches = [root / (stem + ext) for ext in ('.jpg','.jpeg','.png','.JPG') if (root / (stem + ext)).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected 1 original RGB for {image_id} under {root}, found {matches}')
    return matches[0]


def crop_geometry(a, rgb, labels, flip_mode):
    h,w = labels.shape
    if rgb.shape[:2] != (h,w):
        raise ValueError(f'Image/GT dimensions mismatch {rgb.shape} vs {labels.shape} for {a["image_id"]}')
    raw_box = au.get_box(a)
    if np.all(raw_box >= 0) and float(raw_box.max()) <= 1.01 and max(h,w)>8:
        raise ValueError(f'Crop box looks normalized not pixel XYXY: {raw_box}')
    x0,y0,x1,y1 = [int(round(float(q))) for q in raw_box]
    x0,x1 = max(0,min(w,x0)),max(0,min(w,x1))
    y0,y1 = max(0,min(h,y0)),max(0,min(h,y1))
    if x1<=x0 or y1<=y0:
        raise ValueError(f'Invalid crop {raw_box} in {w}x{h}')
    doflip = au.get_flip(a,flip_mode)
    gt_crop = labels[y0:y1,x0:x1]
    rgb_crop = rgb[y0:y1,x0:x1]
    if doflip:
        gt_crop = np.ascontiguousarray(gt_crop[:,::-1])
        rgb_crop = np.ascontiguousarray(rgb_crop[:,::-1,:])
    gt32 = np.asarray(Image.fromarray(gt_crop.astype(np.int32),mode='I').resize((32,32),Image.Resampling.NEAREST),dtype=np.int32)
    return rgb_crop, gt_crop, gt32, (x0,y0,x1,y1), doflip


def sz32(x, side):
    return np.asarray(Image.fromarray(x).resize((side,side),Image.Resampling.NEAREST))


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(path).is_file(): return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def fit_full(frame, src, *, rect=None):
    side=frame.size[0]
    original=Image.fromarray(src,'RGB')
    if rect is not None:
        dr=ImageDraw.Draw(original)
        dr.rectangle(tuple(rect),outline=(0,255,40),width=max(2,min(src.shape[:2])//175))
    original.thumbnail((side,side), Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(side,side),(18,22,28))
    canvas.paste(original,((side-original.width)//2,(side-original.height)//2))
    return canvas


def tint(rgb, gt, pid=None):
    rgb=np.asarray(rgb, dtype=np.float32).copy()
    ign=(gt==255)
    other=(gt!=255) if pid is None else ((gt!=255)&(gt!=pid))
    rgb[ign]=.67*rgb[ign]+.33*np.array([95.,95.,100.])
    rgb[other]=.43*rgb[other]+.57*np.array([255.,164.,40.])
    if pid is not None:
        good=(gt==pid)
        rgb[good]=.25*rgb[good]+.75*np.array([24.,246.,68.])
    return np.uint8(np.clip(rgb,0,255))


def coverage(rgb, gt, fg, pid):
    out=np.asarray(rgb,dtype=np.float32).copy()*.40
    rules=[((gt==pid)&fg, (12,250,46)),
           ((gt==pid)&~fg, (250,24,24)),
           ((gt!=pid)&fg, (40,130,250))]
    for mask,color in rules:
        out[mask]=out[mask]*.16+np.asarray(color)*.84
    return np.uint8(np.clip(out,0,255))


def cmap_rgb(weight, vmax):
    from matplotlib import colormaps
    z = np.clip(weight / max(vmax,1e-12), 0,1)**.42
    return np.uint8(np.rint(colormaps['magma'](z)[...,:3]*255))


def overlay_heat(rgb, w32, scale):
    z=np.clip(w32/max(scale,1e-12),0,1)**.42
    hm=cmap_rgb(w32,scale).astype(np.float32)
    a=.88*z[...,None]
    return np.uint8(np.clip(rgb.astype(np.float32)*(1-a)+hm*a,0,255))


def text_fit(draw, xy, msg, *, f, fill, max_width):
    s=str(msg)
    while len(s)>2 and draw.textbbox(xy,s,font=f)[2]-xy[0] > max_width:
        s=s[:-4]+'...'
    draw.text(xy,s,font=f,fill=fill)


def save_sheet(path, original_rgb, crop_rgb, gt_native, gt32, fg32, bbox, names, ids,
               visual, stage, side, quality, image_id, crop_index, doflip):
    """One sheet for all K parts of one cached object crop (2 header rows)."""
    S=side
    top_h=34
    row_h=S+58
    width=4*S
    height=(2+len(ids))*row_h+top_h
    img=Image.new('RGB',(width,height),(246,247,249))
    dr=ImageDraw.Draw(img)
    bigfont=font(max(11,int(S*.086)))
    smallfont=font(max(10,int(S*.067)))
    dr.rectangle((0,0,width,top_h),fill=(24,37,50))
    text_fit(dr,(9,8),f'PP116 | ID {image_id} | crop {crop_index} | E{stage:03d} | K={len(ids)} | flip={int(doflip)}',
             f=bigfont,fill=(255,255,255),max_width=width-14)
    crop_img=Image.fromarray(crop_rgb,'RGB').resize((S,S),Image.Resampling.BICUBIC)
    rgb=np.asarray(crop_img)
    native_r=np.asarray(Image.fromarray(gt_native.astype(np.int32),mode='I').resize((S,S),Image.Resampling.NEAREST),dtype=np.int32)
    gt_r=sz32(gt32.astype(np.int32),S)
    fg_r=sz32(fg32.astype(np.uint8),S).astype(bool)
    # Heading row: full image with box; cropped RGB; pixel-native labels; predicted FG.
    full=fit_full(Image.new('RGB',(S,S)),original_rgb,rect=bbox)
    items=[(full,'Original RGB + box'),(crop_img,'Reconstructed RGB crop'),
           (Image.fromarray(tint(rgb,native_r)), 'Pixel-native GT: orange/gray'),
           (Image.fromarray(np.where(fg_r[...,None],.66*rgb+.34*np.array([25,255,255]),.32*rgb).clip(0,255).astype('uint8')),'Predicted foreground')]
    for j,(panel,title) in enumerate(items):
        y=top_h
        img.paste(panel,(j*S,y))
        text_fit(dr,(j*S+6,y+S+7),title,f=smallfont,fill=(35,45,55),max_width=S-9)
    # Second heading row: nearest neighbor GT, ignore, both flip assumptions on RGB.
    noflip=np.asarray(Image.fromarray(gt_native.astype(np.int32),mode='I'))
    other_gt=np.ascontiguousarray(noflip[:,::-1])
    gtalt=np.asarray(Image.fromarray(other_gt.astype(np.int32),mode='I').resize((S,S),Image.Resampling.NEAREST),dtype=np.int32)
    items2=[(Image.fromarray(tint(rgb,gt_r)), 'GT nearest to 32x32'),
            (Image.fromarray(np.where((gt_r==255)[...,None],.36*rgb+.64*np.array([146,45,190]),rgb).astype('uint8')),'GT ignore 255: purple'),
            (Image.fromarray(tint(rgb,native_r)), 'Applied GT orientation'),
            (Image.fromarray(tint(rgb,gtalt)), 'Opposite GT orientation')]
    for j,(panel,title) in enumerate(items2):
        y=top_h+row_h
        img.paste(panel,(j*S,y))
        text_fit(dr,(j*S+6,y+S+7),title,f=smallfont,fill=(35,45,55),max_width=S-9)
    for r,pid in enumerate(ids):
        y=top_h+(r+2)*row_h
        w0=visual['lam0',pid]['weight'].reshape(32,32)
        w4=visual['lam4',pid]['weight'].reshape(32,32)
        scale=max(float(w0.max()),float(w4.max()),1e-12)
        p0=visual['lam0',pid]['purity'];p4=visual['lam4',pid]['purity']
        s0=visual['lam0',pid]['support'];s4=visual['lam4',pid]['support']
        fb0=visual['lam0',pid]['fallback'];fb4=visual['lam4',pid]['fallback']
        panels=[tint(rgb,native_r,pid),coverage(rgb,gt_r,fg_r,pid),
                overlay_heat(rgb,sz32(w0,S),scale),overlay_heat(rgb,sz32(w4,S),scale)]
        titles=[f'{names[pid]} | GT-native', 'GT32 vs FG: green kept/red missed',
                f'lam0 P={p0:.3f} S={s0} F={fb0}', f'lam4 P={p4:.3f} S={s4} F={fb4}']
        for j,(panel,title) in enumerate(zip(panels,titles)):
            img.paste(Image.fromarray(panel,'RGB'),(j*S,y))
            text_fit(dr,(j*S+5,y+S+5),title,f=smallfont,fill=(30,40,50),max_width=S-8)
        n_gt=int((gt32==pid).sum());n_kept=int(((gt32==pid)&fg32).sum())
        text_fit(dr,(6,y+S+27),f'Target GT patches: {n_gt}, in FG: {n_kept};  Ignore is unlabelled, NOT necessarily BG',
                 f=smallfont,fill=(84,92,105),max_width=width-12)
        if r%2==0:
            dr.line((0,y-1,width,y-1),fill=(175,188,198),width=2)
    path.parent.mkdir(parents=True,exist_ok=True)
    img.save(path,format='JPEG',quality=quality,optimize=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--images-root',type=Path,default=Path('data/PascalPart116/images/train'))
    p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--run0',type=Path,default=Path('final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep'))
    p.add_argument('--run4',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--iou-csv',type=Path,default=Path('output/pp116_DIAGREL_comparison/per_part_epoch_lambda0_vs_lambda4.csv'))
    p.add_argument('--audit-csv',type=Path,default=Path('output/pp116_purity_full/per_crop_focus.csv'),help='If present cross-check same crop/part/epoch purity with prior audit')
    p.add_argument('--epochs',nargs='+',type=int,default=[0,10,30,100])
    p.add_argument('--viz-epochs',nargs='+',type=int,default=[100],help='Create ONE ALL-PART JPEG SHEET for every crop for each listed epoch')
    p.add_argument('--flip-mode',choices=('auto','none','all'),default='auto')
    p.add_argument('--soft-temperature',type=float,default=.05)
    p.add_argument('--side',type=int,default=208,help='Pixels per montage panel, 4 columns, one row per candidate part')
    p.add_argument('--jpeg-quality',type=int,default=82)
    p.add_argument('--min-target-patches-for-flag',type=int,default=3)
    p.add_argument('--low-purity-flag',type=float,default=.03)
    p.add_argument('--high-ignore-flag',type=float,default=.80)
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_full_allpart_visual_audit'))
    p.add_argument('--max-crops',type=int,default=0,help='0 ALL; >0 first N for smoke test')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--progress-every',type=int,default=100)
    args=p.parse_args()
    if args.self_test:
        return self_test()
    if not args.epochs or len(set(args.epochs))!=len(args.epochs) or any(not 0<=e<=100 for e in args.epochs):
        raise ValueError('epochs must be unique between 0 and 100')
    args.epochs=sorted(args.epochs)
    args.viz_epochs=sorted(set(args.viz_epochs))
    if not args.viz_epochs or not set(args.viz_epochs).issubset(args.epochs):raise ValueError('viz epochs must be subset of epochs')
    if args.side<100 or args.jpeg_quality<50 or args.jpeg_quality>95:raise ValueError('Bad panel size or JPEG quality')
    if args.soft_temperature<=0:raise ValueError('Invalid soft temperature')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):raise FileExistsError(f'Will not overwrite output: {args.out_dir}')
    for path in (args.cache,args.images_root,args.gt_root,args.text_bank,args.projector,args.model_config,args.classes_file,args.iou_csv):
        if not path.exists(): raise FileNotFoundError(path)

    import train_relproto_alignemt as src
    import train_pp116_softsingle_dual as train

    names=au.canonical_names(args.classes_file)
    ious=au.read_iou(args.iou_csv,names)
    for ep in args.epochs:
        for cid in range(C):
            if (ep,cid) not in ious:raise ValueError(f'IoU data missing for E{ep} part {cid}')
    dev=torch.device(args.device)
    if dev.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    raw,raw_names,_=src.load_raw_clip_bank(args.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if list(raw_names)!=names:raise ValueError('CLIP text class order mismatch')
    project,_=src.load_frozen_projector(project_root=Path.cwd(),config_path=args.model_config,weight_path=args.projector,device=dev)
    base=src.normalize_last(src.project_raw_clip_bank(raw,project,device=dev,batch_size=128).detach().to(dtype=torch.float32))
    del project,raw
    assert base.shape==(C,D)
    texts={}
    for arm,folder,lam in (('lam0',args.run0,0),('lam4',args.run4,4)):
        for ep in args.epochs:
            W=au.checkpoint_W(folder,ep,lam).to(dev)
            texts[arm,ep]=src.normalize_last(base.to(device=W.device, dtype=W.dtype)@W).detach()
    if 0 in args.epochs and not torch.equal(texts['lam0',0],texts['lam4',0]):raise AssertionError('E0 text different')
    anns=au.annotations(au.load_pt(args.cache))
    total=len(anns)
    if args.max_crops>0:anns=anns[:args.max_crops]
    if not anns:raise RuntimeError('No cached crops')
    lookup=au.GTLookup(args.gt_root)
    first=anns[0]
    lookup.crop32(first,args.flip_mode) # fail fast on GT location & type
    if not rgb_file(args.images_root,first['image_id']).is_file():raise FileNotFoundError('First RGB missing')

    old_focus={}
    if args.audit_csv.is_file():
        with args.audit_csv.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if int(r['epoch']) in args.epochs and int(r['crop_index']) < len(anns):
                    old_focus[(int(r['crop_index']),int(r['class_id']),r['arm'],int(r['epoch']))]=(float(r['weighted_gt_purity']),int(r['gt_fg_patch_count']))
        print('[CROSS-CHECK] loaded prior focus audit rows',len(old_focus),flush=True)

    args.out_dir.mkdir(parents=True)
    part_cols=['arm','epoch','crop_index','image_id','part_id','part','object','K','gt_part_patches','gt_part_in_fg','gt_fg_recall','fg_patches','fg_valid_gt_patches','fg_ignore_gt_patches','gt_ignore_patch_ratio','fallback','positive_margin_patches','selected_patches','selected_correct_patches','selected_ignore_patches','effective_patches','weight_entropy','weighted_gt_purity','weighted_ignore_mass','weighted_other_annotated_mass','conditioned_valid_purity','top_wrong_gt_part','top_wrong_mass','anchor_margin','max_relative_margin','mean_weighted_relative_margin','val_part_iou_pct','val_gt_valid','alignment_flag','visual_file']
    crop_cols=['crop_index','image_id','object','K','parts','fg_patches','gt_annotated_patches','gt_ignore_patches','annotated_gt_in_fg','gt_annotation_fg_recall','crop_box_xyxy','recorded_flip_metadata','applied_flip','visual_files']
    flag_cols=['arm','epoch','crop_index','image_id','part','gt_part_patches','gt_part_in_fg','weighted_gt_purity','weighted_ignore_mass','fallback','reason','visual_file']
    pf,pw=write_csv(args.out_dir/'per_crop_part_epoch.csv',part_cols)
    gf,gw=write_csv(args.out_dir/'per_crop_geometry.csv',crop_cols)
    ff,fw=write_csv(args.out_dir/'alignment_review_flags.csv',flag_cols)
    gallery=defaultdict(list)  # one link per (crop, candidate Part); never copy JPEGs
    bucket=defaultdict(lambda:defaultdict(float))
    image_bucket=defaultdict(lambda:defaultdict(float))
    object_bucket=defaultdict(lambda:defaultdict(float))
    confusion=defaultdict(lambda:np.zeros(117,dtype=np.float64))
    valid_images=0;multi=0;part_occurrences=0;confirmed=0;generated=0;ambig_flip=0;zero_annotated=0;source_checked=False
    try:
        with torch.no_grad():
            for i,a in enumerate(anns):
                ids=au.part_ids(a)
                if any(k not in a for k in ('image_id','cropaug_box_xyxy','cropaug_patch_tokens','pred_obj_mask_patch')):
                    raise KeyError(f'Missing cache keys in crop {i}')
                object_name=names[ids[0]].split("'s ",1)[0]
                if any(names[pid].split("'s ",1)[0]!=object_name for pid in ids):
                    raise ValueError(f'Mixed object part candidates in crop #{i}: {ids}')
                image_id=str(a['image_id'])
                with Image.open(rgb_file(args.images_root,image_id)) as im: rgb=np.asarray(im.convert('RGB'))
                gt=lookup.full(image_id)
                rgb_crop,gt_native,gt32,box,flipped=crop_geometry(a,rgb,gt,args.flip_mode)
                baseline,baseflip,wh,oldbox=lookup.crop32(a,args.flip_mode)
                if not np.array_equal(baseline.reshape(32,32),gt32) or baseflip!=flipped:
                    raise AssertionError(f'Geometry not identical to prior GT audit for crop {i}')
                tokens=torch.as_tensor(a['cropaug_patch_tokens'],device=dev,dtype=torch.float32)
                fg=torch.as_tensor(a['pred_obj_mask_patch'],device=dev,dtype=torch.bool).reshape(-1)
                if tokens.shape!=(N,D) or fg.shape!=(N,) or not bool(fg.any()):
                    raise ValueError(f'Invalid patch tokens or empty FG crop {i}')
                tokens=src.normalize_last(tokens)
                gt_flat=gt32.reshape(-1)
                gt_tensor=torch.as_tensor(np.where(gt_flat==255,C,gt_flat),device=dev,dtype=torch.long)
                gt_count=np.bincount(gt_flat[gt_flat!=255],minlength=C)
                fg_np=fg.cpu().numpy().reshape(32,32)
                gt_fg=np.bincount(gt_flat[(gt_flat!=255)&fg_np.reshape(-1)],minlength=C)
                valid_gt=int((gt_flat!=255).sum());ign=int((gt_flat==255).sum())
                annotated_fg=int(((gt_flat!=255)&fg_np.reshape(-1)).sum())
                flipped_known=any(k in a for k in ('cropaug_hflip','cropaug_flipped','cropaug_flip','hflip','is_flipped','flipped','flip','flip_direction'))
                ambig_flip+=int(not flipped_known)
                zero_annotated+=int(valid_gt==0)
                preview_files=[f'sheets/E{e:03d}/shard_{i//500:03d}/crop_{i:05d}_{image_id}.jpg' for e in args.viz_epochs]
                gw.writerow({'crop_index':i,'image_id':image_id,'object':object_name,'K':len(ids),'parts':' | '.join(names[j] for j in ids),
                            'fg_patches':int(fg_np.sum()),'gt_annotated_patches':valid_gt,'gt_ignore_patches':ign,
                            'annotated_gt_in_fg':annotated_fg,'gt_annotation_fg_recall':annotated_fg/valid_gt if valid_gt else math.nan,
                            'crop_box_xyxy':json.dumps(box),'recorded_flip_metadata':int(flipped_known),'applied_flip':int(flipped),
                            'visual_files':' | '.join(preview_files)})
                valid_images+=1;multi+=int(len(ids)>=2);part_occurrences+=len(ids)
                for pid in ids: gallery[pid].append((i,image_id,preview_files[-1]))
                visual={e:{} for e in args.viz_epochs}
                for ep in args.epochs:
                    for arm in ('lam0','lam4'):
                        local=texts[arm,ep][ids]
                        weights,selected,fallback=au.assignment(local,tokens,fg,args.soft_temperature,src.relative_scores)
                        if not source_checked and len(ids)>=2:
                            ref,diag=train.soft_relative_prototypes(local[None],tokens[None],fg[None],temperature=args.soft_temperature)
                            own=F.normalize(weights@tokens,dim=-1,eps=1e-12)
                            if (ref[0]-own).abs().max().item()>2e-5 or not torch.equal(fallback,diag['fallback'][0].bool()):
                                raise AssertionError('Soft Relative differs from actual trainer')
                            source_checked=True
                            print('[CHECK] Soft weights/prototypes/fallback match trainer',flush=True)
                        similarity=local@tokens.T
                        if len(ids)>1:
                            rel=src.relative_scores(similarity[None],fg[None])[0]
                        else:
                            rel=similarity.masked_fill(~fg[None],-torch.inf)
                        gt_weight=torch.zeros((len(ids),C+1),device=dev)
                        gt_weight.scatter_add_(1,gt_tensor[None].expand(len(ids),-1),weights)
                        mass=gt_weight.cpu().numpy()
                        if not np.allclose(mass.sum(-1),1,atol=2e-5):raise AssertionError('Mass lost')
                        pos=((rel>0)&fg[None]).sum(-1).cpu().numpy() if len(ids)>1 else np.zeros(len(ids),dtype=int)
                        select_count=selected.sum(-1).cpu().numpy()
                        selected_correct=[int((selected[j]& (gt_tensor==pid)).sum()) for j,pid in enumerate(ids)]
                        selected_ignore=(selected&(gt_tensor==C)[None,:]).sum(-1).cpu().numpy()
                        eff=(1/weights.square().sum(-1)).cpu().numpy()
                        entropy=(-(weights.clamp_min(1e-35)*weights.clamp_min(1e-35).log()).sum(-1)).cpu().numpy()
                        maxm=rel.max(-1).values.cpu().numpy()
                        expect=((weights * rel.masked_fill(~selected,0)).sum(-1)).cpu().numpy()
                        for j,pid in enumerate(ids):
                            n_gt=int(gt_count[pid]);nfg=int(gt_fg[pid])
                            good=float(mass[j,pid]);ignore=float(mass[j,C]);bad=max(0,1-good-ignore)
                            wrong=mass[j,:C].copy();wrong[pid]=-1
                            wid=int(wrong.argmax());wm=max(0.,float(wrong[wid]))
                            pu=good/(1-ignore) if ignore<1-1e-9 else math.nan
                            fk=int(bool(fallback[j])) if len(ids)>1 else 0
                            valid,ious_by_arm=ious[ep,pid]
                            iou=ious_by_arm[arm]
                            reason=[]
                            if n_gt==0: reason.append('GT_ABSENT_IN_CROP')
                            elif nfg==0: reason.append('GT_PRESENT_BUT_NOT_IN_PRED_FG')
                            elif n_gt>=args.min_target_patches_for_flag and good < args.low_purity_flag:
                                reason.append('GT_AVAILABLE_BUT_LOW_WEIGHT')
                            if ignore>args.high_ignore_flag: reason.append('MOST_WEIGHT_ON_GT_IGNORE')
                            # Missing recorded flip metadata is a dataset-level limitation,
                            # not proof of incorrect geometry; store in per_crop_geometry instead.
                            key=(arm,ep,pid)
                            st=bucket[key]
                            st['n']+=1;st['gt_present']+=int(n_gt>0);st['gt_fg_present']+=int(nfg>0)
                            st['gt_patches']+=n_gt;st['gt_fg_patches']+=nfg
                            st['sum_purity']+=good;st['sum_present_purity']+=good*int(n_gt>0)
                            st['sum_presentfg_purity']+=good*int(nfg>0)
                            st['ignore_mass']+=ignore;st['wrong_mass']+=bad;st['fallback']+=fk
                            st['multi']+=int(len(ids)>1);st['selected']+=int(select_count[j]);st['selected_correct']+=selected_correct[j]
                            st['eff']+=float(eff[j]);st['high_ignore']+=int(ignore>args.high_ignore_flag)
                            st['low_purity_with_available_gt']+=int(nfg>0 and good<args.low_purity_flag)
                            if math.isfinite(pu):st['valid_purity']+=pu;st['valid_purity_count']+=1
                            confusion[key]+=mass[j,:]
                            for acc,key2 in ((image_bucket,(arm,ep,image_id)),(object_bucket,(arm,ep,object_name))):
                                ss=acc[key2];ss['n']+=1;ss['purity']+=good;ss['ignore']+=ignore
                                ss['gtfg']+=int(nfg>0);ss['fallback']+=fk;ss['multi']+=int(len(ids)>1)
                            if ep in visual:
                                visual[ep][arm,pid]={'weight':weights[j].cpu().numpy().copy(), 'purity':good,
                                                    'support':int(select_count[j]),'fallback':fk}
                            path=preview_files[args.viz_epochs.index(ep)] if ep in visual else ''
                            row={'arm':arm,'epoch':ep,'crop_index':i,'image_id':image_id,'part_id':pid,'part':names[pid],
                                'object':object_name,'K':len(ids),'gt_part_patches':n_gt,'gt_part_in_fg':nfg,
                                'gt_fg_recall':nfg/n_gt if n_gt else math.nan,'fg_patches':int(fg_np.sum()),
                                'fg_valid_gt_patches':annotated_fg,'fg_ignore_gt_patches':int(fg_np.sum())-annotated_fg,
                                'gt_ignore_patch_ratio':ign/N,'fallback':fk,'positive_margin_patches':int(pos[j]),
                                'selected_patches':int(select_count[j]),'selected_correct_patches':selected_correct[j],
                                'selected_ignore_patches':int(selected_ignore[j]),'effective_patches':float(eff[j]),
                                'weight_entropy':float(entropy[j]),'weighted_gt_purity':good,
                                'weighted_ignore_mass':ignore,'weighted_other_annotated_mass':bad,'conditioned_valid_purity':pu,
                                'top_wrong_gt_part':names[wid] if wm>1e-7 else '', 'top_wrong_mass':wm,
                                'anchor_margin':float(maxm[j]),'max_relative_margin':float(maxm[j]),
                                'mean_weighted_relative_margin':float(expect[j]),'val_part_iou_pct':iou if valid else math.nan,
                                'val_gt_valid':int(valid),'alignment_flag':' | '.join(reason),'visual_file':path}
                            pw.writerow(row)
                            if reason:
                                fw.writerow({'arm':arm,'epoch':ep,'crop_index':i,'image_id':image_id,'part':names[pid],
                                            'gt_part_patches':n_gt,'gt_part_in_fg':nfg,'weighted_gt_purity':good,
                                            'weighted_ignore_mass':ignore,'fallback':fk,'reason':' | '.join(reason),
                                            'visual_file':path})
                            old=old_focus.get((i,pid,arm,ep))
                            if old is not None:
                                if abs(old[0]-good)>3e-4 or old[1]!=nfg:
                                    raise AssertionError(f'Prior GT purity disagreement crop={i}, part={names[pid]}, {arm},E{ep}: prior={old}, now={(good,nfg)}')
                                confirmed+=1
                for ep in args.viz_epochs:
                    rel=preview_files[args.viz_epochs.index(ep)]
                    save_sheet(args.out_dir/rel,rgb,rgb_crop,gt_native,gt32,fg_np,box,names,ids,visual[ep],ep,
                               args.side,args.jpeg_quality,image_id,i,flipped)
                    generated+=1
                if (i+1)%args.progress_every==0 or i+1==len(anns):
                    print(f'[PROGRESS] {i+1}/{len(anns)} crops; images={generated}; part-occurrences={part_occurrences}; prior checks={confirmed}',flush=True)
                    pf.flush();gf.flush();ff.flush()
    finally:
        pf.close();gf.close();ff.close()
    if not source_checked:raise RuntimeError('No K>=2 crop for trainer verification')

    # Category-level and GT-confusion aggregates are only published after full run.
    sum_cols=['arm','epoch','part_id','part','object','n_candidate_crops','n_gt_present_crops','n_gt_part_in_fg_crops',
              'total_gt_patches','total_gt_patches_in_fg','gt_part_fg_recall_micro','mean_purity_all_crops',
              'mean_purity_gt_present','mean_purity_gt_available_in_fg','mean_valid_conditioned_purity',
              'mean_ignore_mass','mean_other_annotated_mass','fallback_rate_multi','mean_selected_patches',
              'selected_correct_patch_fraction','mean_effective_patches','fraction_high_ignore','fraction_low_purity_with_gt_fg',
              'val_iou_pct','val_gt_valid']
    sf,sw=write_csv(args.out_dir/'per_part_epoch_summary.csv',sum_cols)
    summaries={}
    for (arm,ep,pid),st in sorted(bucket.items()):
        n=st['n'];valid,ious_by_arm=ious[ep,pid];iou=ious_by_arm[arm]
        r={'arm':arm,'epoch':ep,'part_id':pid,'part':names[pid],'object':names[pid].split("'s ",1)[0],
           'n_candidate_crops':int(n),'n_gt_present_crops':int(st['gt_present']),
           'n_gt_part_in_fg_crops':int(st['gt_fg_present']), 'total_gt_patches':int(st['gt_patches']),
           'total_gt_patches_in_fg':int(st['gt_fg_patches']),
           'gt_part_fg_recall_micro':st['gt_fg_patches']/st['gt_patches'] if st['gt_patches'] else math.nan,
           'mean_purity_all_crops':st['sum_purity']/n,
           'mean_purity_gt_present':st['sum_present_purity']/st['gt_present'] if st['gt_present'] else math.nan,
           'mean_purity_gt_available_in_fg':st['sum_presentfg_purity']/st['gt_fg_present'] if st['gt_fg_present'] else math.nan,
           'mean_valid_conditioned_purity':st['valid_purity']/st['valid_purity_count'] if st['valid_purity_count'] else math.nan,
           'mean_ignore_mass':st['ignore_mass']/n,'mean_other_annotated_mass':st['wrong_mass']/n,
           'fallback_rate_multi':st['fallback']/st['multi'] if st['multi'] else math.nan,
           'mean_selected_patches':st['selected']/n,
           'selected_correct_patch_fraction':st['selected_correct']/st['selected'] if st['selected'] else math.nan,
           'mean_effective_patches':st['eff']/n,
           'fraction_high_ignore':st['high_ignore']/n,
           'fraction_low_purity_with_gt_fg':st['low_purity_with_available_gt']/n,
           'val_iou_pct':iou if valid else math.nan,'val_gt_valid':int(valid)}
        summaries[arm,ep,pid]=r;sw.writerow(r)
    sf.close()
    conf_cols=['arm','epoch','pseudo_part_id','pseudo_part','gt_label_id','gt_label','sum_weight','mean_weight_per_crop','n_candidate_crops']
    cf,cw=write_csv(args.out_dir/'gt_weight_confusion.csv',conf_cols)
    for (arm,ep,pid),v in sorted(confusion.items()):
        n=bucket[arm,ep,pid]['n']
        for idx,mass in enumerate(v):
            if mass>1e-8:
                cw.writerow({'arm':arm,'epoch':ep,'pseudo_part_id':pid,'pseudo_part':names[pid],
                             'gt_label_id':idx if idx<C else 255, 'gt_label':names[idx] if idx<C else 'IGNORE_255',
                             'sum_weight':float(mass),'mean_weight_per_crop':float(mass/n),'n_candidate_crops':int(n)})
    cf.close()
    agg_cols=['arm','epoch','group','n_part_occurrences','mean_purity','mean_ignore_mass','fraction_part_gt_available_in_fg','fallback_rate_multi']
    for fpath, buckets in ((args.out_dir/'per_image_epoch_summary.csv',image_bucket),
                           (args.out_dir/'per_object_epoch_summary.csv',object_bucket)):
        f,w=write_csv(fpath,agg_cols)
        for (arm,ep,group),b in sorted(buckets.items()):
            n=b['n'];w.writerow({'arm':arm,'epoch':ep,'group':group,'n_part_occurrences':int(n),
                                'mean_purity':b['purity']/n,'mean_ignore_mass':b['ignore']/n,
                                'fraction_part_gt_available_in_fg':b['gtfg']/n,
                                'fallback_rate_multi':b['fallback']/b['multi'] if b['multi'] else math.nan})
        f.close()
    try:
        from scipy.stats import spearmanr
        correlations=[]
        for ep in args.epochs:
            for arm in ('lam0','lam4'):
                group=[r for (a,e,p),r in summaries.items() if a==arm and e==ep and r['val_gt_valid']
                       and r['n_gt_present_crops']>=10]
                for metric in ('mean_purity_all_crops','mean_purity_gt_present',
                               'gt_part_fg_recall_micro','mean_ignore_mass','fallback_rate_multi'):
                    pairs=[(r[metric],r['val_iou_pct']) for r in group if np.isfinite(r[metric])]
                    rho=float(spearmanr(*np.asarray(pairs).T).statistic) if len(pairs)>=5 else math.nan
                    correlations.append({'arm':arm,'epoch':ep,'metric':metric,'n_parts':len(pairs),'spearman_vs_iou':rho})
            group=[]
            for pid in range(C):
                a=summaries.get(('lam0',ep,pid));b=summaries.get(('lam4',ep,pid))
                if a and b and a['val_gt_valid'] and a['n_gt_present_crops']>=10:
                    group.append((a,b))
            xs=[b['mean_purity_all_crops']-a['mean_purity_all_crops'] for a,b in group]
            ys=[b['val_iou_pct']-a['val_iou_pct'] for a,b in group]
            rho=float(spearmanr(xs,ys).statistic) if len(group)>=5 else math.nan
            correlations.append({'arm':'lam4-minus-lam0','epoch':ep,'metric':'delta_purity_vs_delta_IoU',
                                 'n_parts':len(group),'spearman_vs_iou':rho})
        rf,rw=write_csv(args.out_dir/'correlations.csv',['arm','epoch','metric','n_parts','spearman_vs_iou'])
        rw.writerows(correlations);rf.close()
    except ImportError:
        correlations=[]
        print('[WARN] scipy absent: correlations.csv not generated',flush=True)
    stage=max(args.epochs)
    rank=[]
    for pid in range(C):
        a=summaries.get(('lam0',stage,pid));b=summaries.get(('lam4',stage,pid))
        if a is not None and b is not None:
            rank.append({'part':names[pid],'class_id':pid,'lambda0_iou':a['val_iou_pct'],
                         'lambda4_iou':b['val_iou_pct'],'delta_iou':b['val_iou_pct']-a['val_iou_pct'],
                         'lambda0_purity':a['mean_purity_all_crops'],'lambda4_purity':b['mean_purity_all_crops'],
                         'delta_purity':b['mean_purity_all_crops']-a['mean_purity_all_crops'],
                         'GT_fg_recall':b['gt_part_fg_recall_micro'],
                         'ignore_mass_lambda4':b['mean_ignore_mass'],'support':int(b['n_candidate_crops'])})
    rf,rw=write_csv(args.out_dir/f'ranked_parts_E{stage:03d}.csv',
                    ['part','class_id','lambda0_iou','lambda4_iou','delta_iou','lambda0_purity','lambda4_purity','delta_purity',
                     'GT_fg_recall','ignore_mass_lambda4','support'])
    rw.writerows(sorted(rank,key=lambda r:(r['lambda4_purity'],r['part'])));rf.close()
    expected=len(anns)*len(args.viz_epochs)
    if generated!=expected:raise AssertionError(f'Image count mismatch {generated} != {expected}')
    # HTML galleries: browse ALL crops containing each candidate part without JPEG duplication.
    parts_dir=args.out_dir/'part_galleries'
    parts_dir.mkdir(exist_ok=True)
    links=[]
    for pid,examples in sorted(gallery.items()):
        slug=re.sub(r'[^a-z0-9]+','_',names[pid].lower()).strip('_')
        page=f'part_{pid:03d}_{slug}.html'
        cards=[]
        for i,image_id,path in examples:
            href='../'+path
            cards.append(f'<a class="tile" href="{html_escape(href)}" target="_blank">'
                         f'<img src="{html_escape(href)}" loading="lazy" alt="{html_escape(names[pid])} crop {i}">'
                         f'<div>crop #{i} | {html_escape(image_id)}</div></a>')
        doc=('<!doctype html><html lang="en"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1">'
             f'<title>{html_escape(names[pid])} | PP116</title>'
             '<style>body{font:14px sans-serif;margin:20px;background:#f6f8fb;color:#182534}'
             '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}'
             '.tile{display:block;background:white;padding:8px;border:1px solid #ddd;border-radius:6px;color:#182534;text-decoration:none}'
             '.tile img{width:100%;height:180px;object-fit:contain;background:#f1f1f1}'
             '</style><body><p><a href="../index.html">All parts</a></p>'
             f'<h2>{html_escape(names[pid])} — {len(examples)} cached crops</h2>'
             '<p>Click thumbnail to open the complete multi-Part montage (the page uses the last requested visualization epoch). '
             'GT ignore 255 is not confirmed background. The geometry has not been independently verified against DINO token transforms.</p>'
             '<div class="grid">'+''.join(cards)+'</div></body></html>')
        (parts_dir/page).write_text(doc,encoding='utf-8')
        links.append(f'<li><a href="part_galleries/{page}">{html_escape(names[pid])}</a> '
                     f'({len(examples)} cached crops)</li>')
    (args.out_dir/'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>PP116 full part galleries</title>'
        '<body style="font:15px sans-serif;margin:30px">'
        '<h1>PP116 — ALL cached crops, ALL image-level candidate parts</h1>'
        '<p>Each Part gallery links to full crop sheets, with lambda0/lambda4 heatmaps; '
        'one JPEG per crop per requested visualization epoch.</p><ul>'
        +''.join(links)+'</ul></body></html>',encoding='utf-8')

    protocol={'complete':True,'crop_rows':len(anns),'crop_records_total':total,'candidate_part_occurrences':part_occurrences,
              'multi_part_crops':multi,'visualization_epoch_count':len(args.viz_epochs),'visualization_image_count':generated,
              'expected_visualization_images':expected,'epochs':args.epochs,'viz_epochs':args.viz_epochs,
              'arms':['lam0','lam4'],'cache':str(args.cache),'images_root':str(args.images_root),
              'gt_root':str(args.gt_root),'flip_mode':args.flip_mode,'crops_without_recorded_flip_metadata':ambig_flip,
              'crops_with_zero_annotated_GT':zero_annotated,'cross_checked_prior_purity_rows':confirmed,
              'soft_temperature':args.soft_temperature,'GT_only_for_offline_audit':True,
              'visual_format':'one JPEG per cached crop per viz epoch, ALL candidate parts rows, lam0/lam4 columns',
              'GT_geometry':'original pixel xyxy and recorded flip; NEAREST 32x32; matches earlier audit algorithm',
              'IMPORTANT_limitations':['Neither overlap nor matching GT geometry proves cached DINO patch tokens are spatially aligned; unrecorded augments are unknown',
                                       'GT=255 is IGNORE/unlabelled, NOT necessarily background',
                                       'Part label candidates are image-level; may be absent within individual crops',
                                       'Nearest-neighbor GT-to-patch approximation is resolution sensitive']}
    (args.out_dir/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.out_dir/'REPORT.md').open('w',encoding='utf-8') as f:
        f.write(f'# Full PP116 geometry + Soft Relative visual audit\n\n')
        f.write(f'- Audited cached crops: **{len(anns)}** / {total}; candidate part occurrences: **{part_occurrences}**\n')
        f.write(f'- Per-crop ALL-PART visual sheets: **{generated}** (E{args.viz_epochs})\n')
        f.write(f'- Preflight source-prototype consistency checked: **{source_checked}**\n')
        f.write(f'- Prior focus Purity comparisons passed: **{confirmed}**\n')
        f.write(f'- Without recorded flip metadata: **{ambig_flip}** (not proof that no flip occurred)\n')
        f.write(f'- All-ignore GT crop grids: **{zero_annotated}**\n\n')
        f.write('## Read first\n\n')
        f.write('`per_crop_part_epoch.csv`, `per_part_epoch_summary.csv`, `gt_weight_confusion.csv`, `alignment_review_flags.csv`, ')
        f.write(f'`ranked_parts_E{stage:03d}.csv`, `correlations.csv`, `sheets/`, `index.html` (browse all parts).\n\n')
        f.write('**Caution:** GT Ignore 255 is not necessarily background. Visual geometry does not independently verify cached DINO token orientation. ') 
        f.write('No part-level GT data enter Soft Relative scoring/training; this is privileged offline diagnostics only.\n')
    print(f'[DONE] ALL {len(anns)}/{total} crops; {part_occurrences} candidate part occurrences; {generated} image sheets',flush=True)
    print(f'[RESULT] {args.out_dir}',flush=True)


def self_test():
    from tempfile import TemporaryDirectory
    names=[f"mock's part{i}" for i in range(C)]
    rgb=np.zeros((128,190,3),dtype=np.uint8)+175
    gt=np.zeros((128,190),dtype=np.int32)+255
    gt[15:100,22:104]=0
    gt[28:76,104:147]=1
    a={'image_id':'synthetic','cropaug_box_xyxy':[15,9,165,115],'part_category_id':[0,1]}
    crop,gtcrop,gt32,box,flip=crop_geometry(a,rgb,gt,'none')
    assert crop.ndim==3 and gt32.shape==(32,32) and 0 in gt32 and 1 in gt32
    assert np.array_equal(tint(np.uint8(rgb[:32,:32]),np.zeros((32,32),dtype=np.int32),0).shape,(32,32,3))
    vis={}
    for arm in ('lam0','lam4'):
        for pid in (0,1):
            w=np.zeros(1024,dtype=np.float32);w[100 if pid==0 else 400]=1
            vis[arm,pid]={'weight':w,'purity':float(pid==0),'support':1,'fallback':0}
    fg=np.zeros((32,32),dtype=bool);fg[2:25,2:29]=True
    with TemporaryDirectory() as td:
        p=Path(td)/'test.jpg'
        save_sheet(p,rgb,crop,gtcrop,gt32,fg,box,names,[0,1],vis,100,112,76,'synthetic',0,False)
        assert p.is_file() and p.stat().st_size>6000
        with Image.open(p) as img:assert img.width==448 and img.height>500
    print('SELF_TEST_PASS: native/nearest GT, montage ALL parts, lambda0/4 heatmap, JPEG write')


if __name__=='__main__':
    main()

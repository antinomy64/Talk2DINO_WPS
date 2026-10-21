#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess VERIFIED Ours E100 original triptych PNGs, adding per-panel legends.

Reads only: raw Ours prediction ID arrays, official GT label PNGs, original
triptych PNGs, and class taxonomy. NEVER runs inference or changes original panels.
For each class, extracts the exact palette RGB from pixels already displayed in
GT/Ours panel; if alpha<1, inverts the known blend with original image.

Original image protocol: three W-wide panels with 8px gaps; 35px header,
written by ours_val_triptych_capture.py. Output is new PNG, original top area
is pasted unchanged with pixel-perfect equality.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['pp116','ade234'])
    p.add_argument('--capture-root',type=Path,default=None,
                   help='Successful original visualization output, including triptychs/, raw_prediction_ids/ and protocol.json')
    p.add_argument('--gt-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0: all; otherwise first N')
    p.add_argument('--ids',nargs='*',default=None)
    p.add_argument('--names-json',type=Path,default=None,help='optional list of 116/234 class names')
    p.add_argument('--font-size',type=int,default=14)
    p.add_argument('--max-overwrite',action='store_true',help='allow rewriting ONLY new legend outputs')
    return p.parse_args()


def class_names(ds, override):
    want=116 if ds=='pp116' else 234
    if override:
        names=json.loads(override.read_text(encoding='utf-8'))
    else:
        paths=([Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py')]
               if ds=='pp116' else
               [Path('ade20kpart234_taxonomy.py'),
                Path('src/open_vocabulary_segmentation/segmentation/datasets/ade20kpart234.py')])
        names=None
        for path in paths:
            if not path.is_file():continue
            spec=importlib.util.spec_from_file_location('_ovps_legend_names_'+ds,path)
            mod=importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            for key in ('PART_CLASSES','CLASSES'):
                value=getattr(mod,key,None)
                if value is not None and len(value)==want:
                    names=list(value);break
            if names is not None:break
        if names is None:raise FileNotFoundError(f'Cannot find canonical {ds} class taxonomy. Supply --names-json')
    names=[str(x) for x in names]
    if len(names)!=want or len(set(names))!=want:
        raise ValueError(f'Expected {want} unique class names, received {len(names)}')
    return names


def font(size,bold=False):
    candidates=(['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','DejaVuSans-Bold.ttf'] if bold else
                ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','DejaVuSans.ttf'])
    for f in candidates:
        try:return ImageFont.truetype(f,size)
        except OSError:pass
    return ImageFont.load_default()


def load_mask(path):
    if path.suffix=='.npy':
        a=np.load(path,allow_pickle=False)
    else:
        with Image.open(path) as im:a=np.asarray(im)
    if a.ndim!=2:raise ValueError(f'Bad mask (not HxW): {path}: {a.shape}')
    return a.astype(np.int64,copy=False)


def gt_index(root):
    if not root.is_dir():raise FileNotFoundError(f'GT root does not exist: {root.resolve()}')
    files=sorted(p for p in root.rglob('*.png') if p.is_file())
    if not files:raise RuntimeError(f'No GT PNG files under {root.resolve()}')
    result={}
    for p in files:
        if p.stem in result:
            raise RuntimeError(f'Duplicate GT file stem {p.stem!r}: {result[p.stem]} vs {p}')
        result[p.stem]=p
    return result


def palette_from_old_pixels(image, gt, pred, alpha, nclasses):
    """Exact same image colors. Blend inversion uses actual original input panel."""
    hh,ww=gt.shape
    im=np.asarray(image,dtype=np.uint8)
    if im.shape!=(hh+35,3*ww+16,3):
        raise ValueError(f'Original triptych geometry changed: triptych={im.shape}, GT={gt.shape}; '
                         'expected 35px header and two 8px gaps')
    valid=(gt>=0)&(gt<nclasses)
    if not valid.any():return {},valid
    # Each class' displayed color recovered from its panel's valid pixels.
    pal={}
    left=im[35:35+hh,0:ww].astype(np.float32)
    gtrgb=im[35:35+hh,ww+8:2*ww+8].astype(np.float32)
    oursrgb=im[35:35+hh,2*ww+16:3*ww+16].astype(np.float32)
    for ids,mask,panel in ((np.unique(gt[valid]),gt,gtrgb),
                           (np.unique(pred[valid]),pred,oursrgb)):
        for i in ids:
            label=int(i)
            if label in pal:continue
            where=np.flatnonzero(valid&(mask==label))
            if len(where)==0:continue
            sel=where[np.linspace(0,len(where)-1,min(128,len(where)),dtype=np.int64)]
            rgb=panel.reshape(-1,3)[sel]
            bg=left.reshape(-1,3)[sel]
            if alpha<0.01:
                raise ValueError('Original alpha too small to recover palette colors')
            colors=np.rint(np.median((rgb-(1-alpha)*bg)/alpha,axis=0))
            pal[label]=tuple(int(x) for x in np.clip(colors,0,255))
    return pal,valid


def label_width(draw,txt,ft):
    x=draw.textbbox((0,0),txt,font=ft)
    return x[2]-x[0]


def render_legend(w,present,palette,names,title,font_size):
    # One column: robust to long ADE234 class names and narrow images.
    title_ft=font(max(14,font_size+2),bold=True)
    ft=font(font_size)
    square=14
    gap=6
    height_line=24
    x0=8
    text_x=x0+square+8
    max_text=w-text_x-8
    entries=[]
    for idx,n in present:
        label=f'{idx}  {names[idx]}'
        # Wrap by rendered width rather than rough character count.
        if label_width(ImageDraw.Draw(Image.new('RGB',(1,1))),label,ft)<=max_text:
            entries.append((idx,[label]));continue
        tokens=label.split(' '); lines=[];line=''
        draw=ImageDraw.Draw(Image.new('RGB',(1,1)))
        for token in tokens:
            trial=(line+' '+token).strip()
            if line and label_width(draw,trial,ft)>max_text:
                lines.append(line);line=token
            else:line=trial
        if line:lines.append(line)
        if not lines:lines=[label]
        entries.append((idx,lines))
    height=35+sum(height_line*len(lines)+gap for _,lines in entries)+7
    output=Image.new('RGB',(w,max(height,50)),(255,255,255))
    d=ImageDraw.Draw(output)
    d.line((0,0,w,0),fill=(213,218,225),width=1)
    d.text((x0,7),f'{title}  ({len(entries)} parts)',fill=(25,30,40),font=title_ft)
    y=35
    for idx,lines in entries:
        c=palette.get(idx)
        if c is None:raise RuntimeError(f'No sample color for class {idx}')
        d.rectangle((x0,y+3,x0+square,y+3+square),fill=c,outline=(50,50,50),width=1)
        for j,line in enumerate(lines):
            d.text((text_x,y+j*height_line),line,fill=(23,27,35),font=ft)
        y+=height_line*len(lines)+gap
    return output


def main():
    args=parse_args()
    ds=args.dataset
    nclasses=116 if ds=='pp116' else 234
    root=args.capture_root or Path(f'output/ours_e100_val_triptych_{ds}')
    src_dir=root/'triptychs'
    pred_dir=root/'raw_prediction_ids'
    if not src_dir.is_dir() or not pred_dir.is_dir():
        raise FileNotFoundError(f'Expected EXISTING verified triptychs and raw_prediction_ids under {root.resolve()}; '
                                'do not rerun inference merely to append legends')
    protocol=root/'protocol.json'
    if not protocol.is_file():raise FileNotFoundError(f'Missing verification protocol: {protocol}')
    meta=json.loads(protocol.read_text(encoding='utf-8'))
    if meta.get('dataset')!=ds or meta.get('n_classes')!=nclasses:
        raise RuntimeError(f'Unexpected capture protocol: {meta}')
    expected=36.75 if ds=='pp116' else 8.98
    if abs(float(meta.get('verified_miou',-1))-expected)>0.011:
        raise RuntimeError(f'Existing triptychs are NOT verified Ours {ds} ({expected} mIoU): {meta}')
    alpha=float(meta.get('alpha',1.0))
    print(f'[VERIFIED] {ds} mIoU={meta["verified_miou"]} alpha={alpha}; original triptych pixels will be preserved',flush=True)
    names=class_names(ds,args.names_json)
    gt_files=gt_index(args.gt_root)
    pngs=sorted(src_dir.glob('*.png'))
    if args.ids is not None:
        wanted=set(args.ids)
        pngs=[p for p in pngs if p.stem in wanted]
        missing_ids=wanted-{p.stem for p in pngs}
        if missing_ids:raise FileNotFoundError(f'Requested image IDs missing from verified triptychs: {sorted(missing_ids)[:10]}')
    if not pngs:raise RuntimeError(f'No original triptychs in {src_dir}')
    if args.ids is None and args.limit<=0 and len(pngs)!=int(meta.get('n_images',0)):
        raise RuntimeError(f'Expected {meta.get("n_images")} original triptychs from verified protocol, found {len(pngs)}')
    if args.limit>0:pngs=pngs[:args.limit]
    missing_gt=[p.stem for p in pngs if p.stem not in gt_files]
    missing_pred=[p.stem for p in pngs if not (pred_dir/(p.stem+'.npy')).is_file()]
    if missing_gt or missing_pred:
        raise FileNotFoundError(f'Cannot safely label {len(pngs)} original triptychs. '
                                f'Missing GT ({len(missing_gt)}): {missing_gt[:5]}; '
                                f'Missing predictions ({len(missing_pred)}): {missing_pred[:5]}. '
                                f'Check --gt-root and {pred_dir}')
    dst_dir=args.out_dir/'triptychs'
    if dst_dir.exists() and any(dst_dir.iterdir()) and not args.max_overwrite:
        raise FileExistsError(f'Output already contains images: {dst_dir}; choose a new --out-dir')
    dst_dir.mkdir(parents=True,exist_ok=True)
    result=[]
    for k,p in enumerate(pngs,1):
        stem=p.stem
        with Image.open(p) as original: trip=original.convert('RGB')
        gt=load_mask(gt_files[stem]);pred=load_mask(pred_dir/(stem+'.npy'))
        if gt.shape!=pred.shape:
            raise ValueError(f'GT/pred mismatch for {stem}: {gt.shape}, {pred.shape}')
        valid=(gt>=0)&(gt<nclasses)
        if (pred[valid]<0).any() or (pred[valid]>=nclasses).any():
            raise ValueError(f'Out-of-range Ours predicted IDs for {stem}')
        palette,valid=palette_from_old_pixels(trip,gt,pred,alpha,nclasses)
        gt_ids,gt_ct=np.unique(gt[valid],return_counts=True)
        pr_ids,pr_ct=np.unique(pred[valid],return_counts=True)
        gtr=[(int(i),int(c)) for i,c in zip(gt_ids,gt_ct)]
        prr=[(int(i),int(c)) for i,c in zip(pr_ids,pr_ct)]
        gtr.sort(key=lambda e:(-e[1],e[0]))
        prr.sort(key=lambda e:(-e[1],e[0]))
        w=gt.shape[1]
        a=render_legend(w,gtr,palette,names,'GT legend',args.font_size)
        b=render_legend(w,prr,palette,names,'Ours legend',args.font_size)
        # Paste EXACT original image at x=0,y=0 (no recomputation of its masks).
        out=Image.new('RGB',(trip.width,trip.height+max(a.height,b.height)+8),(255,255,255))
        out.paste(trip,(0,0))
        out.paste(a,(w+8,trip.height+4))
        out.paste(b,(2*w+16,trip.height+4))
        outpath=dst_dir/(stem+'.png')
        out.save(outpath,compress_level=4)
        result.append({'image_id':stem,'GT_part_count':len(gtr),'Ours_part_count':len(prr),
                       'valid_pixels':int(valid.sum()),'file':str(outpath)})
        if k%100==0:print(f'[LEGEND] {k}/{len(pngs)}',flush=True)
    with (args.out_dir/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=result[0].keys());wr.writeheader();wr.writerows(result)
    print(f'[SUCCESS] {len(result)} images with GT/Ours legend -> {dst_dir}',flush=True)


if __name__=='__main__':main()

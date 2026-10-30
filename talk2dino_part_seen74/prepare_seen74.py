#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter ORIGINAL Talk2DINO-Part pre-extracted (image, part text) training pairs.

No GT pixel/box/point supervision; no change to Talk2DINO training objective.
PP116 seen=74 / unseen=42, unseen objects: bird/car/dog/motorbike/sheep.
Annotations may omit category IDs: resolve identity via exact raw CLIP prompt-mean
bank only when features are demonstrably the same. Fail closed otherwise.
"""
from __future__ import annotations
import argparse, json, math, os, re, tempfile
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F

UNSEEN_OBJECTS=frozenset({'bird','car','dog','motorbike','sheep'})

def load(path):
    try: return torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError: return torch.load(str(path),map_location='cpu')

def get_split(bank):
    if not isinstance(bank,dict) or 'features' not in bank or 'class_names' not in bank:
        raise ValueError('Required raw CLIP bank with features and class_names')
    names=[str(v) for v in bank['class_names']]
    if len(names)!=116 or len(set(names))!=116 or not all("'s " in n for n in names):
        raise ValueError('Original PP116 taxonomy must contain 116 distinct object\'s part names')
    seen=[i for i,n in enumerate(names) if n.split("'s ",1)[0] not in UNSEEN_OBJECTS]
    unseen=[i for i,n in enumerate(names) if n.split("'s ",1)[0] in UNSEEN_OBJECTS]
    if len(seen)!=74 or len(unseen)!=42:
        raise ValueError(f'Expected 74 seen, 42 unseen; got {len(seen)}/{len(unseen)}')
    if set(n.split("'s ",1)[0] for n in names if n.split("'s ",1)[0] in UNSEEN_OBJECTS)!=UNSEEN_OBJECTS:
        raise ValueError('Unseen object taxonomy mismatch')
    bank_features=torch.as_tensor(bank['features'],dtype=torch.float32)
    if bank_features.shape!=(116,512):
        raise ValueError(f'Expected [116,512] raw CLIP features, got {tuple(bank_features.shape)}')
    return names,seen,unseen,bank_features

def part_id_from_annotation(ann,names):
    """Return explicit 0-based part id when present; metadata is validated later against CLIP."""
    lookup={n.lower():i for i,n in enumerate(names)}
    for k in ('part_class_name','part_name','category_name','class_name','caption','text'):
        v=ann.get(k)
        if isinstance(v,str) and v.strip().lower() in lookup:
            return lookup[v.strip().lower()], k
    # Talk2DINO's historical PP116 features frequently store only a caption
    # such as "a photo of a cat's head." rather than a numeric class ID.
    for k in ('caption','text','name'):
        v=ann.get(k)
        if isinstance(v,str):
            matches=[i for i,n in enumerate(names) if re.search(r'(?<![a-z0-9_])'+re.escape(n.lower())+r'(?![a-z0-9_])',v.lower())]
            if len(matches)==1: return matches[0], k+'_contains_taxonomy_name'
            if len(matches)>1:
                # Ambiguous text is not safe even if a CLIP nearest neighbor exists.
                raise ValueError(f'Ambiguous annotation {k} contains multiple PP116 part names: {v!r}')
    for k in ('part_category_id','part_id','category_id'):
        v=ann.get(k)
        if v is not None:
            if isinstance(v,(tuple,list)) and len(v)==1: v=v[0]
            if hasattr(v,'numel') and v.numel()==1: v=v.item()
            if isinstance(v,(int,float)) and int(v)==v and 0<=int(v)<116:
                return int(v),k
    return None,None

def resolve_ids(anns,names,bank_features,threshold=0.96,delta=0.025):
    raw=F.normalize(bank_features,dim=1)
    output=[]
    counts=Counter()
    misfits=[]
    for start in range(0,len(anns),1024):
        chunk=anns[start:start+1024]
        feats=[]
        for i,a in enumerate(chunk,start):
            if not isinstance(a,dict) or 'ann_feats' not in a:
                raise ValueError(f'row {i}: missing ann_feats in official dataset')
            v=torch.as_tensor(a['ann_feats'],dtype=torch.float32).view(-1)
            if v.numel()!=512 or not torch.isfinite(v).all() or v.norm()<1e-12:
                raise ValueError(f'row {i}: invalid CLIP ann_feats shape/value')
            feats.append(v)
        sim=F.normalize(torch.stack(feats),dim=1) @ raw.T
        top=sim.topk(2,dim=1)
        for j,ann in enumerate(chunk):
            row=start+j
            exact,src=part_id_from_annotation(ann,names)
            guess=int(top.indices[j,0]); best=float(top.values[j,0]); margin=float(top.values[j,0]-top.values[j,1])
            if exact is not None:
                # A metadata class ID must not contradict a clearly matching prompt feature.
                if best>=threshold and margin>=delta and guess!=exact:
                    raise ValueError(f'row {row}: metadata class {exact} ({src}) contradicts CLIP matching class {guess} cosine={best:.5f}. No safe filtering.')
                pid=exact; counts['explicit_'+src]+=1
            else:
                if best<threshold or margin<delta:
                    misfits.append({'row':row,'best_cosine':round(best,5),'best_margin':round(margin,5),'first_guess':names[guess], 'annotation_keys':list(ann)})
                    continue
                pid=guess; counts['verified_clip_match']+=1
            output.append(pid)
    if misfits:
        raise ValueError(f'{len(misfits)} rows lack reliable class IDs. First examples: {json.dumps(misfits[:3],ensure_ascii=False)}; cannot safely filter seen-only.')
    if len(output)!=len(anns): raise RuntimeError('Incomplete class mapping')
    return output,dict(counts)

def prepare(source, bank_path, output, report, dry_run=False):
    bank=load(bank_path)
    names,seen,unseen,feat=get_split(bank)
    p=load(source)
    if not isinstance(p,dict) or not isinstance(p.get('images'),(list,tuple)) or not isinstance(p.get('annotations'),(list,tuple)):
        raise ValueError('Official Talk2DINO feature file must have images[] and annotations[]')
    images=list(p['images']); anns=list(p['annotations'])
    ids,source_counts=resolve_ids(anns,names,feat)
    seen_set=set(seen)
    kept=[a for a,pid in zip(anns,ids) if pid in seen_set]
    kept_ids=[pid for pid in ids if pid in seen_set]
    if not kept: raise ValueError('No seen annotation survived')
    retained_image_ids={a['image_id'] for a in kept}
    kept_images=[img for img in images if img['id'] in retained_image_ids]
    if len(kept_images)!=len(retained_image_ids):
        raise ValueError('Image IDs referenced by kept annotations are not unique / missing')
    if any(names[pid].split("'s ",1)[0] in UNSEEN_OBJECTS for pid in kept_ids):
        raise AssertionError('Unseen annotation leak')
    cnt=Counter(names[pid].split("'s ",1)[0] for pid in kept_ids)
    report_obj={
        'split':'OV-PARTS 74 seen / 42 unseen; unseen objects bird car dog motorbike sheep',
        'source':str(source),'text_bank':str(bank_path),
        'source_annotations':len(anns),'source_images':len(images),
        'retained_seen_annotations':len(kept),'excluded_unseen_annotations':len(anns)-len(kept),
        'retained_images':len(kept_images),
        'seen_class_ids':seen,'unseen_class_ids':unseen,
        'distinct_seen_classes_in_samples':len(set(kept_ids)),
        'seen_classes_without_training_pairs':[names[i] for i in seen if i not in set(kept_ids)],
        'retained_pairs_by_object':dict(sorted(cnt.items())),
        'identity_resolution_counts':source_counts,
        'feature_keys_first_image':list(images[0]) if images else [],
        'feature_keys_first_annotation':list(anns[0]) if anns else [],
        'info':'Original image-level DINO attention features are UNCHANGED; they may depict multiple objects. Only seen part-text annotation pairs are retained. No GT masks/crops are loaded.'
    }
    print(json.dumps(report_obj,ensure_ascii=False,indent=2),flush=True)
    if dry_run: return
    output.parent.mkdir(parents=True,exist_ok=True)
    report.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(f'Refusing to replace feature cache: {output}')
    payload=dict(p)
    payload['images']=kept_images
    payload['annotations']=kept
    fd,tmp=tempfile.mkstemp(dir=output.parent,prefix=output.name+'.',suffix='.tmp');os.close(fd)
    try: torch.save(payload,tmp);os.replace(tmp,output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    report.write_text(json.dumps(report_obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[SAVED]',output, flush=True)
    print('[REPORT]',report, flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='feature/talk2dino_part/pp116_train_subimagenet_officialfull.pth')
    ap.add_argument('--text_bank',default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    ap.add_argument('--output',default='feature/talk2dino_part/pp116_train_subimagenet_seen74.pth')
    ap.add_argument('--report',default='feature/talk2dino_part/pp116_train_subimagenet_seen74_report.json')
    ap.add_argument('--dry_run',action='store_true')
    a=ap.parse_args()
    for file in (a.source,a.text_bank):
        if not Path(file).is_file(): raise FileNotFoundError(file)
    prepare(Path(a.source),Path(a.text_bank),Path(a.output),Path(a.report),a.dry_run)
if __name__=='__main__':main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADE20K-Part-234: train-GT full-image DINOv2 prototype, orthogonal CE W Oracle.

This is PRIVILEGED SUPERVISION.  Never report it as weakly supervised.

Commands:
  self-test, extract, fit, bake, inspect

No web/network; never modify official repo or existing checkpoints. Output paths
refuse overwrite. A GT train mask is indexed 0..233; ignore == 65535.
Every whole training image and GT mask is independently resized to 448x448,
GT nearest to 32x32; DINOv2 reg 1024 patch features are pooled per part.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

C,D,S = 234,768,448
IGNORE=65535
MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)
BASE=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
TEXT=Path('feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt')
CFG=Path('configs/vitb_mlp_infonce.yaml')
DEFAULT_BANK=Path('feature/ade20kpart234_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool_ORACLE.pt')
DEFAULT_W=Path('output/ade234_gt_prototype_orthogonal_CE_oracle/diagnostic_GT_Orthogonal_CE_W.pt')
DEFAULT_BAKED=Path('weights/vitb_mlp_infonce_coco2014_ade234_GTProto_OrthCE_150step.pth')


def load_pt(path):
    try:return torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError:return torch.load(str(path),map_location='cpu')

def names234():
    from ade20kpart234_taxonomy import PART_CLASSES
    names=list(PART_CLASSES)
    if len(names)!=C or len(set(names))!=C:raise RuntimeError('ADE taxonomy must contain 234 distinct classes')
    return names

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def unique_save(obj,path):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Refusing overwrite: {path}')
    path.parent.mkdir(exist_ok=True,parents=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.stem+'_',suffix='.tmp',dir=str(path.parent));os.close(fd)
    try:
        torch.save(obj,tmp)
        if path.exists():raise FileExistsError(path)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.remove(tmp)

def file_pairs(images_root,masks_root):
    files=sorted(Path(masks_root).glob('*.png'))
    if not files:raise FileNotFoundError('No train GT masks in '+str(masks_root))
    pairs=[]
    for mask in files:
        matched=[Path(images_root)/(mask.stem+ext) for ext in ('.jpg','.jpeg','.png','.JPG','.JPEG') if (Path(images_root)/(mask.stem+ext)).is_file()]
        if len(matched)!=1:raise FileNotFoundError(f'Expected exactly 1 image for {mask}, got {matched}')
        pairs.append((matched[0],mask))
    return pairs

def get_frozen_dino(args,device):
    # Explicit local checkout + cached checkpoint. No network even on cache miss.
    repo=Path(args.dino_repo).expanduser().resolve()
    checkpoint=Path(args.dino_weights).expanduser().resolve()
    if not (repo/'hubconf.py').is_file():raise FileNotFoundError('Local DINOv2 hubconf.py missing: '+str(repo))
    if not checkpoint.is_file():raise FileNotFoundError('Local DINOv2 checkpoint missing: '+str(checkpoint))
    model=torch.hub.load(str(repo),'dinov2_vitb14_reg',source='local',pretrained=False)
    pack=load_pt(checkpoint)
    if isinstance(pack,dict):
        for key in ('state_dict','model','teacher'):
            if key in pack and isinstance(pack[key],dict):pack=pack[key];break
    if not isinstance(pack,dict):raise ValueError('DINOv2 checkpoint must be a state_dict')
    if all(k.startswith('module.') for k in pack):pack={k[len('module.'):]:v for k,v in pack.items()}
    model.load_state_dict(pack,strict=True)
    model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return model

def preprocess(rgb,gt):
    if rgb.size != gt.size:raise ValueError(f'RGB and GT size mismatch: {rgb.size}, {gt.size}')
    rgb=rgb.convert('RGB').resize((S,S),Image.Resampling.BICUBIC)
    arr=np.asarray(rgb,dtype=np.float32).copy()/255.0
    arr=(arr-np.asarray(MEAN,dtype=np.float32))/np.asarray(STD,dtype=np.float32)
    x=torch.from_numpy(arr).permute(2,0,1).contiguous()
    y=np.asarray(gt.resize((32,32),Image.Resampling.NEAREST),dtype=np.int64).copy().reshape(-1)
    invalid=(y!=IGNORE)&((y<0)|(y>=C))
    if invalid.any():raise ValueError('GT label outside 0..233/65535: '+str(np.unique(y[invalid])))
    return x,torch.from_numpy(y)

def extract(args):
    torch.manual_seed(123)
    pairs=file_pairs(args.images_root,args.gtpart_root)
    if args.max_images:
        pairs=pairs[:args.max_images]
        print('[SMOKE] extracting limited images; NOT a full oracle bank')
    else:
        if len(pairs)!=7347:raise RuntimeError(f'Expected complete 7347 GT train masks, found {len(pairs)}')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=get_frozen_dino(args,device)
    sums=torch.zeros(C,D,dtype=torch.float64)
    counts=torch.zeros(C,dtype=torch.int64)
    t0=time.time()
    from tqdm import tqdm
    for offset in tqdm(range(0,len(pairs),args.batch_size),desc='Extract full-image GT prototypes'):
        batch=[];lb=[]
        for img_path,mask_path in pairs[offset:offset+args.batch_size]:
            with Image.open(img_path) as im, Image.open(mask_path) as mk:
                a,b=preprocess(im,mk)
            batch.append(a);lb.append(b)
        x=torch.stack(batch,dim=0).to(device)
        with torch.inference_mode():
            d=model.forward_features(x)
            if not isinstance(d,dict) or 'x_norm_patchtokens' not in d:
                raise RuntimeError(f'Expected DINOv2 x_norm_patchtokens; got {list(d) if isinstance(d,dict) else type(d)}')
            feat=d['x_norm_patchtokens']
            if feat.shape!=(len(batch),1024,D):raise RuntimeError(f'Unexpected DINO tokens {feat.shape}')
            feat=feat.float().cpu().double().reshape(-1,D)
        labels=torch.stack(lb).reshape(-1)
        mask=labels!=IGNORE
        sums.index_add_(0,labels[mask],feat[mask])
        counts+=torch.bincount(labels[mask],minlength=C)
        if (offset//args.batch_size)%100==0:
            print(f'[EXTRACT] images={min(offset+args.batch_size,len(pairs))}/{len(pairs)} seen_parts={int((counts>0).sum())} gt_patch={int(counts.sum())}',flush=True)
    proto=F.normalize((sums/counts.clamp_min(1)[:,None]).float(),dim=-1,eps=1e-12)
    proto[counts==0]=0.0
    pack={'format':'ADE234_train_GT_fullimage_DINOv2_vitb14_reg_patch_pool_oracle_v1',
          'source':'PRIVILEGED training PART GT; NEVER weak supervision',
          'prototypes':proto,'counts':counts,'class_names':names234(),'train_images':len(pairs),
          'valid_classes':int((counts>0).sum()),'ignore':IGNORE,
          'preprocessing':'RGB bicubic 448x448 ImageNet normalization; GT nearest 32x32; L2-normalized x_norm_patchtokens averaged per GT part then L2-normalized (NO extra per-patch L2 normalization)',
          'dino_checkpoint_sha256':sha(Path(args.dino_weights).expanduser()),
          'time_seconds':time.time()-t0}
    unique_save(pack,args.out)
    print('[EXTRACT PASS] bank=',args.out,'GT valid parts=',pack['valid_classes'],'time=',round(pack['time_seconds'],1))
    if pack['valid_classes']<C:print('[WARNING] zero-training-GT classes:',[(i,names234()[i]) for i in range(C) if counts[i]==0])


def checked_state(ckpt):
    from collections.abc import Mapping
    if not isinstance(ckpt,Mapping):raise ValueError('Projector checkpoint must be a dict')
    for key in ('state_dict','model_state_dict','projector_state_dict'):
        if key in ckpt and isinstance(ckpt[key],Mapping):ckpt=ckpt[key];break
    expected={'linear_layer.weight':(768,512),'linear_layer.bias':(768,),
              'hidden_layers.0.weight':(768,768),'hidden_layers.0.bias':(768,)}
    if set(ckpt)!=set(expected):raise RuntimeError(f'Unexpected projector keys: {list(ckpt)}')
    for k,shape in expected.items():
        if not torch.is_tensor(ckpt[k]) or tuple(ckpt[k].shape)!=shape:
            raise RuntimeError(f'Wrong shape of {k}')
    return {k:v for k,v in ckpt.items()}


def build_model(config,state):
    import yaml
    from src.model import ProjectionLayer
    cfg=yaml.safe_load(Path(config).read_text())['model']
    mod=ProjectionLayer.from_config(cfg)
    mod.load_state_dict(state,strict=True)
    mod.eval()
    for v in mod.parameters():v.requires_grad_(False)
    return mod


def load_ade234_raw_text(path):
    """Read ADE234's 234x512 *raw* CLIP bank (not PP116's fixed-116 loader)."""
    from collections.abc import Mapping
    payload=load_pt(path)
    if not isinstance(payload,Mapping):
        raise TypeError(f'Raw ADE text bank must be a mapping: {path}')
    if 'features' not in payload:
        raise KeyError(f'Raw ADE text bank lacks features: {path}')
    raw=torch.as_tensor(payload['features'],dtype=torch.float32).cpu()
    if tuple(raw.shape)!=(C,512):
        raise ValueError(f'Expected ADE234 raw CLIP features [234,512], got {tuple(raw.shape)}')
    if not torch.isfinite(raw).all() or not bool((raw.norm(dim=-1)>1e-12).all()):
        raise ValueError('ADE234 text bank contains non-finite or zero features')
    actual=payload.get('class_names',payload.get('classnames',payload.get('names')))
    if actual is None:
        raise KeyError('ADE234 text bank lacks class_names/classnames/names')
    actual=[str(x) for x in actual]
    if actual!=names234():
        raise ValueError('ADE234 text bank class ordering differs from taxonomy')
    if payload.get('clip_model') not in (None,'','ViT-B/16'):
        raise ValueError(f'Unexpected CLIP model: {payload.get("clip_model")}')
    if payload.get('template_set') not in (None,'','sub_imagenet_template'):
        raise ValueError(f'Unexpected CLIP template: {payload.get("template_set")}')
    if payload.get('normalized',False) is not False:
        raise ValueError('Expected raw, pre-projector, non-normalized CLIP bank')
    if payload.get('stage') not in (None,'','pre_projector_prompt_mean'):
        raise ValueError(f'Unexpected text bank stage: {payload.get("stage")}')
    return raw.contiguous(),actual


def load_projected_text(args):
    names=names234()
    raw,actual=load_ade234_raw_text(Path(args.text_bank))
    if tuple(raw.shape)!=(C,512) or list(actual)!=names:raise ValueError('234 CLIP text feature ordering does not match ADE taxonomy')
    base=checked_state(load_pt(args.base_projector))
    mod=build_model(args.model_config,base)
    with torch.no_grad():t0=F.normalize(mod.project_clip_txt(raw.float()),dim=-1)
    if t0.shape!=(C,D):raise RuntimeError('Projector does not produce 234x768')
    return t0,base


def check_bank(args):
    bank=load_pt(args.bank)
    p=bank.get('prototypes') if isinstance(bank,dict) else None
    if not torch.is_tensor(p) or p.shape!=(C,D):raise ValueError('Expected 234x768 GT prototypes tensor')
    if bank.get('class_names')!=names234():raise RuntimeError('GT prototype class order mismatch')
    counts=bank.get('counts')
    if not torch.is_tensor(counts) or counts.shape!=(C,):raise ValueError('Expected 234 counts')
    if (p[counts>0].norm(dim=-1)<0.999).any():raise RuntimeError('GT prototypes not L2 normalized')
    return p.float(),counts

def fit(args):
    torch.manual_seed(args.seed)
    p,counts=check_bank(args)
    t0,_=load_projected_text(args)
    device=torch.device(args.device)
    proto=p.to(device);t0=t0.to(device)
    valid=(counts>0).nonzero(as_tuple=True)[0].to(device)
    if len(valid)<10:raise ValueError('Oracle bank contains too few classes; do not fit a smoke bank')
    W=torch.nn.Parameter(torch.eye(D,device=device))
    opt=torch.optim.Adam([W],lr=args.lr,weight_decay=0.0)
    history=[]
    for step in range(1,args.steps+1):
        opt.zero_grad(set_to_none=True)
        t=F.normalize(t0@W,dim=-1)
        logits=proto[valid].detach()@t.T/args.temperature
        loss=F.cross_entropy(logits,valid)
        if not torch.isfinite(loss):raise RuntimeError('Oracle CE nonfinite')
        loss.backward();opt.step()
        with torch.no_grad():
            # Orthogonal retraction on EVERY step, same constraint as PP116 CE Oracle.
            U,_,Vh=torch.linalg.svd(W,full_matrices=False)
            W.copy_(U@Vh)
            if step==1 or step%10==0 or step==args.steps:
                pred=(proto[valid]@F.normalize(t0@W,dim=-1).T).argmax(dim=-1)
                right=int((pred==valid).sum())
                print(f'[CE] step={step:03d} loss={loss.item():.6f} train_GT_top1={right}/{len(valid)}',flush=True)
                history.append({'step':step,'loss':float(loss.item()),'top1':right})
    orth_err=float((W.T@W-torch.eye(D,device=device)).abs().max())
    if orth_err>1e-3:raise RuntimeError(f'Orthogonality failed: {orth_err}')
    pack={'W':W.detach().cpu(),'source':'TRAIN ADE234 GT ORACLE Orthogonal CE (privileged)',
          'format':'ADE234_GT_ORTHOGONAL_CE_ORACLE_v1','valid_train_parts':len(valid),
          'bank':str(args.bank),'bank_sha256':sha(args.bank),
          'text_bank':str(args.text_bank),'base_projector':str(args.base_projector),
          'steps':args.steps,'lr':args.lr,'temperature':args.temperature,'seed':args.seed,
          'orthogonality_max_error':orth_err,'history':history}
    unique_save(pack,args.out)
    print('[FIT PASS]',args.out,'orth_err=',orth_err)


def bake(args):
    bankW=load_pt(args.w)
    if not isinstance(bankW,dict) or 'TRAIN ADE234 GT ORACLE' not in str(bankW.get('source','')):
        raise RuntimeError('Expected privileged training-GT Oracle W')
    W=torch.as_tensor(bankW['W']).float()
    if W.shape!=(D,D) or (W.T@W-torch.eye(D)).abs().max()>1e-3:
        raise RuntimeError('Bad/nonorthogonal W')
    t0,src=load_projected_text(args)
    with torch.no_grad():expected=F.normalize(t0@W,dim=-1)
    new={k:v.detach().clone() for k,v in src.items()}
    key='hidden_layers.0.weight'; bias='hidden_layers.0.bias'
    new[key]=(W.T.double()@src[key].double()).to(src[key].dtype)
    new[bias]=(W.T.double()@src[bias].double()).to(src[bias].dtype)
    mod=build_model(args.model_config,new)
    raw,_=load_ade234_raw_text(Path(args.text_bank))
    with torch.no_grad():got=F.normalize(mod.project_clip_txt(raw.float()),dim=-1)
    err=float((got-expected).abs().max())
    if not torch.allclose(got,expected,rtol=2e-5,atol=2e-5):raise RuntimeError(f'Bake mismatch {err}')
    unique_save(new,args.out)
    note={'source':'PRIVILEGED ADE234 TRAIN GT PROTOTYPE ORACLE; NOT WEAKLY SUPERVISED',
          'W':str(args.w),'base':str(args.base_projector),'base_sha256':sha(args.base_projector),
          'text_sha256':sha(args.text_bank),'bake_error_max':err,'checkpoint':str(args.out)}
    path=Path(str(args.out)+'.oracle.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[BAKE PASS]',args.out,'max_abs_diff',err)


def self_test(_):
    # Test matrix geometry, correct label 0, ignore 65535, and source CE gradient.
    x=np.zeros((448,448,3),dtype=np.uint8)
    y=np.full((448,448),IGNORE,dtype=np.uint16);y[:224,:224]=0;y[224:,224:]=233
    X,Y=preprocess(Image.fromarray(x),Image.fromarray(y))
    assert X.shape==(3,448,448) and Y.shape==(1024,)
    assert int((Y==0).sum())==256 and int((Y==233).sum())==256 and int((Y==IGNORE).sum())==512
    t=F.normalize(torch.randn(10,16),dim=-1);g=F.normalize(torch.randn(10,16),dim=-1)
    W=torch.nn.Parameter(torch.eye(16));logits=g@F.normalize(t@W,dim=-1).T/0.07
    F.cross_entropy(logits,torch.arange(10)).backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    with torch.no_grad():u,_,v=torch.linalg.svd(W);W.copy_(u@v)
    assert float((W.detach().T@W.detach()-torch.eye(16)).abs().max())<1e-4
    print('SELF_TEST_PASS: 0 valid; 65535 ignore; CE gradients; orthogonal SVD')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='command',required=True)
    p=sp.add_parser('extract');p.add_argument('--images-root',default='../../../dataset/lyx/ADE20KPart234/images/train');p.add_argument('--gtpart-root',default='../../../dataset/lyx/ADE20KPart234/annotations_detectron2_part/train');p.add_argument('--dino-repo',default=str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'));p.add_argument('--dino-weights',default=str(Path.home()/'.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth'));p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-images',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,default=DEFAULT_BANK)
    for cmd in ('fit','bake'):
        p=sp.add_parser(cmd);p.add_argument('--text-bank',type=Path,default=TEXT);p.add_argument('--base-projector',type=Path,default=BASE);p.add_argument('--model-config',type=Path,default=CFG);p.add_argument('--device',default='cuda');p.add_argument('--bank',type=Path,default=DEFAULT_BANK);p.add_argument('--out',type=Path,default=DEFAULT_W if cmd=='fit' else DEFAULT_BAKED)
        if cmd=='fit':p.add_argument('--seed',type=int,default=123);p.add_argument('--steps',type=int,default=150);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--temperature',type=float,default=0.07)
        else:p.add_argument('--w',type=Path,default=DEFAULT_W)
    sp.add_parser('self-test')
    args=ap.parse_args()
    if args.command=='self-test':self_test(args)
    elif args.command=='extract':extract(args)
    elif args.command=='fit':fit(args)
    else:bake(args)
if __name__=='__main__':main()

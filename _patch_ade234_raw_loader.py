from pathlib import Path
import py_compile
import shutil
import sys

p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('ade234_gtprototype_orthoracle.py')
s=p.read_text(encoding='utf-8')
helper='''def load_ade234_raw_text(path):
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


'''
marker='def load_projected_text(args):\n'
old1="raw,actual,_=tr.load_raw_clip_bank(Path(args.text_bank),expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')"
new1='raw,actual=load_ade234_raw_text(Path(args.text_bank))'
old2="from train_relproto_alignemt import load_raw_clip_bank\n    raw,_,_=load_raw_clip_bank(Path(args.text_bank),expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')"
new2='raw,_=load_ade234_raw_text(Path(args.text_bank))'
if 'def load_ade234_raw_text(path):' not in s:
    if s.count(marker)!=1:raise RuntimeError('Unexpected script layout: projected text function')
    s=s.replace(marker,helper+marker,1)
else:
    print('[SKIP] ADE234 raw loader already exists')
if old1 in s:
    if s.count(old1)!=1:raise RuntimeError('Duplicated fit old loader')
    s=s.replace(old1,new1,1)
elif new1 not in s:raise RuntimeError('Fit raw loader location not found')
if old2 in s:
    if s.count(old2)!=1:raise RuntimeError('Duplicated bake old loader')
    s=s.replace(old2,new2,1)
elif new2 not in s:raise RuntimeError('Bake raw loader location not found')
if 'def load_projected_text(args):\n    import train_relproto_alignemt as tr\n' in s:
    s=s.replace('def load_projected_text(args):\n    import train_relproto_alignemt as tr\n','def load_projected_text(args):\n',1)
assert new1 in s and new2 in s and old1 not in s and old2 not in s
if s!=p.read_text(encoding='utf-8'):
    p.write_text(s,encoding='utf-8')
    print('[PATCH] ADE234 fit + bake now use 234-class raw CLIP loader')
else:
    print('[SKIP] Already patched')
py_compile.compile(str(p),doraise=True)
print('[PASS] Syntax check')

"""Read-only real-bank audit: W text path, loss value and gradients."""
from pathlib import Path
import argparse
import torch
from torch.nn import functional as F
import train_relproto_alignemt as wtrainer
from src.part_structure_rawpath_v1 import PartStructureRankLoss


def reference_loss(model, raw, groups, temperature=0.05):
    # Old soft-rank / correlation definition, with corrected raw projector input.
    losses = []
    def rank(v):
        return 1 + torch.sigmoid((v[:, None]-v[None, :])/temperature).sum(1)-0.5
    for ids in groups.values():
        ids = torch.as_tensor(ids, device=raw.device, dtype=torch.long)
        if ids.numel() < 3:
            continue
        x = raw[ids]
        before = F.normalize(x, dim=-1, eps=1e-6)
        after = F.normalize(model.project_clip_txt(x).float(), dim=-1, eps=1e-12)
        r,c = torch.triu_indices(len(ids),len(ids),1,device=raw.device)
        a = rank((before @ before.T)[r,c]).detach()
        b = rank((after @ after.T)[r,c])
        a, b = a-a.mean(), b-b.mean()
        a = a/a.norm().clamp_min(1e-6)
        b = b/b.norm().clamp_min(1e-6)
        losses.append(1-(a*b).sum())
    return torch.stack(losses).mean()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    p.add_argument('--text_bank',default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt')
    p.add_argument('--model_config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    root=Path('.').resolve(); device=torch.device(args.device)
    torch.set_num_threads(2)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
    payload=torch.load(root/args.text_bank,map_location='cpu')
    assert payload.get('normalized',False) is False
    raw=torch.as_tensor(payload.get('features',payload.get('raw_features')),dtype=torch.float32)
    assert raw.shape==(116,512)
    groups=payload['object_groups']
    assert isinstance(groups,dict)
    flat=[int(i) for ids in groups.values() for i in ids]
    assert sorted(flat)==list(range(116)), 'Groups must partition all 116 rows'
    model,_=wtrainer.load_frozen_projector(project_root=root,config_path=root/args.model_config,weight_path=root/args.weights,device=device)
    model.eval()
    criterion=PartStructureRankLoss.from_file(root/args.text_bank).to(device)
    assert torch.equal(criterion.raw_part_features.cpu(),raw), 'Raw input altered'
    expected=wtrainer.project_raw_clip_bank(raw,model,device=device,batch_size=128)
    received=[]
    original=model.project_clip_txt
    def capture(x):
        y=original(x)
        received.append((x.detach().clone(),y.detach().clone()))
        return y
    model.project_clip_txt=capture
    for parameter in model.parameters(): parameter.requires_grad_(True)
    loss=criterion(model)
    model.project_clip_txt=original
    assert len(received)==1
    x,y=received[0]
    assert torch.equal(x.cpu(),raw)
    actual=F.normalize(y.float(),dim=-1,eps=1e-12).cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    print('PASS raw projector input and T0 match actual W trainer')
    parameters=tuple(model.parameters())
    grads=torch.autograd.grad(loss,parameters,allow_unused=True)
    reference=reference_loss(model,raw.to(device),groups)
    ref_grads=torch.autograd.grad(reference,parameters,allow_unused=True)
    torch.testing.assert_close(loss,reference,rtol=1e-4,atol=2e-6)
    peak=0.0
    for a,b in zip(grads,ref_grads):
        assert (a is None)==(b is None)
        if a is not None:
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-5)
            peak=max(peak,(a-b).abs().max().item())
    print('PASS loss and gradients match object-wise reference')
    print('loss:',loss.item(),'reference:',reference.item(),'max gradient difference:',peak)
    print('Exact Spearman on raw -> projector:')
    for name,value in criterion.exact_audit(model).items():print(f'{name:16s} {value:.8f}')
    print('RAWPATH_AUDIT_PASS')

if __name__=='__main__':main()

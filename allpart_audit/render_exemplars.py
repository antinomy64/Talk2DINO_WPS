#!/usr/bin/env python3
"""Replay selected oracle-audit crops; save dense 32x32 score/weight/GT maps.

Selection source = manifest from independent all-crop audit (not model training).
GT NEVER used to compute model scores, weights or prototype assignments.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from audit_all_116 import load_bank,load_ws,load_names,score_all_epochs

def parse():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,default=Path('output/pp116_allpart_audit_ours_v1'))
 p.add_argument('--run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
 p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
 p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
 p.add_argument('--epochs',default='0,3,10,30,100')
 p.add_argument('--max-parts-per-group',type=int,default=4)
 p.add_argument('--device',default='cuda')
 p.add_argument('--flip-mode',default='auto')
 return p.parse_args()

def main():
 a=parse()
 import train_relproto_alignemt as src
 import audit_pp116_gt_weighted_purity as gt_audit
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 device=torch.device(a.device);torch.backends.cuda.matmul.allow_tf32=False
 names,_=load_names(Path(__file__).resolve().parent/'pp116_four_groups_per_part.csv',a.run/'projected_part_bank.pt')
 base,bank_path=load_bank(a.run,None,device,names)
 epochs=[int(e) for e in a.epochs.split(',')]
 with torch.no_grad():tb=load_ws(a.run,epochs,base,device)
 frozen=(base@base.T).detach()
 anns=gt_audit.annotations(src.torch_load(a.cache));lookup=gt_audit.GTLookup(a.gt_root)
 with (a.out/'summary'/'exemplar_crop_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 picked=[]
 for g in 'ABCD':
  ps=[]
  for r in rows:
   if r['group']!=g:continue
   if r['pid'] not in ps:
    if len(ps)>=a.max_parts_per_group:continue
    ps.append(r['pid'])
   picked.append(r)
 viz=a.out/'visuals';viz.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for i,r in enumerate(picked):
  idx=int(r['crop_index']);pid=int(r['pid']);ann=anns[idx]
  ids=[int(v) for v in gt_audit.part_ids(ann)];pos=ids.index(pid)
  tok=F.normalize(torch.as_tensor(ann['cropaug_patch_tokens'],dtype=torch.float32,device=device),dim=-1)
  fg=torch.as_tensor(ann['pred_obj_mask_patch'],dtype=torch.bool,device=device).reshape(-1)
  with torch.no_grad():
   s=score_all_epochs(tb,ids,tok,fg,.05,.07,frozen,.05,.05,check_source=(i==0))
  gt,_,_,_=lookup.crop32(ann,a.flip_mode)
  gt=np.asarray(gt).reshape(-1);f=fg.cpu().numpy().astype(bool)
  truth=(gt==pid)&f;other=(gt!=255)&(gt!=pid)&f
  N=len(epochs);fig,axs=plt.subplots(4,N+1,figsize=(2.5*(N+1),9),dpi=120)
  for ax in axs.ravel():ax.set_xticks([]);ax.set_yticks([])
  label=np.zeros(1024,dtype=np.uint8);label[(gt==255)&f]=1;label[other]=2;label[truth]=3
  axs[0,0].imshow(label.reshape(32,32),cmap=ListedColormap(['#242933','#777777','#f39c42','#24ca77']),vmin=0,vmax=3)
  axs[0,0].set_title('GT: green=part')
  axs[1,0].imshow(f.reshape(32,32),cmap='gray',vmin=0,vmax=1);axs[1,0].set_title('Predicted FG')
  axs[2,0].text(0,.95,'\n'.join(names[z] for z in ids),fontsize=6,va='top');axs[2,0].set_title('Image-level candidates')
  axs[3,0].text(0,.95,f"part={names[pid]}\nABCD={r['group']}\ncrop={idx}\n{r['role']}\nGT-FG patches={truth.sum()}",va='top',fontsize=8)
  for t,ep in enumerate(epochs):
    m=s['relative'][t,pos].copy();m[~f]=np.nan
    axs[0,t+1].imshow(m.reshape(32,32),cmap='RdBu_r',vmin=-.2,vmax=.2,interpolation='nearest');axs[0,t+1].set_title(f'E{ep} margin' if len(ids)>1 else f'E{ep} absolute')
    axs[1,t+1].imshow(s['weight'][t,pos].reshape(32,32),cmap='magma',vmin=0,interpolation='nearest')
    sel=s['selector'][t,pos];cat=np.zeros(1024,dtype=np.uint8)
    cat[(gt==pid)&f]=1;cat[sel&(gt==pid)]=2;cat[sel&(gt==255)]=3;cat[sel&other]=4
    axs[2,t+1].imshow(cat.reshape(32,32),cmap=ListedColormap(['#20242b','#456cad','#28d785','#a88acc','#f9ad50']),vmin=0,vmax=4,interpolation='nearest')
    riv=s['rival'][t,pos].astype(float).copy();riv[~f]=np.nan
    axs[3,t+1].imshow(riv.reshape(32,32),cmap='tab20',vmin=-.5,vmax=max(1,len(ids))-.5,interpolation='nearest')
  axs[1,1].set_ylabel('Pseudo weight',fontsize=8)
  axs[2,1].set_ylabel('Correct/ignore/wrong',fontsize=8)
  axs[3,1].set_ylabel('Rival local ID',fontsize=8)
  fig.suptitle(f'{r["group"]}: {names[pid]} | {r["role"]} | training-crop GT audit, NOT validation',fontsize=12)
  fig.tight_layout(rect=(0,0,1,.95))
  path=viz/f'{r["group"]}_{pid:03d}_{r["role"]}_crop{idx:05d}.png';fig.savefig(path);plt.close(fig)
  np.savez_compressed(viz/f'{r["group"]}_{pid:03d}_{r["role"]}_crop{idx:05d}.npz',
           gt=gt,fg=f,epochs=np.array(epochs),pids=np.array(ids),target_pid=pid,
           weights=s['weight'][:,pos],margin=s['relative'][:,pos],scores=s['scores'][:,pos],
           rival_local=s['rival'][:,pos],selected=s['selector'][:,pos])
  manifest.append(dict(path=str(path),target=names[pid],group=r['group'],role=r['role'],crop_index=idx))
  print('[visual]',i+1,'/',len(picked),names[pid],r['role'],flush=True)
 (viz/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('[DONE] visuals',viz.resolve(),flush=True)
if __name__=='__main__':main()

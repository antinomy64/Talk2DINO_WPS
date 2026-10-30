#!/usr/bin/env python3
"""Optional all-crop GT Ear margin audit of selected *completed* models.

Uses user's existing audit_pp116_ear_patch_e0_e10.py, never writes into training.
Warning: NPZ storage and GPU time scale linearly with variants.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=Path('output/pp116_ear_transfer_20261022_v1'))
    p.add_argument('--variants',default='full_l4,loo_cow_l4,target_nonear_cow_l4,other5_cow_l4,only_cow_l4',
        help='comma-delimited known variants from registry.json')
    p.add_argument('--max-ear-crops',type=int,default=0,help='0 = full dataset; >0 per-ear cap')
    p.add_argument('--gpu',type=str,default='0')
    p.add_argument('--device',type=str,default='cuda')
    a=p.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    script=Path('audit_pp116_ear_patch_e0_e10.py')
    if not script.is_file():raise FileNotFoundError('Previous fullcrop audit script missing: '+str(script))
    reg={v['variant']:v for v in json.loads((a.report/'registry.json').read_text())}
    for key in [t.strip() for t in a.variants.split(',') if t.strip()]:
        if key not in reg:raise ValueError('Unknown variant '+key)
        item=reg[key]
        if not (Path(item['run'])/'W_epoch_010.pt').exists():raise FileNotFoundError(item['run'])
        out=a.report/'allcrop'/key
        if (out/'ear_epoch_summary.csv').exists():
            print('[SKIP already audited]',key,flush=True);continue
        if out.exists():raise RuntimeError('INCOMPLETE output exists; refusing to overwrite: '+str(out))
        cmd=[sys.executable,'-u',str(script),'--run',item['run'],'--lambda-struct',str(item['lambda_struct']),
            '--start-epoch','0','--end-epoch','10','--soft-temperature','0.05','--plots-per-ear','0',
            '--max-ear-crops',str(a.max_ear_crops),'--out-dir',str(out),'--device',a.device]
        log=a.report/f'allcrop_{key}.log'
        print('[RUN ALL-CROP AUDIT]',key,'log=',log,flush=True)
        with log.open('w') as f:
            ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if ret.returncode:raise RuntimeError(f'allcrop failed {key} see {log}')
        if not (out/'ear_epoch_summary.csv').exists():raise RuntimeError(f'No summary created by {key}')
    print('[DONE] selected allcrop audits finished',flush=True)

if __name__=='__main__':main()

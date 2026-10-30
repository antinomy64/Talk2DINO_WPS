#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No .sh: sequential unattended audit orchestration with stage logs and fail-fast checks."""
import argparse,os,subprocess,sys,time
from pathlib import Path

def args_parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',default='output/pp116_allpart_audit_ours_v1')
 p.add_argument('--run',default='final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep')
 p.add_argument('--epochs',default='0,1,2,3,5,10,20,30,40,50,60,70,80,90,100')
 p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
 p.add_argument('--smoke',action='store_true',help='Only first 24 crops, 3 epochs, no gradient or figures, separate output directory')
 p.add_argument('--no-gradient',action='store_true')
 p.add_argument('--no-visuals',action='store_true')
 p.add_argument('--no-plots',action='store_true')
 p.add_argument('--full-gradient-targets',action='store_true',help='Use up to 116 target classes (expensive); source batch still sampled')
 return p.parse_args()

def main():
 a=args_parser();package=Path(__file__).resolve().parent
 if not Path('train_pp116_softsingle_dual.py').is_file():
    raise FileNotFoundError('Run from Talk2DINO_official_bg ROOT with ORIGINAL Ours trainer')
 if not Path('audit_pp116_gt_weighted_purity.py').is_file():
    raise FileNotFoundError('Prior GT audit helper missing: audit_pp116_gt_weighted_purity.py (read-only use)')
 out=Path(a.out if not a.smoke else a.out+'_smoke');out.mkdir(parents=True,exist_ok=True)
 logs=out/'logs';logs.mkdir(exist_ok=True)
 epochs=('0,1,2' if a.smoke else a.epochs)
 def run(label,script,rest):
    command=[sys.executable,'-u',str(package/script)]+list(map(str,rest))
    print(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] START {label}:', ' '.join(command),flush=True)
    begin=time.time()
    with (logs/f'{label}.log').open('w',encoding='utf-8') as f:
       p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',bufsize=1,env={**os.environ,'PYTHONUNBUFFERED':'1'})
       for line in p.stdout:
           print(line,end='',flush=True);f.write(line);f.flush()
       ret=p.wait()
    if ret:raise RuntimeError(f'{label} FAILED rc={ret}; see {logs/ (label+".log")}')
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] DONE {label} in {time.time()-begin:.1f}s',flush=True)
 run('01_audit','audit_all_116.py',['--out',out,'--run',a.run,'--epochs',epochs,'--device',a.device,
                                     '--resume','--verify-source']+(['--max-crops','24'] if a.smoke else []))
 run('02_summary','summarize_all_116.py',['--out',out]+([] if a.no_plots or a.smoke else ['--plot']))
 if not a.smoke and not a.no_gradient:
    run('03_gradients','gradient_source_probe.py',['--out',out,'--run',a.run,'--device',a.device,
       '--epochs','0,1,3,10,30,100','--targets-per-group','40' if a.full_gradient_targets else '4',
       '--num-source-crops','96' if a.full_gradient_targets else '64'])
 if not a.smoke and not a.no_visuals:
    run('04_visuals','render_exemplars.py',['--out',out,'--run',a.run,'--device',a.device,
                                           '--epochs','0,3,10,30,100'])
 run('05_pack','pack_results.py',['--out',out,'--zip',out.parent/f'{out.name}_review_upload.zip'])
 print('\nALL DONE:',out.resolve(),flush=True)
if __name__=='__main__':main()

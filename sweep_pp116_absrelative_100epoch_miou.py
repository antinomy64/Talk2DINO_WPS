#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full PP-116 val mIoU sweep across saved epochs for lambda_abs=0.5,1,8.

For each W_epoch_NNN.pt: bake a *temporary, uniquely named* projector, then
call the exact existing VOC116 official eval (850 images, pred-all, PAMR=false).
Results are checkpoint-specific; no training. Resume safe; completed eval JSON
or per-arm CSV are reused. Keeps the original/official repo files untouched.

Run from Talk2DINO_official_bg root, with its activated conda environment.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_EXPS='final_exp/relproto_alignment'
ARMS={
    'l0p5': ('pp116_absrelative_l0p5_sg_seed123_100ep',0.5),
    'l1': ('pp116_absrelative_l1_sg_seed123_100ep',1.0),
    'l2': ('pp116_absrelative_l2_sg_seed123_100ep',2.0),
    'l3': ('pp116_absrelative_l3_sg_seed123_100ep',3.0),
    'l4': ('pp116_absrelative_l4_sg_seed123_100ep',4.0),
    'l8': ('pp116_absrelative_l8_sg_seed123_100ep',8.0),
}
FMT='vitb_mlp_infonce_coco2014_reproduce_clean'
lock=threading.Lock()


def log(msg):
    with lock: print(msg,flush=True)


def parse_epochs(arg):
    result=[]
    for item in arg.split(','):
        item=item.strip()
        if not item: continue
        if ':' in item:
            tok=list(map(int,item.split(':')))
            if len(tok) not in (2,3): raise ValueError(f'Bad epoch interval: {item}')
            a,b=tok[:2];s=tok[2] if len(tok)==3 else 1
            if s<=0 or a<1 or a>b: raise ValueError(f'Bad epoch interval: {item}')
            result.extend(range(a,b+1,s))
        else:
            result.append(int(item))
    result=sorted(set(result))
    if not result or min(result)<1 or max(result)>100:
        raise ValueError(f'Epochs must be 1..100: {result}')
    return result


def result_metric(path):
    if not path.is_file(): return None
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        number=float(data['voc116_part'])
        if not 0 <= number <= 100: raise ValueError(number)
        return number
    except (OSError,ValueError,KeyError,TypeError):
        return None


def previous_results(root,arm,epoch):
    # Official existing evaluations at epochs 30,100, if still on the server.
    if epoch == 30:
        name=f'{FMT}_absrelative_{arm}_e030'
    elif epoch == 100:
        name=f'{FMT}_pp116_absrelative_{arm}_sg_e100'
    else:
        return None
    path=root/'segmentation_results'/f'{name}.json'
    val=result_metric(path)
    return (val,str(path)) if val is not None else None


def write_arm_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','lambda_abs','epoch','miou','source'])
        w.writeheader()
        for epoch in sorted(rows): w.writerow(rows[epoch])


def load_arm_csv(path,arm,lambda_abs):
    rows={}
    if not path.is_file(): return rows
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('arm')!=arm: continue
            epoch=int(row['epoch']);v=float(row['miou'])
            if 1<=epoch<=100 and 0<=v<=100:
                rows[epoch]=dict(arm=arm,lambda_abs=lambda_abs,epoch=epoch,miou=v,source=row.get('source','resumed CSV'))
    return rows


def run_logged(cmd,env,logfile,append=False):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a' if append else 'w',encoding='utf-8') as f:
        if append: f.write('\n\n')
        f.write('COMMAND: '+' '.join(map(str,cmd))+'\n')
        f.flush()
        proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=False)
    if proc.returncode:
        raise RuntimeError(f'Exit code={proc.returncode} from {cmd[1:4]} (log={logfile})')


def evaluate_arm(root,output,arm,epochs,gpu,index,keep_baked):
    exp,weight=ARMS[arm]
    path=output/f'{arm}.csv'
    rows=load_arm_csv(path,arm,weight)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    # Prevent single GPU multiprocessing CPU oversubscription with 3 workers.
    env.setdefault('OMP_NUM_THREADS','4')
    for epoch in epochs:
        if epoch in rows:
            log(f'[{arm} E{epoch:03d}] CACHE: mIoU={float(rows[epoch]["miou"]):.2f}')
            continue
        known=previous_results(root,arm,epoch)
        if known:
            metric,origin=known
            rows[epoch]=dict(arm=arm,lambda_abs=weight,epoch=epoch,miou=metric,source=origin)
            write_arm_csv(path,rows)
            log(f'[{arm} E{epoch:03d}] REUSE: mIoU={metric:.2f}')
            continue

        w_path=root/ROOT_EXPS/exp/f'W_epoch_{epoch:03d}.pt'
        if not w_path.is_file(): raise FileNotFoundError(w_path)
        pname=f'__tmp_sweep_absrelative_{arm}_e{epoch:03d}'
        baked=root/'weights'/f'{pname}.pth'
        json_result=root/'segmentation_results'/f'{pname}.json'
        val=result_metric(json_result)
        if val is None:
            logfile=output/'logs'/arm/f'epoch_{epoch:03d}.log'
            bake=[sys.executable,'bake_predobj_relproto_w_into_projector_final.py',
                  '--project_root','.', '--projector',f'weights/{FMT}.pth',
                  '--w_checkpoint',str(w_path.relative_to(root)),
                  '--text_bank','feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt',
                  '--model_config','configs/vitb_mlp_infonce.yaml',
                  '--output',str(baked.relative_to(root)), '--device','cuda']
            eval_cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1',
                      f'--master_port={33420+index}',
                      'src/open_vocabulary_segmentation/main.py','--eval',
                      '--output',str((output/'eval_outputs'/arm/f'e{epoch:03d}').relative_to(root)),
                      '--eval_cfg','src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
                      '--eval_base_cfg','src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
                      '--opts',f'model.proj_name={pname}','evaluate.pamr=false']
            log(f'[{arm} E{epoch:03d}] BAKE+EVAL gpu={gpu}')
            try:
                # Explicit workdir is repo root; avoid depending on current shell.
                run_logged_in_root(bake,env,logfile,root,append=False)
                run_logged_in_root(eval_cmd,env,logfile,root,append=True)
                val=result_metric(json_result)
                if val is None:
                    raise RuntimeError(f'No valid mIoU at {json_result}; check {logfile}')
            finally:
                if not keep_baked:
                    baked.unlink(missing_ok=True)  # Only our uniquely named baked file.
        if val is None:
            raise RuntimeError(f'No result for {arm}/{epoch}')
        rows[epoch]=dict(arm=arm,lambda_abs=weight,epoch=epoch,miou=val,source=str(json_result))
        write_arm_csv(path,rows)  # persist after *each* eval: resume safe.
        log(f'[{arm} E{epoch:03d}] mIoU={val:.2f}%   ({len(rows)} checkpoints recorded)')
    return rows


def run_logged_in_root(cmd,env,logfile,root,append):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a' if append else 'w',encoding='utf-8') as f:
        f.write('COMMAND: '+' '.join(map(str,cmd))+'\n')
        f.flush()
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=root,check=False)
    if result.returncode:
        raise RuntimeError(f'Code={result.returncode}; see {logfile}')


def summarize(output, arms):
    records=[]
    for arm in arms:
        exp,weight=ARMS[arm]
        records.extend(load_arm_csv(output/f'{arm}.csv',arm,weight).values())
    records.sort(key=lambda x:(x['epoch'],x['lambda_abs']))
    if not records:
        raise RuntimeError('No completed evaluations found')
    with (output/'all_miou.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','lambda_abs','epoch','miou','source'])
        writer.writeheader();writer.writerows(records)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5.1))
        for arm in arms:
            rs=[r for r in records if r['arm']==arm]
            if not rs: continue
            rs.sort(key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in rs],[r['miou'] for r in rs],
                    label=f'lambda_abs={ARMS[arm][1]:g}',linewidth=1.8)
            best=max(rs,key=lambda r:r['miou'])
            log(f'[{arm}] Best observed E{best["epoch"]:03d} mIoU={best["miou"]:.2f}%')
        ax.axhline(36.75,linestyle='--',linewidth=1.1,alpha=.65,label='No-Abs baseline E100: 36.75%')
        ax.set_xlabel('Epoch');ax.set_ylabel('PP-116 mIoU (%)')
        ax.set_xlim(1,100);ax.grid(alpha=.2);ax.legend()
        fig.tight_layout()
        fig.savefig(output/'miou_vs_epoch.png',dpi=200)
        fig.savefig(output/'miou_vs_epoch.pdf')
        plt.close(fig)
        log(f'[PLOT] {output/"miou_vs_epoch.png"}')
    except ImportError:
        log('[WARN] matplotlib not installed; CSV generated, plot skipped')
    log(f'[RESULT] {output/"all_miou.csv"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',default='output/pp116_absrelative_miou_sweep')
    parser.add_argument('--arms',default='l0p5,l1,l8')
    parser.add_argument('--gpus',default='0',help='One GPU (serial) or one GPU per arm, e.g. 0,1,2')
    parser.add_argument('--epochs',default='1:100',help='1:100 for all; 1:100:5 for coarse')
    parser.add_argument('--keep-baked',action='store_true')
    parser.add_argument('--plot-only',action='store_true',help='Make combined CSV/plot from finished results')
    args=parser.parse_args()
    root=Path(args.root).resolve()
    output=root/args.output
    output.mkdir(parents=True,exist_ok=True)
    arms=[a.strip() for a in args.arms.split(',') if a.strip()]
    if not arms or any(a not in ARMS for a in arms) or len(set(arms))!=len(arms):
        raise ValueError(f'Bad arms: {arms}')
    if args.plot_only:
        summarize(output,arms)
        return
    epochs=parse_epochs(args.epochs)
    gpus=[g.strip() for g in args.gpus.split(',') if g.strip()]
    if len(gpus) not in (1,len(arms)):
        raise ValueError(f'--gpus needs 1 or {len(arms)} IDs')
    if not (root/'bake_predobj_relproto_w_into_projector_final.py').is_file():
        raise FileNotFoundError('Cannot find existing bake script; run from repository root')
    for arm in arms:
        for ep in epochs:
            w=root/ROOT_EXPS/ARMS[arm][0]/f'W_epoch_{ep:03d}.pt'
            if not w.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {w}')
    log(f'[START] {len(epochs)} epochs x {len(arms)} arms; GPU assignment={gpus}')
    log('[NOTE] 850-image full forward inference per missing checkpoint, may take hours.')
    if len(gpus)==1:
        for i,arm in enumerate(arms):
            evaluate_arm(root,output,arm,epochs,gpus[0],i,args.keep_baked)
    else:
        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            jobs={pool.submit(evaluate_arm,root,output,arm,epochs,gpus[i],i,args.keep_baked):arm
                  for i,arm in enumerate(arms)}
            for future in as_completed(jobs):
                future.result()
    summarize(output,arms)

if __name__=='__main__':
    main()

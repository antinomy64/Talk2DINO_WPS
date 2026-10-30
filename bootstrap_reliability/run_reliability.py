#!/usr/bin/env python3
"""One GPU sequential, provenance protected training + official PP116 evaluation."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,subprocess,sys,zipfile
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
TRAIN=HERE/'train_pp116_bootstrap_reliability.py'
REF=HERE/'reference_softsingle_dual.py'
DATA='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'
TEXT='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
WEIGHTS='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
CONFIG='configs/vitb_mlp_infonce.yaml'
BAKE='bake_predobj_relproto_w_into_projector_final.py'
EVALUATOR='src/open_vocabulary_segmentation/main.py'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def variants():
    return [
      ('baseline_global',{'evidence':'none','nce_mode':'global'},'Exact Ours 116-way control'),
      ('temporal_only',{'evidence':'temporal'},'Time consistency + support'),
      ('cross_only',{'evidence':'cross'},'Cross-crop opposite-fold anchors + support'),
      ('temporal_cross',{'evidence':'temporal,cross'},'Time + cross-crop consensus'),
      ('temporal_cross_perturb',{'evidence':'temporal,cross,perturb'},'Add deterministic perturbation'),
      ('temporal_cross_perturb_trend',{'evidence':'temporal,cross,perturb,trend'},'Protect recovering parts'),
      ('hard12_control',{'evidence':'none','nce_mode':'hard12'},'Best prior hard-negative single-factor control'),
      ('hard12_selective',{'evidence':'temporal,cross,perturb,trend','nce_mode':'hard12'},'Exploratory composition after standalone comparisons'),
    ]

def emit(s,log):
    line=f'[{datetime.now().isoformat(timespec="seconds")}] {s}'
    print(line,flush=True)
    with log.open('a',encoding='utf8') as f:f.write(line+'\n')

def run(cmd,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as out:
        out.write('ARGS: '+json.dumps([str(x) for x in cmd])+'\n');out.flush()
        returncode=subprocess.run([str(x) for x in cmd],cwd=Path.cwd(),env=env,stdout=out,stderr=subprocess.STDOUT).returncode
        out.write(f'\nEXIT_CODE={returncode}\n')
    if returncode:
        raise RuntimeError(f'Command failed, see {log}\n'+ '\n'.join(log.read_text(errors='replace').splitlines()[-35:]))

def parse_iou(raw):
    match=re.search(r'mIoU of 850 test images:\s*([0-9.]+)%',raw)
    return float(match.group(1)) if match else None

def parse_perclass(raw):
    records=[]
    in_table=False
    for line in raw.splitlines():
        if 'per class results:' in line:in_table=True;continue
        if in_table and 'Summary:' in line:break
        if in_table and line.strip().startswith('|') and 'Class' not in line and 'IoU' not in line:
            fields=[x.strip() for x in line.strip().split('|')[1:-1]]
            if len(fields)==3:
                try:records.append({'part':fields[0],'iou':float(fields[1]),'acc':float(fields[2])})
                except ValueError:pass
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','core','all'),default='core')
    p.add_argument('--gpu',default='3')
    p.add_argument('--tag',default='pp116_bootstrap_reliability_20261027_v1')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--eval-epochs',default='3,10,20,30')
    p.add_argument('--skip-eval',action='store_true')
    p.add_argument('--master-port',type=int,default=32857)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]{4,70}',args.tag):raise ValueError('Invalid tag')
    if not args.gpu.isdigit():raise ValueError('Invalid GPU')
    if args.epochs<1:raise ValueError('epochs must be positive')
    eps=[int(x) for x in args.eval_epochs.split(',') if x]
    if not args.skip_eval and any(x<1 or x>args.epochs for x in eps):
        raise ValueError('Eval epochs must fall within training length')
    specs=variants()
    if args.profile=='smoke':specs=specs[:1]+specs[3:4]
    if args.profile=='core':specs=specs[:6]
    label=f'{args.tag}_{args.profile}_e{args.epochs}'
    ckroot=Path('final_exp/relproto_alignment')/label
    out=Path('output')/label
    out.mkdir(parents=True,exist_ok=True);ckroot.mkdir(parents=True,exist_ok=True)
    status=out/'run_status.log'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED='1',PYTHONPATH=str(Path.cwd())+os.pathsep+env.get('PYTHONPATH',''))
    for f in [TRAIN,REF,'train_relproto_alignemt.py',DATA,TEXT,CONFIG,WEIGHTS]:
        if not Path(f).is_file():raise FileNotFoundError(f)
    if not args.skip_eval:
        for f in [BAKE,EVALUATOR]:
            if not Path(f).is_file():raise FileNotFoundError(f)
    provenance={'version':1,'label':label,'training_script_sha256':digest(TRAIN),
        'reference_sha256':digest(REF),'cache':DATA,'text_bank':TEXT,'projector':WEIGHTS,
        'protocol':{'supervision':'image-level part Presence; no GT masks in training','epochs':args.epochs,
        'seed':123,'lr':1e-5,'batch_size':16,'soft_temp':.05,'nce_temp':.07,
        'structure_lambda':4.,'rank_delta':.05,'rank_margin':.05},
        'variants':[{'name':n,'overrides':o,'hypothesis':h} for n,o,h in specs],'eval_epochs':eps,'skip_eval':args.skip_eval}
    prov=out/'provenance.json'
    if prov.exists() and json.loads(prov.read_text())!=provenance:
        raise RuntimeError('Provenance changed: refusing to mix experiments in same output folder')
    if not prov.exists():prov.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
    emit(f'START {label}, variants={len(specs)}, epochs={args.epochs}, GPU={args.gpu}',status)
    run([sys.executable,HERE/'selftest_reliability.py'],out/'selftest.log',env)
    rec=[]; perclass=[]
    for j,(name,over,description) in enumerate(specs,1):
        ckdir=ckroot/name
        final=ckdir/f'W_epoch_{args.epochs:03d}.pt'
        finished=ckdir/'summary.json'
        if final.exists() and finished.exists():
            emit(f'REUSE training {j}/{len(specs)} {name}',status)
        else:
            if ckdir.exists():
                raise RuntimeError(f'Incomplete checkpoint dir {ckdir}; refusing to overwrite. Inspect and rename it before rerun.')
            emit(f'TRAIN {j}/{len(specs)} {name}: {description}',status)
            cmd=[sys.executable,'-u',TRAIN,'--part_setting','presence','--project_root','.',
                '--train_dataset',DATA,'--text_bank',TEXT,'--model_config',CONFIG,
                '--weights',WEIGHTS,'--out_dir',ckdir,'--soft_temperature','0.05',
                '--nce_temperature','0.07','--lambda_struct','4.0','--rank_delta','0.05',
                '--rank_margin','0.05','--epochs',args.epochs,'--batch_size','16',
                '--lr','1e-5','--seed','123','--expected_bg_thresh','0.55',
                '--require_pamr','--require_cache_projector_match','--device','cuda']
            for key,value in over.items():cmd.extend([f'--{key}',str(value)])
            run(cmd,out/'train'/f'{name}.log',env)
            if not (final.exists() and finished.exists()):raise RuntimeError('Training lacks final checkpoint/summary')
            emit(f'TRAIN COMPLETED {name}',status)
        if args.skip_eval:continue
        for ep in eps:
            evaldir=out/'val'/name/f'e{ep:03d}'
            evaldir.mkdir(parents=True,exist_ok=True)
            eval_log=evaldir/'evaluation_stdout.log'
            done=evaldir/'EVALUATION_DONE'
            if done.exists() and eval_log.exists() and parse_iou(eval_log.read_text(errors='replace')) is not None:
                emit(f'REUSE EVAL {name} E{ep}',status)
            else:
                project=f'vitb_mlp_infonce_coco2014_reproduce_clean_br_{hashlib.sha256(label.encode()).hexdigest()[:7]}_{name}_e{ep:03d}'
                baked=Path('weights')/(project+'.pth')
                if not baked.exists():
                    emit(f'BAKE {name} E{ep}',status)
                    run([sys.executable,BAKE,'--project_root','.','--projector',WEIGHTS,
                         '--w_checkpoint',ckdir/f'W_epoch_{ep:03d}.pt','--text_bank',TEXT,
                         '--model_config',CONFIG,'--output',baked,'--device','cuda'],
                        out/'bake'/f'{name}_e{ep:03d}.log',env)
                emit(f'OFFICIAL EVAL {name} E{ep}',status)
                run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1',
                    '--master_port',args.master_port,EVALUATOR,'--eval','--output',
                    evaldir/'official_output','--eval_cfg',
                    'src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
                    '--eval_base_cfg','src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
                    '--opts',f'model.proj_name={project}','evaluate.pamr=false'],eval_log,env)
                if parse_iou(eval_log.read_text(errors='replace')) is None:
                    raise RuntimeError('Official evaluation lacks 850 image summary')
                done.write_text('completed\n')
                emit(f'EVAL COMPLETED {name} E{ep}',status)
            txt=eval_log.read_text(errors='replace')
            rec.append({'variant':name,'epoch':ep,'miou':parse_iou(txt)})
            for pc in parse_perclass(txt):perclass.append({'variant':name,'epoch':ep,**pc})
        with (out/'scorecard.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=('variant','epoch','miou'));writer.writeheader();writer.writerows(rec)
    if rec:
        for filename,rows,fields in [('scorecard.csv',rec,['variant','epoch','miou']),
                                      ('per_part_official.csv',perclass,['variant','epoch','part','iou','acc'])]:
            with (out/filename).open('w',newline='',encoding='utf8') as fp:
                wr=csv.DictWriter(fp,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    report_zip=out/'review_upload.zip'
    with zipfile.ZipFile(report_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f!=report_zip and (
                f.name in {'provenance.json','scorecard.csv','per_part_official.csv','run_status.log','selftest.log','evaluation_stdout.log'} or
                (f.name.endswith('.log') and 'train' in f.parts)):
                z.write(f,f.relative_to(out))
        for name,_,_ in specs:
            f=ckroot/name/'training_history.csv'
            if f.exists():z.write(f,'training_history/'+name+'.csv')
    emit(f'ALL DONE. UPLOAD {report_zip}',status)

if __name__=='__main__':main()

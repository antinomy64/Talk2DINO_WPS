#!/usr/bin/env python3
"""Sequential, crash-safe experimental training and official evaluation.

Source modifications are limited to train_pp116_pipeline_search.py. This
scheduler does not modify upstream trainers, training inputs or evaluation code.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time, csv, shutil, math
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAIN=HERE/'train_pp116_hardest.py'
DATA='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'
BANK='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
PROJECTOR='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
CONFIG='configs/vitb_mlp_infonce.yaml'
BAKER='bake_predobj_relproto_w_into_projector_final.py'
EVALUATOR='src/open_vocabulary_segmentation/main.py'

def variants():
    # Each tuple: identifier, ONLY the overrides, explicit falsifiable hypothesis.
    # Hardest-16 reproduces the original ablation; baseline is exact 116-class CE.
    core=[
        ('baseline_116',{'nce_variant':'global'},'original 116-way NCE control'),
        ('hard_k04',{'nce_variant':'hardneg','hardneg_k':4},'very concentrated negatives'),
        ('hard_k08',{'nce_variant':'hardneg','hardneg_k':8},'smaller than 16'),
        ('hard_k12',{'nce_variant':'hardneg','hardneg_k':12},'moderate negatives'),
        ('hard_k16',{'nce_variant':'hardneg','hardneg_k':16},'reproduce 37.32% historical Hardest16'),
        ('hard_k24',{'nce_variant':'hardneg','hardneg_k':24},'larger than 16'),
        ('hard_k32',{'nce_variant':'hardneg','hardneg_k':32},'broader hard negatives'),
        ('hard_k48',{'nce_variant':'hardneg','hardneg_k':48},'broad candidate set'),
        ('hard_k64',{'nce_variant':'hardneg','hardneg_k':64},'very broad candidates'),
        ('hard_k96',{'nce_variant':'hardneg','hardneg_k':96},'near-global control'),
        ('hard_k115',{'nce_variant':'hardneg','hardneg_k':115},'global NCE exact equivalence'),
        ('random_k16',{'nce_variant':'hardneg_random','hardneg_k':16},'test hard mining vs same quantity of random negatives'),
        ('static_text_k16',{'nce_variant':'hardneg_textstatic','hardneg_k':16},'text-only negative list versus sample-specific image-text ranking'),
    ]
    mechanism=[
        ('easy_k16',{'nce_variant':'hardneg_easiest','hardneg_k':16},'easy-only negative control'),
        ('quota_k16_local02',{'nce_variant':'hardneg_quota','hardneg_k':16,'hardneg_local_quota':2},'force 2 local negatives then fill globally'),
        ('quota_k16_local04',{'nce_variant':'hardneg_quota','hardneg_k':16,'hardneg_local_quota':4},'force 4 local negatives then fill globally'),
        ('quota_k16_local08',{'nce_variant':'hardneg_quota','hardneg_k':16,'hardneg_local_quota':8},'force up to 8 local negatives then fill globally'),
        ('quota_k32_local04',{'nce_variant':'hardneg_quota','hardneg_k':32,'hardneg_local_quota':4},'larger global pool with local quota'),
        ('samename_down05',{'nce_variant':'hardneg_samename','hardneg_k':16,'same_name_weight':0.5},'reduce cross-object same-part-name negative coefficient'),
        ('samename_down025',{'nce_variant':'hardneg_samename','hardneg_k':16,'same_name_weight':0.25},'stronger semantic same-name discount'),
        ('schedule_115_to_16_linear',{'nce_variant':'hardneg','hardneg_start':115,'hardneg_k':16,'hardneg_schedule':'linear'},'progressive focusing on hardest negatives'),
        ('schedule_115_to_16_cosine',{'nce_variant':'hardneg','hardneg_start':115,'hardneg_k':16,'hardneg_schedule':'cosine'},'smooth difficulty annealing'),
        ('schedule_115_warm5_16',{'nce_variant':'hardneg','hardneg_k':16,'hardneg_schedule':'delayed','hardneg_warmup_epochs':5},'global NCE warmup then hard16'),
        ('schedule_16_to_115',{'nce_variant':'hardneg','hardneg_start':16,'hardneg_k':115,'hardneg_schedule':'linear'},'reverse curriculum control'),
        ('hard16_structure2',{'nce_variant':'hardneg','hardneg_k':16,'lambda_struct':2.0},'loss balance interaction with hard-negative selection'),
        ('hard16_structure6',{'nce_variant':'hardneg','hardneg_k':16,'lambda_struct':6.0},'larger structural weight interaction'),
        ('hard16_globalmix005',{'nce_variant':'hardneg_mixglobal','hardneg_k':16,'hardneg_mix_global':0.05},'retain 5% full global CE'),
        ('hard16_globalmix015',{'nce_variant':'hardneg_mixglobal','hardneg_k':16,'hardneg_mix_global':0.15},'retain 15% full global CE'),
        ('hard16_globalmix030',{'nce_variant':'hardneg_mixglobal','hardneg_k':16,'hardneg_mix_global':0.30},'retain 30% full global CE'),
        ('softneg_k16_tau03',{'nce_variant':'hardneg_softgate','hardneg_k':16,'hardneg_gate_tau':0.3},'soft threshold negative weights around 16th-hardest'),
        ('softneg_k16_tau10',{'nce_variant':'hardneg_softgate','hardneg_k':16,'hardneg_gate_tau':1.0},'smoother negative weights around 16th-hardest'),
    ]
    return core,mechanism

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def emit(msg,log):
    stamp=f'[{datetime.now().isoformat(timespec="seconds")}] {msg}'
    print(stamp,flush=True)
    with log.open('a',encoding='utf8') as f:f.write(stamp+'\n')

def execute(argv,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as f:
        f.write('ARGS: '+json.dumps([str(x) for x in argv],ensure_ascii=False)+'\n');f.flush()
        rc=subprocess.run([str(x) for x in argv],cwd=Path.cwd(),env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        f.write('\nEXIT_CODE: '+str(rc)+'\n')
    if rc:
        tail='\n'.join(log.read_text(errors='replace').splitlines()[-35:])
        raise RuntimeError(f'Failed: {log}\n{tail}')

def recover_completed_nan_summary(ckdir, train_log, epochs, status_log):
    """Recover a fully trained run whose *last JSON write* rejected NaN diagnostics.

    This neither trains nor rewrites checkpoints/CSV. It is intentionally
    forbidden for OOM, failed gradients, partial epochs, or malformed weights.
    """
    import torch
    final = ckdir / f'W_epoch_{epochs:03d}.pt'
    last = ckdir / 'W_last.pt'
    summary = ckdir / 'summary.json'
    csv_path = ckdir / 'training_history.csv'
    if summary.exists():
        raise RuntimeError('Refuse to overwrite existing summary: '+str(summary))
    if not all(p.is_file() for p in (final, last, csv_path, train_log)):
        raise RuntimeError('Cannot recover: checkpoint, W_last, CSV or train log missing in '+str(ckdir))
    raw = train_log.read_text(encoding='utf8', errors='replace')
    marker = 'Out of range float values are not JSON compliant: nan'
    if marker not in raw:
        raise RuntimeError('Refuse recovery: trainer did not fail at NaN JSON serialization in '+str(train_log))
    if not re.search(rf'\[epoch\]\s*0*{epochs}/0*{epochs}\b', raw):
        raise RuntimeError('Refuse recovery: no successful final epoch message in '+str(train_log))

    # torch.load is used solely on our own locally generated, trusted training files.
    saved = torch.load(final, map_location='cpu', weights_only=False)
    saved_last = torch.load(last, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved_last, dict):
        raise RuntimeError('Checkpoint payload is not a mapping')
    if int(saved.get('epoch', -1)) != epochs or int(saved_last.get('epoch', -1)) != epochs:
        raise RuntimeError('Checkpoint epoch mismatch; recovery aborted')
    if int(saved.get('global_step', -1)) < epochs or saved.get('global_step') != saved_last.get('global_step'):
        raise RuntimeError('Checkpoint global-step inconsistency; recovery aborted')

    with csv_path.open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != epochs or [int(float(row['epoch'])) for row in rows] != list(range(1,epochs+1)):
        raise RuntimeError('Training CSV does not have all consecutive epochs; recovery aborted')

    history = saved.get('history')
    if not isinstance(history, list) or len(history) != epochs:
        raise RuntimeError('Checkpoint training history missing or incomplete; recovery aborted')
    if [int(row['epoch']) for row in history] != list(range(1,epochs+1)):
        raise RuntimeError('Checkpoint epoch history inconsistent; recovery aborted')

    # Never silently hide a broken optimizer, nonfinite objective, or nonfinite W.
    mandatory = ('loss_total','loss_nce','loss_relation','mean_grad_norm','orthogonality_max_abs')
    nonfinite=[]
    safe_history=[]
    for row in history:
        for field in mandatory:
            val=float(row[field])
            if not math.isfinite(val):
                raise RuntimeError(f'Nonfinite core training value E{row["epoch"]} {field}: {val}')
        clean={}
        for field, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                if field in mandatory:
                    raise RuntimeError('Nonfinite critical training value')
                nonfinite.append({'epoch':int(row['epoch']), 'field':field, 'value':str(value)})
                clean[field]=None
            else:
                clean[field]=value
        safe_history.append(clean)

    # Check the actual learned W is finite; avoid accepting a broken checkpoint.
    found_w=[]
    def collect_w(item, depth=0):
        if depth>3 or len(found_w)>3:return
        if torch.is_tensor(item) and tuple(item.shape)==(768,768):
            found_w.append(item)
        elif isinstance(item,dict):
            for key,value in item.items():
                if isinstance(key,str) and key.lower() in ('optimizer','optimizer_state_dict','history'):
                    continue
                collect_w(value,depth+1)
    collect_w(saved)
    if not found_w:
        raise RuntimeError('Unable to locate trained 768x768 W in checkpoint; recovery aborted')
    if not all(bool(torch.isfinite(w).all()) for w in found_w):
        raise RuntimeError('Nonfinite learned W in checkpoint; recovery aborted')
    if not nonfinite:
        raise RuntimeError('JSON failed but no nonfinite history metrics found; investigate before recovery')

    audit={'recovery_reason':'JSON serialization rejected nonfinite diagnostic(s) AFTER final checkpoint',
           'trained_epochs':epochs,'global_step':int(saved['global_step']),
           'nonfinite_diagnostics':nonfinite,
           'source_training_log':str(train_log),
           'original_csv_and_checkpoints_unchanged':True,
           'warning':'NaN diagnostics require inspection; this is NOT evidence that the variant is valid'}
    audit_path=ckdir/'summary_recovery_audit.json'
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    result={'status':'completed', 'version':saved.get('version','unknown'),
            'checkpoint':'W_last.pt','epochs':epochs,
            'global_step':int(saved['global_step']), 'history':safe_history,
            'recovered_after_json_nan':True,
            'recovery_audit':audit_path.name}
    tmp=summary.with_name(summary.name+'.recovery_tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(summary)
    emit(f'RECOVERED completed {ckdir.name}: checkpoint E{epochs}, '
         f'{len(nonfinite)} nonfinite diagnostic entries (see {audit_path})',status_log)
    return True


def valid_eval(log):
    if not log.exists():return False
    s=log.read_text(errors='replace')
    return bool(re.search(r'mIoU of 850 test images:\s*[0-9.]+%',s)) and 'per class results:' in s

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('pilot','core','mechanism','all'),default='all')
    p.add_argument('--gpu',default='0')
    p.add_argument('--tag',default='pp116_hardest_deep_20261024_v1')
    p.add_argument('--epochs',type=int,default=30,help='Training epochs for EVERY variant; default 30')
    p.add_argument('--eval-epochs',default='',help='comma-separated; default E3/E10/E20/E30 for 30-epoch runs')
    p.add_argument('--max-variants',type=int,default=0,help='0=all in profile')
    p.add_argument('--master-port',type=int,default=32823)
    p.add_argument('--skip-eval',action='store_true')
    p.add_argument('--eval-only',action='store_true',help='do not train, only use completed matching checkpoints')
    args=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]{4,60}',args.tag):raise ValueError('unsafe tag')
    if not re.fullmatch('[0-9]+',args.gpu):raise ValueError('GPU must be integer index')
    root=Path.cwd()
    for path in [TRAIN, 'train_relproto_alignemt.py','train_pp116_softsingle_dual.py',DATA,BANK,PROJECTOR,CONFIG]:
        if not Path(path).is_file():raise FileNotFoundError(path)
    if not args.skip_eval:
        for path in [BAKER,EVALUATOR]:
            if not Path(path).is_file():raise FileNotFoundError(path)
    single,interaction=variants()
    if args.profile=='pilot': specs=[single[0],single[4]]
    elif args.profile=='core': specs=single
    elif args.profile=='mechanism': specs=interaction
    else:specs=single+interaction
    if args.max_variants>0:specs=specs[:args.max_variants]
    epochs=args.epochs
    if epochs<1:raise ValueError('epochs >=1')
    epaths=args.eval_epochs or (
        '3,10,20,30' if epochs==30 else str(epochs)
    )
    eval_epochs=[int(x) for x in epaths.split(',') if x.strip()]
    if any(ep<1 or ep>epochs for ep in eval_epochs):raise ValueError('bad eval epoch for requested training length')
    # Keep profiles isolated to avoid mixing independent run histories.
    # A 100-epoch model always gets an _e100 output directory.
    label=f'{args.tag}_{args.profile}_e{epochs}'
    out_root=Path('final_exp/relproto_alignment')/label
    report=Path('output')/label
    report.mkdir(parents=True,exist_ok=True);out_root.mkdir(parents=True,exist_ok=True)
    status=report/'run_status.log'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED='1',PYTHONPATH=str(root)+os.pathsep+env.get('PYTHONPATH',''))
    emit(f'START profile={args.profile} epochs={epochs} variants={len(specs)} eval_epochs={eval_epochs}',status)
    emit('TOTAL REQUESTED TRAINING EPOCHS: '+str(len(specs)*epochs)+' (sequential, one GPU)',status)
    # Mandatory equivalence and all-variant gradient checks before any training.
    execute([sys.executable,HERE/'selftest_hardest.py'],report/'preflight_selftest.log',env)
    source_hash=sha(TRAIN)
    prov=report/'provenance.json'
    payload={'label':label,'source_sha256':source_hash,'original_sha256':sha('train_pp116_softsingle_dual.py'),
        'train_cache':DATA,'text_bank':BANK,'frozen_projector':PROJECTOR,
        'training_protocol':{'part_setting':'presence','epochs':epochs,'seed':123,'batch_size':16,
                             'lr':1e-5,'temperature_soft':0.05,'temperature_nce':0.07,
                             'structure_lambda':4.0,'delta':0.05,'margin':0.05},
        'protocol':'Only image-level Presence labels, frozen cached DINOv2, no spatial GT, official validation only after checkpoint',
        'variants':[{'name':name,'overrides':over,'hypothesis':why} for name,over,why in specs],
        'fixed_probe':{'enabled':True,'count_per_part':3,'snapshots':[ep for ep in (0,1,3,5,10,20,30) if ep<=epochs]}}
    if prov.exists():
        previous=json.loads(prov.read_text())
        if previous!=payload:raise RuntimeError('Existing provenance differs: refuse incompatible resume '+str(prov))
    else:prov.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    records=[]
    for ordinal,(name,over,why) in enumerate(specs):
        ckdir=out_root/name
        final=ckdir/f'W_epoch_{epochs:03d}.pt'
        complete=ckdir/'summary.json'
        if final.is_file() and complete.is_file():
            emit(f'REUSE trained {name}: {final}',status)
        elif ckdir.exists():
            # Only recover runs that finished every epoch and failed while
            # serializing NaN *diagnostic fields* at the final summary write.
            recover_completed_nan_summary(ckdir,report/'train'/f'{name}.log',epochs,status)
            emit(f'REUSE recovered training {name}: {final}',status)
        elif args.eval_only:
            raise RuntimeError(f'Checkpoint missing in eval-only mode: {final}')
        else:
            emit(f'TRAIN {name} ({ordinal+1}/{len(specs)}), {why}',status)
            argv=[sys.executable,'-u',TRAIN,'--part_setting','presence','--project_root','.',
              '--train_dataset',DATA,'--text_bank',BANK,'--model_config',CONFIG,
              '--weights',PROJECTOR,'--out_dir',ckdir,
              '--soft_temperature','0.05','--nce_temperature','0.07',
              '--lambda_struct','4.0','--rank_delta','0.05','--rank_margin','0.05',
              '--epochs',epochs,'--batch_size','16','--lr','1e-5','--seed','123',
              '--expected_bg_thresh','0.55','--require_pamr','--require_cache_projector_match','--device','cuda',
              '--probe','--probe_per_part','3','--probe_epochs',','.join(str(ep) for ep in (0,1,3,5,10,20,30) if ep<=epochs)]
            for k,v in over.items():
                if isinstance(v,bool):
                    if v:argv.append('--'+k)
                else:argv.extend(['--'+k,str(v)])
            try:
                execute(argv,report/'train'/f'{name}.log',env)
            except RuntimeError:
                # Fatal training errors are still fatal. Recovery requires a
                # complete E30 checkpoint and the exact final-JSON NaN error.
                recover_completed_nan_summary(ckdir,report/'train'/f'{name}.log',epochs,status)
            if not (final.exists() and complete.exists()):raise RuntimeError('missing checkpoint after training: '+str(final))
            emit(f'TRAIN COMPLETED {name}',status)
        rec={'variant':name,'epochs':epochs,'train_dir':str(ckdir),'hypothesis':why,'overrides':over}
        records.append(rec)
        (report/'registry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
        if args.skip_eval:continue
        for ep in eval_epochs:
            evaldir=report/'val'/name/f'e{ep:03d}'
            evaldir.mkdir(parents=True,exist_ok=True)
            log=evaldir/'evaluation_stdout.log'
            if (evaldir/'EVALUATION_DONE').exists() and valid_eval(log):
                emit(f'REUSE EVAL {name} E{ep}',status)
                continue
            run_uid=hashlib.sha256(label.encode('utf-8')).hexdigest()[:8]
            project_name=f'vitb_mlp_infonce_coco2014_reproduce_clean_ps_{run_uid}_{name}_e{ep:03d}'
            baked=Path('weights')/(project_name+'.pth')
            if not baked.exists():
                emit(f'BAKE {name} E{ep}',status)
                execute([sys.executable,BAKER,'--project_root','.','--projector',PROJECTOR,
                         '--w_checkpoint',ckdir/f'W_epoch_{ep:03d}.pt',
                         '--text_bank',BANK,'--model_config',CONFIG,'--output',baked,'--device','cuda'],
                        report/'bake'/f'{name}_e{ep:03d}.log',env)
            emit(f'OFFICIAL EVAL {name} E{ep}',status)
            execute([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1',
                     '--master_port',str(args.master_port),EVALUATOR,
                     '--eval','--output',evaldir/'official_output',
                     '--eval_cfg','src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
                     '--eval_base_cfg','src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
                     '--opts',f'model.proj_name={project_name}','evaluate.pamr=false'],log,env)
            if not valid_eval(log):raise RuntimeError(f'EVAL log lacks official 850-image mIoU: {log}')
            (evaldir/'EVALUATION_DONE').write_text(f'official complete {name} E{ep}\n')
            emit(f'EVAL COMPLETED {name} E{ep}',status)
    execute([sys.executable,HERE/'summarize_hardest.py','--report',report],report/'summarize.log',env)
    emit(f'ALL DONE {report} (summarized)',status)
    print('UPLOAD:',report/'review_upload.zip',flush=True)

if __name__=='__main__':main()

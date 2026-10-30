#!/usr/bin/env python3
"""Safe sequential overnight PP116 Ear transfer experiment runner.

No training GT. Existing checkpoints never overwritten. Resume completed only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from ear_transfer_summarize import parse_eval

EAR = ('cat','cow','dog','horse','person','sheep')
SCRIPT = Path(__file__).resolve().parent
CACHE='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'
BANK='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
PROJECTOR='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
BASELINE='final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'
PRIOR='final_exp/relproto_alignment/ear_mechanism_20261021_v1'
PRIOR_REPORT='output/ear_mechanism_20261021_v1'
CONTROLS={
  'full_l4':('full','cow',4.,BASELINE,None),
  'full_l0':('full','cow',0.,PRIOR+'/full_nostruct','full_nostruct'),
  'non_ear_l4':('non_ear','cow',4.,PRIOR+'/drop_ear_struct','drop_ear_struct'),
  'non_ear_l0':('non_ear','cow',0.,PRIOR+'/drop_ear_nostruct','drop_ear_nostruct'),
  'all_ears_l4':('all_ears','cow',4.,PRIOR+'/only_ear_struct','only_ear_struct'),
  'all_ears_l0':('all_ears','cow',0.,PRIOR+'/only_ear_nostruct','only_ear_nostruct'),
}

def selected(profile):
    specs={}
    def add(t,mode,lam):
        prefix={'drop_target':'loo','target_plus_non_ear':'target_nonear','other_five_ears':'other5','only_target':'only'}[mode]
        specs[f'{prefix}_{t}_l{int(lam)}']=(mode,t,float(lam),None,None)
    if profile in ('core','extended','cow'):
        if profile!='cow':
            for t in EAR:
                add(t,'drop_target',4)
                add(t,'target_plus_non_ear',4)
        else:
            add('cow','drop_target',4);add('cow','target_plus_non_ear',4)
        for mode in ('other_five_ears','only_target'):
            add('cow',mode,4)
        for mode in ('drop_target','target_plus_non_ear','other_five_ears','only_target'):
            add('cow',mode,0)
    if profile=='extended':
        for t in EAR:
            if t!='cow':
                for mode in ('other_five_ears','only_target'):
                    add(t,mode,4)
                for mode in ('drop_target','target_plus_non_ear','other_five_ears','only_target'):
                    add(t,mode,0)
    if profile=='smoke':
        add('cow','drop_target',4)
    return specs

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def completed(run):
    return (run/'W_epoch_010.pt').is_file() and (run/'summary.json').is_file()

def emit(message,status_file):
    msg=f'[{datetime.now().isoformat(timespec="seconds")}] {message}'
    print(msg,flush=True)
    with open(status_file,'a',encoding='utf8') as f:f.write(msg+'\n')

def execute(cmd,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as f:
        f.write('COMMAND (argv): '+json.dumps([str(v) for v in cmd],ensure_ascii=False)+'\n')
        f.flush()
        proc=subprocess.run([str(v) for v in cmd],cwd=Path.cwd(),stdout=f,stderr=subprocess.STDOUT,env=env)
        f.write('\nEXIT_CODE: '+str(proc.returncode)+'\n')
    if proc.returncode:
        tail='\n'.join(log.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'COMMAND FAILED ({proc.returncode}): {log}\n{tail}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=('core','extended','cow','smoke'),default='core')
    ap.add_argument('--tag',default='pp116_ear_transfer_20261022_v1')
    ap.add_argument('--gpu',default='0')
    ap.add_argument('--eval-epochs',default='3,10',help='comma list 1..10')
    ap.add_argument('--skip-train',action='store_true')
    ap.add_argument('--skip-eval',action='store_true')
    ap.add_argument('--skip-probe',action='store_true')
    ap.add_argument('--gradient-baseline',action='store_true',help='baseline E0/E1/E3/E10 gradient attribution for 6 target Ears')
    ap.add_argument('--max-variants',type=int,default=0,help='smoke/resumability; 0=all')
    args=ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_]{3,48}',args.tag):raise ValueError('tag must be 3-48 safe chars')
    if not (Path.cwd()/'train_relproto_alignemt.py').exists():
        raise RuntimeError('Run from Talk2DINO_official_bg repository root')
    for name in ('train_pp116_softsingle_dual.py',CACHE,BANK,PROJECTOR,
        'configs/vitb_mlp_infonce.yaml','bake_predobj_relproto_w_into_projector_final.py',
        'src/open_vocabulary_segmentation/main.py'):
        if not Path(name).is_file():raise FileNotFoundError(name)
    report=Path('output')/args.tag
    outroot=Path('final_exp/relproto_alignment')/args.tag
    report.mkdir(parents=True,exist_ok=True);outroot.mkdir(parents=True,exist_ok=True)
    status=report/'run_status.log'
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=args.gpu;env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=str(Path.cwd())+os.pathsep+env.get('PYTHONPATH','')
    env['OMP_NUM_THREADS']=env.get('OMP_NUM_THREADS','4')
    base=Path(BASELINE)
    if not completed(base):raise FileNotFoundError('Missing reference completed baseline '+str(base))
    execute([sys.executable,SCRIPT/'test_transfer.py'],report/'test_transfer.log',env)
    control_specs=dict(CONTROLS)
    new_specs=selected(args.profile)
    if args.max_variants>0:new_specs=dict(list(new_specs.items())[:args.max_variants])
    full_specs={**control_specs,**new_specs}
    provenance={
      'tag':args.tag,'profile':args.profile,'created':datetime.now().isoformat(),
      'git_head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
      'script_sha256':{p.name:sha(p) for p in SCRIPT.glob('*.py')},
      'baseline_source':BASELINE,'prior_source':PRIOR,
      'train_cache':CACHE,'text_bank':BANK,'projector':PROJECTOR,
      'fixed_protocol':{'seed':123,'batch':16,'lr':1e-5,'epochs':10,
          'soft_temperature':0.05,'nce_temperature':0.07,'rank_delta':0.05,
          'rank_margin':0.05,'presence':True,'PAMR_evaluation':False,
          'objective_scale':'original 1/K per crop, equal-annotation; masked target terms remain zero, no renorm'},
      'variants':{k:{'mode':v[0],'target':v[1],'lambda':v[2],
           'reference_run_if_available':v[3]} for k,v in full_specs.items()},
      'GT_use':'GT ONLY in offline fixed Crop probes & official validation, NEVER training',
    }
    (report/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    epochs=[int(i.strip()) for i in args.eval_epochs.split(',') if i.strip()]
    if not epochs or any(not 1<=ep<=10 for ep in epochs):raise ValueError('eval-epochs needs values 1..10')
    records=[]
    for key,(mode,target,lam,prior_run,prior_label) in full_specs.items():
        if prior_run and completed(Path(prior_run)):
            run=Path(prior_run)
            emit(f'REUSE completed: {key} from {run}',status)
            reused=True
        else:
            if key=='full_l4':raise FileNotFoundError('Primary baseline cannot be auto retrained')
            run=outroot/key
            reused=False
            if completed(run):
                emit(f'REUSE local completed: {key}',status)
            elif run.exists():
                raise RuntimeError(f'Incomplete run exists, REFUSE mixing: {run}')
            else:
                if args.skip_train:raise FileNotFoundError(f'missing trained run {key}: {run}')
                emit(f'TRAIN {key} 10 epochs; target={target}, mode={mode}, lambda={lam}',status)
                command=[sys.executable,'-u',SCRIPT/'ear_transfer_train.py',
                    '--part_setting','presence','--target_ear',target,'--positive_mode',mode,
                    '--project_root','.','--train_dataset',CACHE,'--text_bank',BANK,
                    '--model_config','configs/vitb_mlp_infonce.yaml',
                    '--weights',PROJECTOR,'--out_dir',run,
                    '--soft_temperature','0.05','--nce_temperature','0.07',
                    '--lambda_struct',str(lam),'--rank_delta','0.05','--rank_margin','0.05',
                    '--epochs','10','--batch_size','16','--lr','1e-5','--seed','123',
                    '--expected_bg_thresh','0.55','--require_pamr',
                    '--require_cache_projector_match','--device','cuda']
                execute(command,report/'train'/f'{key}.log',env)
                if not completed(run):raise RuntimeError('Training exited without final completed output '+str(run))
        rec={'variant':key,'run':str(run),'mode':mode,'target':target,
              'lambda_struct':lam,'reused_prior':reused,'source_label':prior_label or ''}
        records.append(rec)
        with (report/'registry.json').open('w') as f:json.dump(records,f,indent=2)
        if not args.skip_probe:
            probeout=report/'probes'/key
            if (probeout/'fixed_crop_ear_epoch.csv').is_file():
                emit(f'REUSE fixed probe: {key}',status)
            else:
                emit(f'AUDIT fixed six Ear crops: {key}',status)
                command=[sys.executable,'-u',SCRIPT/'ear_transfer_probe.py',
                    '--run',run,'--output',probeout,'--lambda-struct',str(lam),
                    '--epochs','0,1,2,3,4,5,6,7,8,9,10','--device','cuda']
                if key=='full_l4' and args.gradient_baseline:
                    command+=['--gradient-targets',','.join(EAR)]
                execute(command,report/'logs'/f'probe_{key}.log',env)
        if not args.skip_eval:
            for ep in epochs:
                ed=report/'val'/key/f'e{ep:03d}'
                ed.mkdir(parents=True,exist_ok=True)
                done=ed/'EVALUATION_DONE'
                elog=ed/'evaluation_stdout.log'
                if done.exists() and parse_eval(elog):
                    emit(f'REUSE evaluation: {key}, E{ep}',status)
                    continue
                if prior_label is not None:
                    previous=Path(PRIOR_REPORT)/'val'/prior_label/f'e{ep:03d}'/'evaluation_stdout.log'
                    if previous.is_file() and parse_eval(previous):
                        shutil.copyfile(previous,elog)
                        (ed/'source_evaluation.txt').write_text(f'Reused unchanged prior checkpoint evaluation: {previous}\n',encoding='utf8')
                        done.touch()
                        emit(f'REUSE prior official eval {key} E{ep}',status)
                        continue
                checkpoint=run/f'W_epoch_{ep:03d}.pt'
                if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
                project_name=f'vitb_mlp_infonce_coco2014_transfer_{args.tag}_{key}_e{ep:03d}'
                weight=Path('weights')/(project_name+'.pth')
                if not weight.is_file():
                    emit(f'BAKE {key} E{ep}',status)
                    command=[sys.executable,'-u','bake_predobj_relproto_w_into_projector_final.py',
                        '--project_root','.','--projector',PROJECTOR,'--w_checkpoint',checkpoint,
                        '--text_bank',BANK,'--model_config','configs/vitb_mlp_infonce.yaml',
                        '--output',weight,'--device','cuda']
                    execute(command,ed/'bake.log',env)
                    if not weight.is_file():raise RuntimeError('Baked projector not created '+str(weight))
                emit(f'OFFICIAL VAL {key} E{ep}',status)
                command=[sys.executable,'-u','-m','torch.distributed.run',
                    '--nproc_per_node=1','--master_port',os.getenv('MASTER_PORT','32684'),
                    'src/open_vocabulary_segmentation/main.py','--eval','--output',ed,
                    '--eval_cfg','src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
                    '--eval_base_cfg','src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
                    '--opts','model.proj_name='+project_name,'evaluate.pamr=false']
                execute(command,elog,env)
                if not parse_eval(elog):raise RuntimeError(f'No parseable 116-class eval results: {elog}')
                done.touch()
        emit('COMPLETED '+key,status)
    emit('ALL REQUESTED VARIANTS COMPLETED; running summary',status)
    command=[sys.executable,SCRIPT/'ear_transfer_summarize.py','--report',report]
    execute(command,report/'summary.log',env)
    emit('ALL DONE: '+str(report/'summary'),status)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Produce a compact upload ZIP: summaries + plots + gradient stats, NO weights/huge SQLite."""
import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('output/pp116_allpart_audit_ours_v1'))
p.add_argument('--zip',type=Path,default=Path('output/pp116_allpart_mechanism_review_upload.zip'))
a=p.parse_args()
if not (a.out/'summary'/'manifest.json').is_file():raise FileNotFoundError('Run summarize_all_116.py first')
a.zip.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for sub in ('summary','gradient','visuals'):
  d=a.out/sub
  if not d.exists():continue
  for f in d.rglob('*'):
   if not f.is_file():continue
   if f.suffix not in ('.csv','.md','.json','.png','.txt','.log'):continue
   if f.stat().st_size>40*1024**2:continue
   z.write(f,arcname=f'{a.out.name}/{f.relative_to(a.out)}')
 for fname in ('audit_config.json','audit_completion.json'):
  f=a.out/fname
  if f.exists():z.write(f,arcname=f'{a.out.name}/{fname}')
print('READY:',a.zip.resolve(),'MiB:',round(a.zip.stat().st_size/1024**2,2))

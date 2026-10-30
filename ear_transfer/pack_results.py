#!/usr/bin/env python3
"""Create a compact read-only review zip (no weights, caches, NPZ, masks)."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=Path('output/pp116_ear_transfer_20261022_v1'))
    p.add_argument('--out',type=Path,default=Path('output/pp116_ear_transfer_review_upload.zip'))
    a=p.parse_args()
    if not a.report.is_dir():raise FileNotFoundError(a.report)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    keep={'.csv','.json','.log','.txt','.md'}
    excluded={'dense_npz','audit_only_reference','__pycache__'}
    allfiles=[]
    for f in a.report.rglob('*'):
        if not f.is_file() or f.suffix.lower() not in keep or any(part in excluded for part in f.parts):continue
        if f.stat().st_size>25*1024**2:
            print('[SKIP large]',f)
            continue
        allfiles.append(f)
    with ZipFile(a.out,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(allfiles):z.write(f,arcname=str(f.relative_to(a.report.parent)))
        # Code/provenance is crucial to review exact experiment setup.
        here=Path(__file__).resolve().parent
        for f in here.glob('*'):
            if f.is_file() and f.suffix in {'.py','.md','.patch'}:
                z.write(f,arcname='experiment_code/'+f.name)
    print('Saved:',a.out.resolve(),'files:',len(allfiles),'MB:',round(a.out.stat().st_size/2**20,2))

if __name__=='__main__':main()

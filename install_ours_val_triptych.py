#!/usr/bin/env python3
"""Copy original main.py to NEW file and add exactly two read-only capture hooks."""
from pathlib import Path
import ast
import hashlib
import py_compile
import shutil

root=Path('src/open_vocabulary_segmentation')
src=root/'main.py'
dst=root/'main_ours_val_triptych.py'
helper=root/'ours_val_triptych_capture.py'
helper_source=Path(__file__).resolve().parent/'ours_val_triptych_capture.py'
if not src.is_file() or not helper_source.is_file():
    raise FileNotFoundError('Run installer from repository root with helper next to installer')
source=src.read_text(encoding='utf-8')
tree=ast.parse(source)
funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_seg']
if len(funcs)!=1:raise RuntimeError('Expected exactly one validate_seg')
fn=funcs[0]
lines=source.splitlines(keepends=True)
make=[];save=[]
for node in ast.walk(fn):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='build_dinotext_seg_inference':
        make.append(node.end_lineno)
for node in fn.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='miou_result' for t in node.targets):
        save.append(node.end_lineno)
if len(make)!=1 or len(save)!=1 or make[0]>=save[0]:
    raise RuntimeError(f'Unexpected validate_seg structure: build={make}, miou={save}')
print('Original main.py sha256:',hashlib.sha256(src.read_bytes()).hexdigest())
print('Inserting capture at original lines',make[0],'and',save[0])
insert={make[0]: '    from ours_val_triptych_capture import Capture\n    _ours_vis_capture = Capture(seg_model, data_loader.dataset)\n',
        save[0]: '    _ours_vis_capture.finish(miou_result)\n'}
output=''.join(line+insert.get(i,'') for i,line in enumerate(lines,1))
# The original main.py normally saves metrics in segmentation_results/<proj>.json.
# Redirect ONLY this copied evaluator to a dedicated directory: never overwrite
# the existing metrics JSON used to establish provenance of the E100 checkpoint.
if 'segmentation_results' not in output:
    raise RuntimeError('Cannot locate metric JSON destination in main.py; refuse to risk overwriting prior evaluations')
output=output.replace('segmentation_results', 'output/ours_triptych_diagnostic_json')
ast.parse(output)

for p in (dst,helper):
    if p.exists():
        raise FileExistsError(f'Will not overwrite existing file {p}; remove your previous generated file manually after checking it')
dst.write_text(output,encoding='utf-8')
shutil.copy2(helper_source,helper)
py_compile.compile(str(dst),doraise=True)
py_compile.compile(str(helper),doraise=True)
if source != src.read_text(encoding='utf-8'):
    raise AssertionError('Original main.py was modified')
print('[INSTALL PASS] Created:',dst,helper)
print('[ORIGINAL UNMODIFIED] main.py sha256:',hashlib.sha256(src.read_bytes()).hexdigest())

#!/usr/bin/env bash
# Talk2DINO_WPS comprehensive overnight audit
#
# Safety contract:
#   * Never writes to original PascalPart116 images/masks/split files.
#   * Never overwrites an existing cache or checkpoint.
#   * New run artifacts live under output/nightly_audit_<timestamp>/.
#   * Causal-control weights are newly named under weights/.
#
# Default: runs static/dynamic audits PLUS the missing FT0 causal control,
# fixed-cache W comparisons, bake certification, 116/117 eval, and a post-hoc
# GT-only anchor/RelProto purity audit.
#
# Run:
#   cd ~/code/lyx/Talk2DINO_official_bg
#   chmod +x nightly_full_audit.sh
#   ./nightly_full_audit.sh
#
# Optional:
#   RUN_CAUSAL_ABLATION=0 ./nightly_full_audit.sh
#   RUN_LIVE_TEXT_AUDIT=0 ./nightly_full_audit.sh
#   RUN_ANCHOR_AUDIT=0 ./nightly_full_audit.sh
#   ANCHOR_AUDIT_MAX=0 ./nightly_full_audit.sh     # all annotations
#   GPU=1 ./nightly_full_audit.sh

set -uo pipefail
umask 022

ROOT="${ROOT:-$(pwd)}"
GPU="${GPU:-0}"
RUN_CAUSAL_ABLATION="${RUN_CAUSAL_ABLATION:-1}"
RUN_LIVE_TEXT_AUDIT="${RUN_LIVE_TEXT_AUDIT:-1}"
RUN_ANCHOR_AUDIT="${RUN_ANCHOR_AUDIT:-1}"
ANCHOR_AUDIT_MAX="${ANCHOR_AUDIT_MAX:-2000}"
EXPECTED_CACHE_BG_THRESH="${EXPECTED_CACHE_BG_THRESH:-0.54}"
EVAL_BG_THRESH="${EVAL_BG_THRESH:-0.4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29620}"

cd "$ROOT" || exit 2
if [[ ! -f train_relproto_alignemt.py || ! -f train_partstruct_ft.py || ! -f src/loss.py ]]; then
  echo "[FATAL] Run from Talk2DINO_WPS/Talk2DINO_official_bg repository root."
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${AUDIT_OUT:-output/nightly_audit_${TS}}"
mkdir -p "$OUT"/{logs,artifacts,w_runs,eval,preflight}
STATUS="$OUT/status.tsv"
EVAL_TSV="$OUT/eval_results.tsv"
MASTER_LOG="$OUT/nightly_master.log"
: > "$STATUS"
printf 'label\tprotocol\tmiou\tlog\n' > "$EVAL_TSV"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "======================================================================"
echo "Talk2DINO_WPS NIGHTLY FULL AUDIT"
echo "start          : $(date -Is)"
echo "repo           : $ROOT"
echo "output         : $OUT"
echo "GPU            : $GPU"
echo "causal ablation: $RUN_CAUSAL_ABLATION"
echo "live text audit: $RUN_LIVE_TEXT_AUDIT"
echo "anchor audit   : $RUN_ANCHOR_AUDIT (max=$ANCHOR_AUDIT_MAX; 0=all)"
echo "======================================================================"

record_status() {
  local st="$1" name="$2" note="${3:-}"
  note="${note//$'\t'/ }"; note="${note//$'\n'/ }"
  printf '%s\t%s\t%s\n' "$st" "$name" "$note" >> "$STATUS"
}
run_step() {
  local name="$1"; shift
  local log="$OUT/logs/${name}.log"
  echo
  echo "----------------------------------------------------------------------"
  echo "[STEP] $name"
  echo "----------------------------------------------------------------------"
  "$@" > >(tee "$log") 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $name"; record_status PASS "$name" "$log"
  elif [[ $rc -eq 3 ]]; then
    echo "[WARN/SKIP] $name"; record_status WARN "$name" "$log"
  else
    echo "[FAIL rc=$rc] $name"; record_status FAIL "$name" "$log"
  fi
  return 0
}
gpu_ready() {
  CUDA_VISIBLE_DEVICES="$GPU" python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

MODEL_CONFIG="configs/vitb_mlp_infonce.yaml"
TEXT_BANK="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt"
INITIAL_WEIGHT="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
PARTSTRUCT_WEIGHT="weights/vitb_mlp_infonce_coco2014_clean_ft10_partstruct_rawpath_w1e4_lr1e5.pth"
COCO_TRAIN="feature/coco2014_official_reproduce/captions_train2014_ready.pth"
COCO_VAL="feature/coco2014_official_reproduce/captions_val2014_ready.pth"

# Deliberately ONE fixed visual cache for controlled text-side comparison.
FIXED_CACHE="feature/voc116_predobj_cropaug_rawpath/train_predobj_cropaug_bg054.pth"

PARTSTRUCT_EXISTING_W="final_exp/relproto_alignment/partstruct_rawpath_predobj_bg054_cap4_orth/W_last.pt"
EVAL_CFG="src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml"
EVAL116="src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml"
EVAL117="src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part117.yml"

FT0_SUFFIX="coco2014_clean_ft10_ft0_rawpath_lr1e5"
FT0_WEIGHT="weights/vitb_mlp_infonce_${FT0_SUFFIX}.pth"

# ---------------------------------------------------------------------------
# 00 Environment / git snapshot
# ---------------------------------------------------------------------------
environment_audit() {
  python - "$OUT/artifacts/environment.json" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
out=Path(sys.argv[1])
def cmd(*a):
    try: return subprocess.check_output(a,stderr=subprocess.STDOUT,text=True).strip()
    except Exception as e: return f"ERROR: {e}"
d={"python":sys.version,"platform":platform.platform(),"cwd":str(Path.cwd().resolve())}
mods={}
for n in ["torch","torchvision","numpy","scipy","mmcv","mmseg","clip","yaml","pandas"]:
    try:
        m=__import__(n); mods[n]=getattr(m,"__version__","installed")
    except Exception as e: mods[n]=f"ERROR {type(e).__name__}: {e}"
d["modules"]=mods
try:
    import torch
    d["cuda_available"]=bool(torch.cuda.is_available())
    d["cuda_version"]=str(torch.version.cuda)
    d["gpus"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception: pass
d["git_head"]=cmd("git","rev-parse","HEAD")
d["git_branch"]=cmd("git","branch","--show-current")
d["git_status"]=cmd("git","status","--short")
d["git_remote"]=cmd("git","remote","-v")
out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(d,indent=2,ensure_ascii=False))
PY
  git status --short > "$OUT/artifacts/git_status.txt" 2>&1 || true
  git diff --stat > "$OUT/artifacts/git_diff_stat.txt" 2>&1 || true
  git diff > "$OUT/artifacts/git_diff.patch" 2>&1 || true
  nvidia-smi > "$OUT/artifacts/nvidia_smi.txt" 2>&1 || true
}
run_step "00_environment_git" environment_audit

# ---------------------------------------------------------------------------
# 01 Prerequisites
# ---------------------------------------------------------------------------
prerequisite_audit() {
  local bad=0
  local req=(
    "$MODEL_CONFIG" "$TEXT_BANK" "$INITIAL_WEIGHT" "$PARTSTRUCT_WEIGHT"
    "$COCO_TRAIN" "$COCO_VAL"
    "src/loss.py" "src/train_util_partstruct_ft.py" "train_partstruct_ft.py"
    "train_relproto_alignemt.py" "bake_predobj_relproto_w_into_projector_final.py"
    "extract_predobj_cropaug.py" "extract_clip_text_bank.py"
    "$EVAL_CFG" "$EVAL116" "$EVAL117"
    "src/open_vocabulary_segmentation/models/dinotext/dinotext.py"
    "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_builder.py"
    "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py"
    "src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py"
    "src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part117.py"
  )
  for p in "${req[@]}"; do
    if [[ -f "$p" ]]; then echo "[OK]   $p"; else echo "[MISS] $p"; bad=1; fi
  done
  for p in "$FIXED_CACHE" "$PARTSTRUCT_EXISTING_W"; do
    [[ -f "$p" ]] && echo "[OPTIONAL OK] $p" || echo "[OPTIONAL MISS] $p"
  done
  [[ $bad -eq 0 ]]
}
run_step "01_prerequisites" prerequisite_audit

# ---------------------------------------------------------------------------
# 02 Static code/narrative consistency
# ---------------------------------------------------------------------------
static_code_audit() {
python - "$OUT/artifacts/static_code_audit.json" <<'PY'
import json, re, sys
from pathlib import Path
out=Path(sys.argv[1])
def rd(p): return Path(p).read_text(encoding="utf-8")
loss=rd("src/loss.py")
w=rd("train_relproto_alignemt.py")
bake=rd("bake_predobj_relproto_w_into_projector_final.py")
dino=rd("src/open_vocabulary_segmentation/models/dinotext/dinotext.py")
builder=rd("src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_builder.py")
seg=rd("src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py")
cache=rd("extract_predobj_cropaug.py")
readme=rd("README.md") if Path("README.md").is_file() else ""
checks=[]
def add(name,ok,severity,detail):
    checks.append({"name":name,"ok":bool(ok),"severity":severity,"detail":detail})

add("PartStruct projects RAW CLIP",
    "projector.project_clip_txt(" in loss and "self.raw_part_features" in loss,
    "FAIL","PartStructureRankLoss should project the unnormalized prompt mean.")
add("Raw is normalized only for raw-side cosine",
    "raw_for_similarity = F.normalize(raw_part_features" in loss,
    "FAIL","Raw CLIP can be normalized for cosine reference, not before projector.")
add("RelProto projects raw CLIP then normalizes",
    "raw CLIP prompt means are NOT normalized before the projector" in w
    and "projected_bank = project_raw_clip_bank" in w,
    "FAIL","T0 must be normalize(projector(raw_CLIP)).")
mean_i=dino.find("text_embs = text_embs.mean(dim=1).float()")
proj_i=dino.find("self.proj.project_clip_txt(text_embs)",mean_i)
norm_i=dino.find("us.normalize(text_embs",proj_i)
add("Official eval order mean->projector->normalize",
    0 <= mean_i < proj_i < norm_i,"FAIL",
    f"indices mean={mean_i}, proj={proj_i}, norm={norm_i}")
add("Independent anchors; collisions allowed",
    "No global greedy / no one-to-one assignment." in w
    and "anchor_indices = relative.argmax(dim=2)" in w,
    "FAIL","Current method is NOT globally-greedy distinct-anchor.")
add("K=1 degenerates to absolute score",
    "if k == 1:" in w and "relative = absolute.clone()" in w,
    "WARN","Narrative must not claim relative competition for K=1.")
add("Anchor is kept even when R<=0",
    "support[:, :, 0] = anchor_indices" in w and "eligible <= 0.0" in w,
    "WARN","Supplementary support needs R>0; the primary anchor does not.")
add("W is SVD-retracted to orthogonal group",
    "torch.linalg.svd(W" in w and "W.copy_(u @ vh)" in w,
    "FAIL","Orthogonality is a core invariant.")
add("Bake uses W.T on final affine map",
    "W64.T @ A.double()" in bake and "W64.T @ b.double()" in bake,
    "FAIL","Bake direction must match row-vector T@W.")
add("PredObj discards GT spatial arrays before localization",
    "From here onward no GT spatial array is used in any computation." in cache,
    "WARN","GT is still used to derive image-level object/part presence.")
add("Object competitors are image-present GT semantic labels",
    "background + image-present semantic object labels" in cache,
    "WARN","Paper must disclose oracle image-level object presence.")
add("117 builder recognizes leading background",
    'CLASSES[0] == "background"' in builder,
    "FAIL","Needed for with_bg=True.")
add("117 inference prepends background channel",
    "torch.cat([background, masks], dim=1)" in seg,
    "FAIL","Prediction channel 0 must be background.")
add("Strict cache SHA truly requires SHA",
    'if cache_sha and cache_sha != projector_weight_sha256' not in w,
    "WARN","Known loophole: missing cache SHA currently passes strict-match mode.")
if readme:
    drift=("globally-greedy" in readme.lower()
           or "distinct anchor" in readme.lower()
           or "distinct-anchor" in readme.lower())
    add("README matches independent-anchor implementation",
        not drift,"WARN","README still appears to contain stale distinct-anchor narrative.")

fails=[x for x in checks if not x["ok"] and x["severity"]=="FAIL"]
warns=[x for x in checks if not x["ok"] and x["severity"]=="WARN"]
out.write_text(json.dumps({"checks":checks,"failures":fails,"warnings":warns},
                          indent=2,ensure_ascii=False)+"\n")
for x in checks:
    print(f"[{'PASS' if x['ok'] else x['severity']}] {x['name']}: {x['detail']}")
print(f"failures={len(fails)}, warnings={len(warns)}")
raise SystemExit(1 if fails else 0)
PY
}
run_step "02_static_code_narrative" static_code_audit

# ---------------------------------------------------------------------------
# 03 Embedded custom self-tests
# ---------------------------------------------------------------------------
custom_self_tests() {
  python extract_predobj_cropaug.py --self_test
  python train_relproto_alignemt.py --self_test
}
run_step "03_custom_self_tests" custom_self_tests

# ---------------------------------------------------------------------------
# 04 Saved raw text bank vs freshly encoded official text path
# ---------------------------------------------------------------------------
live_text_bank_audit() {
  [[ "$RUN_LIVE_TEXT_AUDIT" == "1" ]] || return 3
  CUDA_VISIBLE_DEVICES="$GPU" python - "$TEXT_BANK" "$OUT/artifacts/live_text_bank_audit.json" <<'PY'
import json, sys, traceback
from pathlib import Path
try:
    import torch
    import torch.nn.functional as F
    import extract_clip_text_bank as ext
    bp=Path(sys.argv[1]); op=Path(sys.argv[2])
    d=torch.load(bp,map_location="cpu")
    saved=d.get("features",d.get("raw_features"))
    names=d.get("class_names",d.get("classnames",d.get("names")))
    if saved is None or names is None: raise KeyError("bank features/names missing")
    saved=torch.as_tensor(saved,dtype=torch.float32)
    names=[str(x) for x in names]
    if tuple(saved.shape)!=(116,512) or len(names)!=116:
        raise ValueError((saved.shape,len(names)))
    device="cuda" if torch.cuda.is_available() else "cpu"
    fresh,templates=ext.encode_talk2dino_text_bank(
        class_names=names,clip_model_name="ViT-B/16",
        template_set="sub_imagenet_template",device=device,chunk_size=32)
    fresh=fresh.float()
    cos=(F.normalize(saved,dim=-1)*F.normalize(fresh,dim=-1)).sum(-1)
    diff=(saved-fresh).abs()
    r={"status":"PASS","shape":list(saved.shape),"templates":len(templates),
       "metadata_normalized":d.get("normalized"),"metadata_stage":d.get("stage"),
       "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
       "min_row_cosine":float(cos.min()),"mean_row_cosine":float(cos.mean())}
    if d.get("normalized",False) is not False or r["min_row_cosine"]<0.9995:
        r["status"]="FAIL"
    op.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r,indent=2))
    raise SystemExit(0 if r["status"]=="PASS" else 1)
except SystemExit: raise
except Exception as e:
    op=Path(sys.argv[2])
    r={"status":"WARN/SKIP","error":f"{type(e).__name__}: {e}",
       "traceback":traceback.format_exc()}
    op.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r,indent=2))
    raise SystemExit(3)
PY
}
run_step "04_live_text_bank_vs_eval" live_text_bank_audit

# ---------------------------------------------------------------------------
# 05 Exact object-macro Spearman PLUS value geometry
#     (Spearman alone does not preserve relative-margin scale.)
# ---------------------------------------------------------------------------
structure_geometry_audit() {
  CUDA_VISIBLE_DEVICES="$GPU" python - \
    "$TEXT_BANK" "$MODEL_CONFIG" "$INITIAL_WEIGHT" "$PARTSTRUCT_WEIGHT" "$FT0_WEIGHT" \
    "$OUT/artifacts/structure_geometry.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import train_relproto_alignemt as tr
from src.loss import PartStructureRankLoss

bp,cp,ip,pp,fp,op=map(Path,sys.argv[1:])
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
crit=PartStructureRankLoss.from_file(bp,rank_temperature=0.05,min_parts=3).to(device)
data=torch.load(bp,map_location="cpu")
raw=torch.as_tensor(data.get("features",data.get("raw_features")),dtype=torch.float32)
groups=data["object_groups"]

def pairs(feat,ids):
    ids=torch.as_tensor(ids,dtype=torch.long)
    z=F.normalize(feat.index_select(0,ids).float(),dim=-1)
    s=z@z.T; tri=torch.triu_indices(len(ids),len(ids),offset=1)
    return s[tri[0],tri[1]].detach().cpu().numpy()

def one(weight):
    proj,info=tr.load_frozen_projector(project_root=Path(".").resolve(),
        config_path=cp.resolve(),weight_path=weight.resolve(),device=device)
    with torch.inference_mode():
        z=tr.project_raw_clip_bank(raw,proj,device=device,batch_size=128)
    exact=crit.exact_audit(proj)
    rows={}; maes=[]; rmses=[]; ratios=[]; slopes=[]; pears=[]
    for obj,ids in groups.items():
        if len(ids)<3: continue
        a,b=pairs(raw,ids),pairs(z.cpu(),ids)
        va=float(np.var(a)); sa=float(np.std(a)); sb=float(np.std(b))
        row={
            "parts":len(ids),
            "spearman":float(spearmanr(a,b).correlation),
            "cosine_mae":float(np.mean(np.abs(a-b))),
            "cosine_rmse":float(np.sqrt(np.mean((a-b)**2))),
            "raw_std":sa,"projected_std":sb,
            "std_ratio":float(sb/sa) if sa>0 else None,
            "linear_slope_b_vs_a":float(np.cov(a,b,ddof=0)[0,1]/va) if va>0 else None,
            "pearson":float(np.corrcoef(a,b)[0,1]),
        }
        rows[str(obj)]=row
        maes.append(row["cosine_mae"]); rmses.append(row["cosine_rmse"])
        ratios.append(row["std_ratio"]); slopes.append(row["linear_slope_b_vs_a"])
        pears.append(row["pearson"])
    return {"weight":str(weight),"sha256":info["weights_sha256"],
      "exact_macro_spearman":float(exact["macro"]),
      "exact_per_object":{k:v for k,v in exact.items() if k!="macro"},
      "object_macro_cosine_mae":float(np.mean(maes)),
      "object_macro_cosine_rmse":float(np.mean(rmses)),
      "object_macro_std_ratio":float(np.mean(ratios)),
      "object_macro_linear_slope":float(np.mean(slopes)),
      "object_macro_pearson":float(np.mean(pears)),"per_object":rows}

models={"initial":ip,"partstruct":pp}
if fp.is_file(): models["ft0"]=fp
rep={"protocol":"raw prompt mean -> projector -> L2 normalize","models":{}}
for label,p in models.items():
    if not p.is_file():
        rep["models"][label]={"status":"MISSING","weight":str(p)}; continue
    print("[structure]",label,p)
    rep["models"][label]=one(p)
    x=rep["models"][label]
    print(json.dumps({"label":label,"spearman":x["exact_macro_spearman"],
      "cosMAE":x["object_macro_cosine_mae"],"stdRatio":x["object_macro_std_ratio"],
      "slope":x["object_macro_linear_slope"]},indent=2))
op.write_text(json.dumps(rep,indent=2,ensure_ascii=False)+"\n")
PY
}
run_step "05_structure_geometry_before_ft0" structure_geometry_audit

# ---------------------------------------------------------------------------
# 06 Loss-scale audit
# ---------------------------------------------------------------------------
loss_scale_audit() {
python - "$OUT/artifacts/loss_scale_audit.json" <<'PY'
import json,re,sys
from pathlib import Path
op=Path(sys.argv[1])
cand=[Path("output/coco2014_clean_ft10_partstruct_rawpath_w1e4_lr1e5/train.log"),
      Path("output/projector_baselines/rawpath_partstruct_train.log")]
lp=next((p for p in cand if p.is_file()),None)
r={"structure_weight":1e-4,
   "contrastive_extra_batch_squared_division_present":
     "/ scores.shape[0]**2" in Path("src/loss.py").read_text(),
   "log":str(lp) if lp else None}
if lp:
    s=lp.read_text(errors="replace")
    m=re.findall(r"train_object=([0-9.eE+-]+).*?train_struct=([0-9.eE+-]+).*?"
                 r"val_object=([0-9.eE+-]+).*?val_struct=([0-9.eE+-]+)",s)
    if m:
        a,b,c,d=map(float,m[-1]); r.update(
          last_train_object=a,last_train_structure=b,last_train_weighted_structure=b*1e-4,
          weighted_structure_over_object=(b*1e-4/a if a else None),
          last_val_object=c,last_val_structure=d,last_val_weighted_structure=d*1e-4,
          weighted_val_structure_over_object=(d*1e-4/c if c else None))
    else: r["warning"]="could not parse epoch loss"
else: r["warning"]="known PartStruct log not found"
op.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r,indent=2))
PY
}
run_step "06_loss_scale" loss_scale_audit

# ---------------------------------------------------------------------------
# 07 PredObj cache provenance / split / tensor contract
# ---------------------------------------------------------------------------
cache_metadata_audit() {
  [[ -f "$FIXED_CACHE" ]] || return 3
  python - "$FIXED_CACHE" "$PARTSTRUCT_WEIGHT" "$OUT/artifacts/cache_metadata.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
cp=Path(sys.argv[1]).resolve(); wp=Path(sys.argv[2]).resolve(); op=Path(sys.argv[3])
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
try: d=torch.load(cp,map_location="cpu",weights_only=False,mmap=True)
except TypeError: d=torch.load(cp,map_location="cpu")
meta=dict(d.get("pred_obj_cropaug_meta",{})); anns=d.get("annotations",[])
ids={str(x.get("image_id","")) for x in anns}; ids.discard("")
r={"cache":str(cp),"annotations":len(anns),"unique_image_ids":len(ids),"meta":meta,
   "expected_projector":str(wp)}
if wp.is_file():
    r["expected_projector_sha256"]=sha(wp)
    r["cache_projector_sha256"]=meta.get("projector_weight_sha256")
    r["projector_sha_match"]=r["cache_projector_sha256"]==r["expected_projector_sha256"]

def stems(p):
    return {Path(x.strip()).stem for x in p.read_text().splitlines() if x.strip()}
trainc=[Path("data/PascalPart116/train.txt"),Path("/mnt/sda/master/dataset/lyx/PascalPart116/train.txt")]
valc=[Path("data/PascalPart116/val.txt"),Path("/mnt/sda/master/dataset/lyx/PascalPart116/val.txt")]
tp=next((p for p in trainc if p.is_file()),None); vp=next((p for p in valc if p.is_file()),None)
if tp:
    trn=stems(tp); r["train_split"]=str(tp); r["cache_ids_not_in_train_count"]=len(ids-trn)
    r["cache_ids_not_in_train_first100"]=sorted(ids-trn)[:100]
if vp:
    val=stems(vp); r["val_split"]=str(vp); r["cache_val_overlap_count"]=len(ids&val)
    r["cache_val_overlap_first100"]=sorted(ids&val)[:100]

bad=[]; pcs=[]; fgs=[]
for i,a in enumerate(anns):
    try:
        pids=[int(x) for x in a["part_category_id"]]; names=[str(x) for x in a["part_class_name"]]
        if not pids or len(pids)!=len(names) or len(set(pids))!=len(pids): raise ValueError("presence")
        if any(x<0 or x>=116 for x in pids): raise ValueError("part id")
        tok=a["cropaug_patch_tokens"]; fg=a["pred_obj_mask_patch"]; box=a["cropaug_box_xyxy"]
        if tuple(tok.shape)!=(1024,768): raise ValueError(f"token shape {tuple(tok.shape)}")
        if tuple(fg.shape)!=(1024,): raise ValueError(f"fg shape {tuple(fg.shape)}")
        if tuple(box.shape)!=(4,): raise ValueError(f"box shape {tuple(box.shape)}")
        pcs.append(len(pids)); fgs.append(int(fg.sum().item()))
    except Exception as e:
        bad.append({"index":i,"error":f"{type(e).__name__}: {e}"})
        if len(bad)>=20: break
r["contract_bad_first20"]=bad
r["presence_mean"]=float(np.mean(pcs)) if pcs else None
r["foreground_patch_mean"]=float(np.mean(fgs)) if fgs else None
fail=[]; warn=[]
if meta.get("pamr") is not True: fail.append("cache PAMR != True")
try:
    if abs(float(meta.get("bg_thresh"))-0.54)>1e-12: warn.append(f"bg_thresh={meta.get('bg_thresh')}")
except Exception: fail.append("cache bg_thresh missing")
if not meta.get("projector_weight_sha256"): fail.append("projector SHA missing")
if r.get("projector_sha_match") is False: fail.append("projector SHA mismatch")
if r.get("cache_val_overlap_count",0): fail.append("validation leakage")
if r.get("cache_ids_not_in_train_count",0): fail.append("cache IDs outside train split")
if bad: fail.append("annotation contract error")
r["warnings"]=warn; r["failures"]=fail
op.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(r,indent=2,ensure_ascii=False))
raise SystemExit(1 if fail else 0)
PY
}
run_step "07_predobj_cache_metadata_split" cache_metadata_audit

# ---------------------------------------------------------------------------
# 08 Trainer-native full preflight
# ---------------------------------------------------------------------------
cache_trainer_preflight() {
  [[ -f "$FIXED_CACHE" ]] || return 3
  local od="$OUT/preflight/partstruct_matching_cache"
  mkdir -p "$od"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train_relproto_alignemt.py \
    --project_root . --train_dataset "$FIXED_CACHE" --text_bank "$TEXT_BANK" \
    --model_config "$MODEL_CONFIG" --weights "$PARTSTRUCT_WEIGHT" \
    --feature_name cropaug_patch_tokens --foreground_key pred_obj_mask_patch \
    --part_id_key part_category_id --part_name_key part_class_name \
    --clip_model ViT-B/16 --template_set sub_imagenet_template \
    --prototype_max_patches 4 --expected_bg_thresh "$EXPECTED_CACHE_BG_THRESH" \
    --device cuda --preflight_only --out_dir "$od"
}
run_step "08_relproto_full_cache_preflight" cache_trainer_preflight

# ---------------------------------------------------------------------------
# 09 W invariant helper + existing W audit
# ---------------------------------------------------------------------------
audit_w_checkpoint() {
  local wpath="$1" label="$2" jsonout="$3"
  CUDA_VISIBLE_DEVICES="$GPU" python - "$wpath" "$label" "$TEXT_BANK" "$MODEL_CONFIG" "$jsonout" <<'PY'
import json,sys
from pathlib import Path
import torch,torch.nn.functional as F
import train_relproto_alignemt as tr
wp=Path(sys.argv[1]); label=sys.argv[2]; bp=Path(sys.argv[3]); cp=Path(sys.argv[4]); op=Path(sys.argv[5])
if not wp.is_file(): raise SystemExit(3)
try: ck=torch.load(wp,map_location="cpu",weights_only=False)
except TypeError: ck=torch.load(wp,map_location="cpu")
W=torch.as_tensor(ck["W"],dtype=torch.float64)
I=torch.eye(768,dtype=torch.float64)
orth=float((W.T@W-I).abs().max())
raw,_,_=tr.load_raw_clip_bank(bp.resolve(),expected_clip_model="ViT-B/16",
                              expected_template="sub_imagenet_template")
pinfo=ck.get("projector",{}); pp=Path(pinfo.get("weights",""))
if not pp.is_absolute(): pp=(Path(".").resolve()/pp).resolve()
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
proj,_=tr.load_frozen_projector(project_root=Path(".").resolve(),
    config_path=cp.resolve(),weight_path=pp,device=dev)
T0=tr.project_raw_clip_bank(raw,proj,device=dev,batch_size=128).double()
T1=F.normalize(T0@W,dim=-1)
pres=float(((T0@T0.T)-(T1@T1.T)).abs().max())
r={"label":label,"checkpoint":str(wp),"epoch":ck.get("epoch"),
   "global_step":ck.get("global_step"),"orthogonality_max_abs":orth,
   "pairwise_cosine_preservation_max_abs":pres,"source_projector":str(pp),
   "source_projector_sha256_recorded":pinfo.get("weights_sha256"),
   "method":ck.get("method",{}),"dataset":ck.get("dataset",{}),
   "last_history":(ck.get("history") or [None])[-1]}
op.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(r,indent=2,ensure_ascii=False))
raise SystemExit(1 if orth>1e-3 or pres>2e-4 else 0)
PY
}
existing_w_audit() {
  audit_w_checkpoint "$PARTSTRUCT_EXISTING_W" "partstruct_existing_completeflow" \
    "$OUT/artifacts/w_existing_partstruct.json"
}
run_step "09_existing_W_orthogonality_geometry" existing_w_audit

# ---------------------------------------------------------------------------
# 10 Existing bake re-certification to audit-only file
# ---------------------------------------------------------------------------
existing_bake_recertify() {
  [[ -f "$PARTSTRUCT_EXISTING_W" ]] || return 3
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python bake_predobj_relproto_w_into_projector_final.py \
    --project_root . --projector "$PARTSTRUCT_WEIGHT" \
    --w_checkpoint "$PARTSTRUCT_EXISTING_W" --text_bank "$TEXT_BANK" \
    --model_config "$MODEL_CONFIG" \
    --output "$OUT/artifacts/rebake_existing_partstruct.pth" --device cuda
}
run_step "10_existing_bake_equivalence" existing_bake_recertify

# ---------------------------------------------------------------------------
# 11 Full-117 GT exact conversion + ACTUAL MMSeg pre_eval proof
#     READ-ONLY on source/derived masks.
# ---------------------------------------------------------------------------
full117_audit() {
python - "$OUT/artifacts/full117_gt_metric_audit.json" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
op=Path(sys.argv[1])
srcs=[Path("data/PascalPart116/annotations_detectron2_part/val"),
      Path("/mnt/sda/master/dataset/lyx/PascalPart116/annotations_detectron2_part/val")]
dsts=[Path("data/PascalPart116_part117_eval/annotations/val")]
splits=[Path("data/PascalPart116/val.txt"),Path("/mnt/sda/master/dataset/lyx/PascalPart116/val.txt")]
imgs=[Path("data/PascalPart116/images/val"),Path("/mnt/sda/master/dataset/lyx/PascalPart116/images/val")]
src=next((p.resolve() for p in srcs if p.is_dir()),None)
dst=next((p.resolve() for p in dsts if p.is_dir()),None)
split=next((p.resolve() for p in splits if p.is_file()),None)
imgdir=next((p.resolve() for p in imgs if p.is_dir()),None)
if not all([src,dst,split,imgdir]):
    print(json.dumps({"status":"WARN/SKIP","src":str(src),"dst":str(dst),
                      "split":str(split),"imgdir":str(imgdir)},indent=2))
    raise SystemExit(3)
stems=[Path(x.strip()).stem for x in split.read_text().splitlines() if x.strip()]
if len(stems)!=len(set(stems)): raise RuntimeError("duplicate split")
sh=np.zeros(256,np.int64); dh=np.zeros(256,np.int64); bad=[]
for stem in stems:
    sp=src/f"{stem}.png"; dp=dst/f"{stem}.png"
    if not sp.is_file() or not dp.is_file(): bad.append((stem,"missing")); continue
    a=np.asarray(Image.open(sp)); b=np.asarray(Image.open(dp))
    if a.ndim!=2 or b.shape!=a.shape: bad.append((stem,"shape")); continue
    if not set(np.unique(a).tolist()) <= (set(range(116))|{255}):
        bad.append((stem,"source ids")); continue
    e=np.zeros(a.shape,np.uint8); fg=a!=255; e[fg]=a[fg].astype(np.uint8)+1
    if not np.array_equal(e,b): bad.append((stem,int(np.count_nonzero(e!=b)))); continue
    sh+=np.bincount(a.reshape(-1),minlength=256)
    dh+=np.bincount(b.reshape(-1),minlength=256)
if bad:
    print("mismatch first20",bad[:20]); raise SystemExit(1)
if dh[255]!=0 or sh[255]!=dh[0]: raise RuntimeError("background/ignore invariant")
for i in range(116):
    if sh[i]!=dh[i+1]: raise RuntimeError(f"shift mismatch {i}->{i+1}")

# Actual MMSeg evaluator proof.
import sys as _sys, torch, mmcv
_sys.path.insert(0,str(Path("src/open_vocabulary_segmentation").resolve()))
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import PIPELINES
if PIPELINES.get("FloatImage") is None:
    @PIPELINES.register_module()
    class FloatImage:
        def __call__(self,res):
            res["img"]=res["img"].astype(np.float32); return res
import segmentation.datasets.pascalpart116_part117  # noqa

cfgp=Path("src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/pascalpart116_part117.py")
if not cfgp.is_file():
    cfgp=Path("src/open_vocabulary_segmentation/segmentation/configs/base/datasets/pascalpart116_part117.py")
cfg=mmcv.Config.fromfile(str(cfgp))
cfg.data.test.img_dir=str(imgdir); cfg.data.test.ann_dir=str(dst); cfg.data.test.split=str(split)
dataset=build_dataset(cfg.data.test)
if len(dataset.CLASSES)!=117 or dataset.CLASSES[0]!="background": raise RuntimeError("classes")
if dataset.ignore_index!=255 or dataset.reduce_zero_label is not False: raise RuntimeError("dataset flags")
perfect=[]; broken=[]
for i in range(len(dataset)):
    gt=dataset.get_gt_seg_map_by_idx(i)
    stem=Path(dataset.img_infos[i]["filename"]).stem
    disk=np.asarray(Image.open(dst/f"{stem}.png"))
    if not np.array_equal(gt,disk): raise RuntimeError(f"MMSeg altered GT {stem}")
    if np.any(gt==255): raise RuntimeError(f"255 remains {stem}")
    perfect.extend(dataset.pre_eval([gt.copy()],[i]))
    pred=gt.copy(); pred[pred==0]=1
    broken.extend(dataset.pre_eval([pred],[i]))
def agg(xs):
    inter=sum(x[0] for x in xs); union=sum(x[1] for x in xs)
    iou=inter.double()/union.double()
    return iou,float(torch.nanmean(iou))
pi,pmiou=agg(perfect); bi,bmiou=agg(broken)
r={"status":"PASS","source":str(src),"derived117":str(dst),"images":len(stems),
   "background0_pixels":int(dh[0]),"semantic_pixels":int(dh[1:117].sum()),
   "ignore255_pixels":int(dh[255]),"perfect_mIoU":pmiou,
   "perfect_background_IoU":float(pi[0]),"destroy_background_mIoU":bmiou,
   "destroy_background_background_IoU":float(bi[0])}
op.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r,indent=2))
if abs(pmiou-1)>1e-8 or abs(float(pi[0])-1)>1e-8 or abs(float(bi[0]))>1e-12 or bmiou>=pmiou:
    raise SystemExit(1)
PY
}
run_step "11_full117_gt_and_metric" full117_audit

# ---------------------------------------------------------------------------
# 12 Historical local eval result inventory
# ---------------------------------------------------------------------------
historical_eval_inventory() {
python - "$OUT/artifacts/historical_eval_inventory.tsv" <<'PY'
import re,sys
from pathlib import Path
op=Path(sys.argv[1]); rows=[]
for p in sorted(Path("output").rglob("*.log")):
    try: s=p.read_text(errors="replace")
    except Exception: continue
    vals=re.findall(r"INFO\s+>>\s*([0-9.]+)\s*,\s*([0-9.]+)",s)
    if not vals: continue
    bg=re.findall(r"Building DINOTextSegInference with (\d+) classes.*?with_bg=(True|False).*?bg_thresh=([0-9.]+)",s)
    n,w,t=(bg[-1] if bg else ("","","")); a,b=vals[-1]
    rows.append((str(p),n,w,t,a,b))
op.write_text("log\tclasses\twith_bg\tbg_thresh\tmetric1\tmetric2\n"+
              "\n".join("\t".join(r) for r in rows)+"\n")
print(op.read_text())
PY
}
run_step "12_historical_eval_inventory" historical_eval_inventory

# ===========================================================================
# CAUSAL BLOCK
# ===========================================================================
if [[ "$RUN_CAUSAL_ABLATION" == "1" ]]; then
  echo "======================================================================"
  echo "CAUSAL ABLATION: initial vs FT0 vs PartStruct; SAME fixed cache for W"
  echo "======================================================================"

  train_ft0() {
    if [[ -f "$FT0_WEIGHT" ]]; then echo "[reuse] $FT0_WEIGHT"; return 0; fi
    [[ -f "$COCO_TRAIN" && -f "$COCO_VAL" ]] || return 3
    gpu_ready || return 3
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python train_partstruct_ft.py \
      --model_config "$MODEL_CONFIG" --train_dataset "$COCO_TRAIN" --val_dataset "$COCO_VAL" \
      --feature_name disentangled_self_attn --text_features ann_feats \
      --optimizer Adam --scheduler linear --warmup 0 \
      --init_weight "$INITIAL_WEIGHT" --num_epochs 10 --lr 1e-5 \
      --structure_weight 0 --name_pedix "$FT0_SUFFIX"
  }
  run_step "13_train_FT0_causal_control" train_ft0
  run_step "14_structure_geometry_with_FT0" structure_geometry_audit

  PORT_COUNTER=0
  eval_one() {
    local label="$1" weight="$2" protocol="$3" basecfg
    [[ -f "$weight" ]] || return 3
    gpu_ready || return 3
    [[ "$(dirname "$weight")" == "weights" ]] || { echo "weight must be under weights/: $weight"; return 1; }
    [[ "$protocol" == "part116_ignorebg" ]] && basecfg="$EVAL116" || basecfg="$EVAL117"
    local bn="$(basename "$weight" .pth)" dir="$OUT/eval/${label}_${protocol}"
    local log="$OUT/logs/eval_${label}_${protocol}.log"
    mkdir -p "$dir"
    local port=$((MASTER_PORT_BASE+PORT_COUNTER)); PORT_COUNTER=$((PORT_COUNTER+1))
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$port" \
      src/open_vocabulary_segmentation/main.py --eval --output "$dir" \
      --eval_cfg "$EVAL_CFG" --eval_base_cfg "$basecfg" \
      --opts "model.proj_name=${bn}" evaluate.pamr=false "evaluate.bg_thresh=${EVAL_BG_THRESH}" \
      2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || return $rc
    local v="$(grep -E 'INFO  >>' "$log" | tail -n1 | sed -E 's/.*INFO  >>[[:space:]]*([0-9.]+).*/\1/' || true)"
    [[ -n "$v" ]] || return 1
    printf '%s\t%s\t%s\t%s\n' "$label" "$protocol" "$v" "$log" >> "$EVAL_TSV"
    echo "[EVAL_RESULT] $label $protocol mIoU=$v"
  }
  eval_pair() {
    eval_one "$1" "$2" part116_ignorebg || return $?
    eval_one "$1" "$2" part117_bg040 || return $?
  }

  run_step "15_eval_initial_direct" eval_pair initial_direct "$INITIAL_WEIGHT"
  run_step "16_eval_FT0_direct" eval_pair ft0_direct "$FT0_WEIGHT"
  run_step "17_eval_partstruct_direct" eval_pair partstruct_direct "$PARTSTRUCT_WEIGHT"

  train_fixed_w() {
    local label="$1" weight="$2" strict="$3" od="$OUT/w_runs/$1"
    [[ -f "$FIXED_CACHE" && -f "$weight" ]] || return 3
    gpu_ready || return 3
    [[ -f "$od/W_last.pt" ]] && { echo "[reuse] $od/W_last.pt"; return 0; }
    mkdir -p "$od"
    local extra=()
    [[ "$strict" == "1" ]] || extra=(--no-require_cache_projector_match)
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python train_relproto_alignemt.py \
      --project_root . --train_dataset "$FIXED_CACHE" --text_bank "$TEXT_BANK" \
      --model_config "$MODEL_CONFIG" --weights "$weight" \
      --feature_name cropaug_patch_tokens --foreground_key pred_obj_mask_patch \
      --part_id_key part_category_id --part_name_key part_class_name \
      --clip_model ViT-B/16 --template_set sub_imagenet_template \
      --prototype_max_patches 4 --epochs 10 --batch_size 16 --lr 1e-3 --seed 123 \
      --device cuda --expected_bg_thresh "$EXPECTED_CACHE_BG_THRESH" \
      "${extra[@]}" --out_dir "$od"
  }
  # Initial / FT0 mismatch is INTENTIONAL: same visual cache is held fixed.
  run_step "18_fixedcache_W_initial" train_fixed_w initial "$INITIAL_WEIGHT" 0
  run_step "19_fixedcache_W_FT0" train_fixed_w ft0 "$FT0_WEIGHT" 0
  run_step "20_fixedcache_W_partstruct" train_fixed_w partstruct "$PARTSTRUCT_WEIGHT" 1

  all_fixed_w_audit() {
    local any=0
    for label in initial ft0 partstruct; do
      local p="$OUT/w_runs/$label/W_last.pt"
      if [[ -f "$p" ]]; then
        any=1; audit_w_checkpoint "$p" "$label" "$OUT/artifacts/w_fixed_${label}.json" || return $?
      fi
    done
    [[ $any -eq 1 ]] || return 3
  }
  run_step "21_fixedcache_W_invariants" all_fixed_w_audit

  bake_eval() {
    local label="$1" source="$2" w="$OUT/w_runs/$1/W_last.pt"
    [[ -f "$w" ]] || return 3
    local bn="vitb_mlp_infonce_nightly_${TS}_${label}_fixedcache_w_baked"
    local bw="weights/${bn}.pth"
    if [[ ! -f "$bw" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      python bake_predobj_relproto_w_into_projector_final.py \
        --project_root . --projector "$source" --w_checkpoint "$w" \
        --text_bank "$TEXT_BANK" --model_config "$MODEL_CONFIG" \
        --output "$bw" --device cuda || return $?
    fi
    eval_pair "${label}_fixedcache_W" "$bw"
  }
  run_step "22_bake_eval_fixedW_initial" bake_eval initial "$INITIAL_WEIGHT"
  run_step "23_bake_eval_fixedW_FT0" bake_eval ft0 "$FT0_WEIGHT"
  run_step "24_bake_eval_fixedW_partstruct" bake_eval partstruct "$PARTSTRUCT_WEIGHT"
else
  record_status WARN "13-24_causal_ablation" "disabled by RUN_CAUSAL_ABLATION=0"
fi

# ---------------------------------------------------------------------------
# 25 Post-hoc GT-only anchor / support purity at W=I and trained W.
#     GT is NEVER used by W training; this is diagnosis only.
# ---------------------------------------------------------------------------
anchor_purity_audit() {
  [[ "$RUN_ANCHOR_AUDIT" == "1" ]] || return 3
  [[ -f "$FIXED_CACHE" ]] || return 3
  local gt=""
  for p in "data/PascalPart116/annotations_detectron2_part/train" \
           "/mnt/sda/master/dataset/lyx/PascalPart116/annotations_detectron2_part/train"; do
    [[ -d "$p" ]] && { gt="$p"; break; }
  done
  [[ -n "$gt" ]] || return 3

  local specs=()
  [[ -f "$INITIAL_WEIGHT" ]] && specs+=("initial::$INITIAL_WEIGHT::$OUT/w_runs/initial/W_last.pt")
  [[ -f "$FT0_WEIGHT" ]] && specs+=("ft0::$FT0_WEIGHT::$OUT/w_runs/ft0/W_last.pt")
  [[ -f "$PARTSTRUCT_WEIGHT" ]] && specs+=("partstruct::$PARTSTRUCT_WEIGHT::$OUT/w_runs/partstruct/W_last.pt")

  CUDA_VISIBLE_DEVICES="$GPU" python - "$FIXED_CACHE" "$TEXT_BANK" "$MODEL_CONFIG" \
    "$gt" "$ANCHOR_AUDIT_MAX" "$OUT/artifacts/anchor_relproto_purity.json" "${specs[@]}" <<'PY'
import json,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image
import torch.nn.functional as F
import train_relproto_alignemt as tr

cachep=Path(sys.argv[1]); bankp=Path(sys.argv[2]); cfgp=Path(sys.argv[3])
gtdir=Path(sys.argv[4]); maxn=int(sys.argv[5]); op=Path(sys.argv[6]); specs=sys.argv[7:]
if not specs: raise SystemExit(3)
try: d=torch.load(cachep,map_location="cpu",weights_only=False,mmap=True)
except TypeError: d=torch.load(cachep,map_location="cpu")
anns=d["annotations"]
idx=list(range(len(anns)))
if maxn>0 and len(idx)>maxn:
    idx=sorted(random.Random(123).sample(idx,maxn))
samples=[]; missing=0
for ii in idx:
    a=anns[ii]; stem=str(a.get("image_id","")); mp=gtdir/f"{stem}.png"
    if not mp.is_file(): missing+=1; continue
    gt=np.asarray(Image.open(mp))
    box=np.asarray(a["cropaug_box_xyxy"]).reshape(-1).astype(int).tolist()
    x1,y1,x2,y2=box; crop=gt[y1:y2,x1:x2]
    h,w=crop.shape
    if h<=0 or w<=0: continue
    yy=np.arange(h,dtype=np.int64)[:,None]; xx=np.arange(w,dtype=np.int64)[None,:]
    cell=((yy*32)//h)*32 + ((xx*32)//w)
    cell=np.broadcast_to(cell,(h,w)).reshape(-1)
    total=np.bincount(cell,minlength=1024).astype(np.float64)
    flat=crop.reshape(-1)
    pids=[int(x) for x in a["part_category_id"]]
    anym=np.zeros((len(pids),1024),bool); purity=np.zeros((len(pids),1024),np.float32)
    for k,pid in enumerate(pids):
        cnt=np.bincount(cell[flat==pid],minlength=1024).astype(np.float64)
        anym[k]=cnt>0
        purity[k]=np.divide(cnt,total,out=np.zeros_like(cnt),where=total>0).astype(np.float32)
    samples.append((ii,pids,anym,purity,str(a.get("class_name",""))))
groups=defaultdict(list)
for pos,(_,pids,_,_,_) in enumerate(samples): groups[len(pids)].append(pos)
print(f"annotations={len(anns)} requested={len(idx)} usable={len(samples)} missing_gt={missing}")

raw,_,_=tr.load_raw_clip_bank(bankp.resolve(),expected_clip_model="ViT-B/16",
                              expected_template="sub_imagenet_template")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state(spec):
    label,ws,wtrain=spec.split("::",2); wp=Path(ws); Wp=Path(wtrain)
    proj,_=tr.load_frozen_projector(project_root=Path(".").resolve(),
        config_path=cfgp.resolve(),weight_path=wp.resolve(),device=dev)
    T0=tr.project_raw_clip_bank(raw,proj,device=dev,batch_size=128).to(dev)
    arr=[("I",torch.eye(768,device=dev))]
    if Wp.is_file():
        try: ck=torch.load(Wp,map_location="cpu",weights_only=False)
        except TypeError: ck=torch.load(Wp,map_location="cpu")
        arr.append(("W",torch.as_tensor(ck["W"],dtype=torch.float32,device=dev)))
    return label,T0,arr

def evaluate(T0,W):
    n=ah=am=sn=sh=posr=kr=0; ap=sp=rels=pc=0.0
    per=defaultdict(lambda:[0,0])
    for k,members in sorted(groups.items()):
      for st in range(0,len(members),16):
        ch=members[st:st+16]; ai=[samples[p][0] for p in ch]
        pids=torch.tensor([samples[p][1] for p in ch],dtype=torch.long,device=dev)
        tok=torch.stack([anns[i]["cropaug_patch_tokens"].float() for i in ai]).to(dev)
        tok=F.normalize(tok,dim=-1)
        fg=torch.stack([anns[i]["pred_obj_mask_patch"].bool() for i in ai]).to(dev)
        cur=F.normalize(T0[pids]@W,dim=-1)
        sel=tr.build_relproto(cur.detach(),tok,fg,max_patches=4)
        anc=sel["anchor_indices"].cpu().numpy()
        ar=sel["anchor_relative_scores"].cpu().numpy()
        sup=sel["support_indices"].cpu().numpy()
        cnt=sel["prototype_counts"].cpu().numpy()
        for br,p in enumerate(ch):
          _,ids,anym,purity,obj=samples[p]
          for kk,_pid in enumerate(ids):
            a=int(anc[br,kk]); pur=float(purity[kk,a]); hit=bool(anym[kk,a])
            n+=1; ah+=int(hit); am+=int(pur>0.5); ap+=pur
            per[obj][0]+=int(hit); per[obj][1]+=1
            if len(ids)>1: kr+=1; posr+=int(float(ar[br,kk])>0)
            rels+=float(ar[br,kk]); pc+=int(cnt[br,kk])
            for si in sup[br,kk]:
              si=int(si)
              if si<0: continue
              sn+=1; sh+=int(bool(anym[kk,si])); sp+=float(purity[kk,si])
        del tok,fg,cur,sel
    return {"part_queries":n,
      "anchor_any_gt_hit_rate":ah/max(n,1),
      "anchor_gt_majority_rate":am/max(n,1),
      "anchor_mean_gt_pixel_purity":ap/max(n,1),
      "support_patch_any_gt_hit_rate":sh/max(sn,1),
      "support_patch_mean_gt_pixel_purity":sp/max(sn,1),
      "mean_prototype_patch_count":pc/max(n,1),
      "positive_anchor_relative_fraction_Kgt1":posr/max(kr,1),
      "mean_anchor_relative_score":rels/max(n,1),
      "per_object_anchor_any_hit":{o:a/b for o,(a,b) in sorted(per.items()) if b}}

rep={"cache":str(cachep),"gt_dir":str(gtdir),"requested":len(idx),"usable":len(samples),
     "missing_gt":missing,
     "note":"GT is used ONLY post-hoc to audit semantic anchor/support purity; never for W training.",
     "states":{}}
for spec in specs:
    label,T0,states=state(spec)
    for sn,W in states:
        key=f"{label}_{sn}"; print("[anchor]",key)
        rep["states"][key]=evaluate(T0,W)
        print(json.dumps({key:rep["states"][key]},indent=2))
op.write_text(json.dumps(rep,indent=2,ensure_ascii=False)+"\n")
PY
}
run_step "25_GT_only_anchor_relproto_purity" anchor_purity_audit

# ---------------------------------------------------------------------------
# 26 Final summary + compact paste-back file
# ---------------------------------------------------------------------------
final_report() {
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
def loadj(n):
    p=root/"artifacts"/n
    if not p.is_file(): return None
    try: return json.loads(p.read_text())
    except Exception: return None
status=[]
for line in (root/"status.tsv").read_text().splitlines():
    if line.strip():
        p=line.split("\t",2); status.append(p+[""]*(3-len(p)))
evals=[]
if (root/"eval_results.tsv").is_file():
  for line in (root/"eval_results.tsv").read_text().splitlines()[1:]:
    p=line.split("\t")
    if len(p)>=4:
      try: evals.append({"label":p[0],"protocol":p[1],"miou":float(p[2]),"log":p[3]})
      except ValueError: pass
struct=loadj("structure_geometry.json"); cache=loadj("cache_metadata.json")
full=loadj("full117_gt_metric_audit.json"); anchor=loadj("anchor_relproto_purity.json")
scale=loadj("loss_scale_audit.json")

L=["# Talk2DINO_WPS Nightly Audit Summary","",
   "## Step status",""]
for st,n,note in status: L.append(f"- **{st}** `{n}` — {note}")
L+=["","## Controlled evaluator results",""]
if evals:
    L+=["| label | protocol | mIoU |","|---|---|---:|"]
    for x in evals: L.append(f"| {x['label']} | {x['protocol']} | {x['miou']:.2f} |")
else: L.append("No new controlled eval results.")
L+=["","## Text structure + cosine-value geometry",""]
if struct:
    L+=["| model | macro Spearman | cosine MAE | std ratio | linear slope |",
        "|---|---:|---:|---:|---:|"]
    for label,x in struct.get("models",{}).items():
      if "exact_macro_spearman" in x:
        L.append(f"| {label} | {x['exact_macro_spearman']:.6f} | "
                 f"{x['object_macro_cosine_mae']:.6f} | {x['object_macro_std_ratio']:.4f} | "
                 f"{x['object_macro_linear_slope']:.4f} |")
L+=["","## PredObj cache",""]
if cache:
    for k,v in [
      ("annotations",cache.get("annotations")),("unique image IDs",cache.get("unique_image_ids")),
      ("bg_thresh",cache.get("meta",{}).get("bg_thresh")),("PAMR",cache.get("meta",{}).get("pamr")),
      ("projector SHA match",cache.get("projector_sha_match")),
      ("val overlap",cache.get("cache_val_overlap_count")),
      ("IDs outside train",cache.get("cache_ids_not_in_train_count"))]:
        L.append(f"- {k}: {v}")
L+=["","## 117 full-image evaluator proof",""]
if full:
    for k,v in full.items():
      if k!="source" and k!="derived117": L.append(f"- {k}: {v}")
L+=["","## GT-only anchor / RelProto semantic audit",""]
if anchor:
    L+=["| state | anchor hit | anchor majority | anchor purity | support hit | support purity | positive-R anchor K>1 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s,x in anchor.get("states",{}).items():
      L.append(f"| {s} | {x['anchor_any_gt_hit_rate']:.4f} | {x['anchor_gt_majority_rate']:.4f} | "
               f"{x['anchor_mean_gt_pixel_purity']:.4f} | {x['support_patch_any_gt_hit_rate']:.4f} | "
               f"{x['support_patch_mean_gt_pixel_purity']:.4f} | "
               f"{x['positive_anchor_relative_fraction_Kgt1']:.4f} |")
L+=["","## Loss scale",""]
if scale:
    for k in ("last_train_object","last_train_structure","last_train_weighted_structure",
              "weighted_structure_over_object","contrastive_extra_batch_squared_division_present"):
        L.append(f"- {k}: {scale.get(k)}")
def val(l,p):
    a=[x["miou"] for x in evals if x["label"]==l and x["protocol"]==p]
    return a[-1] if a else None
fmt=lambda x:"NA" if x is None else f"{x:.2f}"
L+=["","## Causal table to inspect first","",
    "| model | direct 116 | direct 117 | fixed-cache + W 116 | fixed-cache + W 117 |",
    "|---|---:|---:|---:|---:|"]
for b in ("initial","ft0","partstruct"):
    L.append(f"| {b} | {fmt(val(b+'_direct','part116_ignorebg'))} | "
             f"{fmt(val(b+'_direct','part117_bg040'))} | "
             f"{fmt(val(b+'_fixedcache_W','part116_ignorebg'))} | "
             f"{fmt(val(b+'_fixedcache_W','part117_bg040'))} |")
L+=["","## Interpretation guardrails","",
"- **FT0 vs PartStruct** isolates the structure term from identical extra COCO fine-tuning.",
"- **initial / FT0 / PartStruct + W on the same fixed cache** isolates text-side effects from PredObj crop changes.",
"- Spearman near 1 preserves rank, not cosine-margin magnitudes; cosine geometry and anchor purity test this missing mechanism.",
"- 117 is a custom full-image protocol: original 255 is redefined as background 0; it is not the official foreground-only PascalPart116 metric.",
"- GT in the anchor audit is post-hoc diagnosis only and must never be described as W-training supervision.",
"- If PartStruct loses to FT0 under the same cache, the strong causal story 'structure preservation improves correspondence' is not supported by this setup."]
summary="\n".join(L)+"\n"; (root/"FINAL_SUMMARY.md").write_text(summary); print(summary)

C=["=== NIGHTLY STATUS ==="]+[f"{a}\t{b}" for a,b,_ in status]
C+=["","=== CONTROLLED EVAL ==="]+[f"{x['label']}\t{x['protocol']}\t{x['miou']:.2f}" for x in evals]
if struct:
  C+=["","=== STRUCTURE ==="]
  for l,x in struct.get("models",{}).items():
    if "exact_macro_spearman" in x:
      C.append(f"{l}\tSpearman={x['exact_macro_spearman']:.8f}\t"
               f"cosMAE={x['object_macro_cosine_mae']:.8f}\t"
               f"stdRatio={x['object_macro_std_ratio']:.6f}\t"
               f"slope={x['object_macro_linear_slope']:.6f}")
if anchor:
  C+=["","=== ANCHOR/PURITY ==="]
  for l,x in anchor.get("states",{}).items():
    C.append(f"{l}\tanchorHit={x['anchor_any_gt_hit_rate']:.5f}\t"
             f"anchorPurity={x['anchor_mean_gt_pixel_purity']:.5f}\t"
             f"supportHit={x['support_patch_any_gt_hit_rate']:.5f}\t"
             f"supportPurity={x['support_patch_mean_gt_pixel_purity']:.5f}\t"
             f"positiveR={x['positive_anchor_relative_fraction_Kgt1']:.5f}")
if full: C+=["","=== FULL117 ===",json.dumps(full,ensure_ascii=False)]
C+=["",f"FULL_REPORT={root/'FINAL_SUMMARY.md'}"]
(root/"RESULTS_TO_SEND.txt").write_text("\n".join(C)+"\n")
PY
}
run_step "26_generate_final_summary" final_report

echo
echo "======================================================================"
echo "NIGHTLY AUDIT FINISHED: $(date -Is)"
echo "Main summary : $OUT/FINAL_SUMMARY.md"
echo "Paste to me  : $OUT/RESULTS_TO_SEND.txt"
echo "Master log   : $MASTER_LOG"
echo
echo "Tomorrow send:"
echo "  cat '$OUT/RESULTS_TO_SEND.txt'"
echo
echo "Step status:"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo "======================================================================"

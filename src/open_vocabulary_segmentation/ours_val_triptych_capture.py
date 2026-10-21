#!/usr/bin/env python3
"""Read-only side-effect capture of *exact* predictions returned to the original mmseg evaluator.
Adds no model inference, no GT/object masking to predictions, and no evaluation changes.
The official GT is read via the dataset's get_gt_seg_map_by_idx ONLY after evaluation.
"""
from __future__ import annotations
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Capture:
    def __init__(self, model, loader_dataset):
        self.dataset_name = os.environ["OVPS_VIS_DATASET"].lower()
        self.expected_count = int(os.environ["OVPS_VIS_EXPECTED_COUNT"])
        self.expected_miou = float(os.environ["OVPS_VIS_EXPECTED_MIOU"])
        self.out = Path(os.environ["OVPS_VIS_OUTPUT"]).resolve()
        self.alpha = float(os.environ.get("OVPS_VIS_ALPHA", "1.0"))
        if self.dataset_name not in ("pp116", "ade234"):
            raise RuntimeError("Set OVPS_VIS_DATASET to pp116 or ade234")
        if not 0 <= self.alpha <= 1:
            raise RuntimeError("OVPS_VIS_ALPHA must be within [0,1]")
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"Will not mix with existing visualization run: {self.out}")
        self.rawdir = self.out / "raw_prediction_ids"
        self.rawdir.mkdir(parents=True, exist_ok=True)
        self.core_dataset = self._unwrap(loader_dataset)
        self.infos = getattr(self.core_dataset, "img_infos", None)
        if self.infos is None:
            self.infos = getattr(self.core_dataset, "data_infos", None)
        if self.infos is None:
            raise RuntimeError("Underlying dataset has neither img_infos nor data_infos")
        self.idx_by_stem = {}
        for i, info in enumerate(self.infos):
            p = info.get("filename", info.get("img_path", ""))
            if not p and isinstance(info.get("img_info"), dict):
                p = info["img_info"].get("filename", "")
            stem = Path(str(p)).stem
            if not stem or stem in self.idx_by_stem:
                raise ValueError(f"Non-unique / empty val filename stem {stem!r}")
            self.idx_by_stem[stem] = i
        self.images = {}
        self.count = 0
        original = model.forward_test
        capture = self

        def intercepted(this, *args, **kwargs):
            # The *only* inference call: unchanged forward_test, unchanged return value.
            prediction = original(*args, **kwargs)
            metas = kwargs.get("img_metas", args[1] if len(args) >= 2 else None)
            capture._save_prediction(prediction, metas)
            return prediction

        model.forward_test = types.MethodType(intercepted, model)
        print(f"[TRIPTYCH CAPTURE ATTACHED] {self.dataset_name}: "
              f"{len(self.infos)} image records; hook=forward_test", flush=True)

    @staticmethod
    def _unwrap(dataset):
        for _ in range(6):
            inner = getattr(dataset, "dataset", None)
            if inner is None:
                break
            dataset = inner
        return dataset

    @staticmethod
    def _flatten_metas(o):
        if isinstance(o, dict):
            if "ori_filename" in o or "filename" in o:
                yield o
            else:
                for v in o.values():
                    yield from Capture._flatten_metas(v)
        elif isinstance(o, (tuple, list)):
            for v in o:
                yield from Capture._flatten_metas(v)
        else:
            # DataContainer compatibility in case custom user wrapper forwards it
            if hasattr(o, "data") and not isinstance(o, (str, bytes, np.ndarray)):
                yield from Capture._flatten_metas(o.data)

    def _save_prediction(self, prediction, img_metas):
        if isinstance(prediction, np.ndarray):
            predictions = [prediction] if prediction.ndim == 2 else list(prediction)
        elif isinstance(prediction, (tuple, list)):
            predictions = list(prediction)
        else:
            raise TypeError(f"Unexpected prediction return type: {type(prediction)}")
        metas = list(self._flatten_metas(img_metas))
        if len(predictions) != 1 or len(metas) != 1:
            raise RuntimeError(f"Expected single-image inference, got {len(predictions)} preds, {len(metas)} metas")
        meta = metas[0]
        filename = str(meta.get("ori_filename") or meta.get("filename") or "")
        stem = Path(filename).stem
        if stem not in self.idx_by_stem:
            alternative = Path(str(meta.get("filename", ""))).stem
            if alternative in self.idx_by_stem:
                stem = alternative
            else:
                raise KeyError(f"Prediction metadata {filename!r} not in validation dataset")
        if stem in self.images:
            raise RuntimeError(f"Duplicate prediction: {stem}")
        pred = predictions[0]
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred)
        if pred.ndim != 2 or pred.dtype.kind not in "biu":
            raise ValueError(f"Not a 2D integer label mask: {stem} {pred.shape} {pred.dtype}")
        expected_classes = 116 if self.dataset_name == "pp116" else 234
        if np.any(pred < 0) or np.any(pred >= expected_classes):
            raise ValueError(f"Invalid predicted semantic IDs for {stem}")
        pred = pred.astype(np.uint8, copy=False)
        np.save(self.rawdir / f"{stem}.npy", pred, allow_pickle=False)
        self.images[stem] = {"meta_filename": str(meta.get("filename", "")),
                             "meta_ori_filename": str(meta.get("ori_filename", "")),
                             "index": self.idx_by_stem[stem]}
        self.count += 1
        if self.count % 200 == 0:
            print(f"[TRIPTYCH CAPTURE] {self.count}/{self.expected_count}", flush=True)

    def _palette(self, count):
        # Prefer the user's original visualization palette when callable.
        try:
            import sys
            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
            from visualize_pp116_final import make_palette
            sig = inspect.signature(make_palette)
            required = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            if len(required) == 0:
                arr = np.asarray(make_palette())
            elif len(required) == 1:
                arr = np.asarray(make_palette(count))
            else:
                arr = None
            if arr is not None and arr.ndim == 2 and arr.shape[0] >= count and arr.shape[1] == 3:
                if np.isfinite(arr[:count]).all():
                    print("[PALETTE] Existing visualize_pp116_final.make_palette", flush=True)
                    return np.clip(arr[:count], 0, 255).astype(np.uint8)
        except Exception as ex:
            print(f"[PALETTE] Existing make_palette unavailable: {type(ex).__name__}: {ex}", flush=True)
        # Reuse an official dataset palette if it exists.
        official = getattr(self.core_dataset, "PALETTE", None)
        if official is not None:
            arr = np.asarray(official)
            if arr.ndim == 2 and arr.shape[0] >= count and arr.shape[1] == 3:
                print("[PALETTE] Dataset.PALETTE", flush=True)
                return np.clip(arr[:count], 0, 255).astype(np.uint8)
        print("[PALETTE] VOC bit-interleaved stable palette", flush=True)
        out = np.zeros((count, 3), np.uint8)
        for idx in range(count):
            lab = idx
            for bit in range(8):
                for ch in range(3):
                    out[idx, ch] |= ((lab >> ch) & 1) << (7-bit)
                lab >>= 3
        return out

    def _find_image(self, stem, meta, info):
        pths = [meta.get("meta_filename", ""), meta.get("meta_ori_filename", ""),
                str(info.get("filename", "")), str(info.get("img_path", ""))]
        img_dir = str(getattr(self.core_dataset, "img_dir", ""))
        data_root = str(getattr(self.core_dataset, "data_root", ""))
        manual = os.environ.get("OVPS_VIS_IMAGE_ROOT", "")
        candidates = []
        for txt in pths:
            if not txt:
                continue
            p = Path(txt)
            candidates.append(p)
            if img_dir:
                candidates.append(Path(img_dir) / p)
                candidates.append(Path(img_dir) / p.name)
            if data_root:
                candidates.append(Path(data_root) / p)
        if manual:
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidates.append(Path(manual) / f"{stem}{ext}")
        for p in candidates:
            if p.is_file():
                return p
        raise FileNotFoundError(f"Cannot find RGB image for {stem}; dataset img_dir={img_dir!r}; "
                                f"meta={meta}; use OVPS_VIS_IMAGE_ROOT")

    @staticmethod
    def _title(im, label):
        w, h = im.size
        bar = 35
        out = Image.new("RGB", (w, h+bar), "white")
        out.paste(im, (0, bar))
        d = ImageDraw.Draw(out)
        d.rectangle((0, 0, w-1, bar-1), fill=(246, 247, 249))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        bbox = d.textbbox((0,0), label, font=font)
        d.text(((w-(bbox[2]-bbox[0]))/2, (bar-(bbox[3]-bbox[1]))/2), label, fill=(10,10,10), font=font)
        return out

    def finish(self, miou):
        print(f"[PROTOCOL CHECK] original evaluator mIoU={miou:.4f}; expected={self.expected_miou:.4f}", flush=True)
        if abs(float(miou)-self.expected_miou) > 0.011:
            raise RuntimeError("mIoU does NOT match target: refusing to draw misleading triptychs. "
                               f"Got {miou}, expected {self.expected_miou}. Check baked projector & config.")
        if self.count != self.expected_count or len(self.images) != self.expected_count:
            raise RuntimeError(f"Incomplete predictions: {self.count} of {self.expected_count}")
        if len(self.infos) != self.expected_count:
            raise RuntimeError(f"Validation length unexpected: {len(self.infos)} vs {self.expected_count}")
        count = 116 if self.dataset_name == "pp116" else 234
        colors = self._palette(count)
        outdir = self.out / "triptychs"
        outdir.mkdir(exist_ok=True)
        rows = []
        for k, stem in enumerate(sorted(self.images)):
            meta = self.images[stem]
            idx = meta["index"]
            gt = np.asarray(self.core_dataset.get_gt_seg_map_by_idx(idx))
            pred = np.load(self.rawdir / f"{stem}.npy", allow_pickle=False)
            if gt.ndim != 2 or pred.shape != gt.shape:
                raise RuntimeError(f"GT vs eval prediction shape mismatch: {stem} {gt.shape} vs {pred.shape}")
            info = self.infos[idx]
            img_path = self._find_image(stem, meta, info)
            im = np.asarray(Image.open(img_path).convert("RGB"))
            if im.shape[:2] != gt.shape:
                raise RuntimeError(f"Original RGB vs GT spatial mismatch: {stem} {im.shape[:2]} vs {gt.shape}")
            # GT valid pixels are EXACTLY the only pixels recolored in BOTH panes.
            valid = (gt >= 0) & (gt < count)
            gtcol = im.copy()
            predcol = im.copy()
            if self.alpha == 1:
                gtcol[valid] = colors[gt[valid].astype(np.int64)]
                predcol[valid] = colors[pred[valid].astype(np.int64)]
            else:
                x = im[valid].astype(np.float32)
                gtcol[valid] = np.rint(x*(1-self.alpha)+self.alpha*colors[gt[valid].astype(np.int64)]).astype(np.uint8)
                predcol[valid] = np.rint(x*(1-self.alpha)+self.alpha*colors[pred[valid].astype(np.int64)]).astype(np.uint8)
            images = [self._title(Image.fromarray(im), "Input"),
                      self._title(Image.fromarray(gtcol), "GT"),
                      self._title(Image.fromarray(predcol), "Ours")]
            gap = 8
            merged = Image.new("RGB", (3*im.shape[1]+2*gap, im.shape[0]+35), "white")
            for j, panel in enumerate(images):
                merged.paste(panel, (j*(im.shape[1]+gap),0))
            dst = outdir / f"{stem}.png"
            merged.save(dst, compress_level=2)
            rows.append({"image_id":stem,"dataset_index":idx,"rgb_path":str(img_path),
                         "raw_pred":str(self.rawdir / f"{stem}.npy"),"triptych":str(dst),
                         "valid_pixels":int(valid.sum()),"height":int(gt.shape[0]),"width":int(gt.shape[1])})
            if (k+1)%200==0:
                print(f"[TRIPTYCH RENDER] {k+1}/{self.expected_count}", flush=True)
        with (self.out / "manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (self.out / "protocol.json").write_text(json.dumps({
            "dataset":self.dataset_name,"verified_miou":float(miou),"expected_miou":self.expected_miou,
            "n_images":self.count,"n_classes":count,"alpha":self.alpha,
            "predictions":"exact forward_test outputs consumed by original us.multi_gpu_test/pre_eval",
            "GT":"official dataset.get_gt_seg_map_by_idx; valid 0 <= GT < n_classes",
            "model_provenance":"Original evaluation config and baked projector selected by run script",
        },indent=2), encoding="utf-8")
        print(f"[TRIPTYCH SUCCESS] {self.count} images -> {outdir}",flush=True)

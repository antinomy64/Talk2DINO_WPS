#!/usr/bin/env python3
"""No GPU or real data: test strict seen filtering and true 74-way InfoNCE."""
from __future__ import annotations
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from seen_split import UNSEEN_OBJECTS, split_from_text_names, filter_seen_examples
from eval_seen_unseen import parse_official_output


def make_names():
    # Counts checked against original PP116 taxonomy; *no assumed hard-coded IDs*.
    by_obj = dict(aeroplane=6, bicycle=5, bird=9, bottle=2, bus=10,
                  car=10, cat=9, cow=9, dog=10, horse=9, motorbike=4,
                  person=14, pottedplant=2, sheep=9, train=7, tvmonitor=1)
    return [f"{obj}'s placeholder_part_{i}" for obj, count in by_obj.items() for i in range(count)]


def main():
    names = make_names()
    spec = split_from_text_names(names)
    assert len(spec["seen_ids"]) == 74 and len(spec["unseen_ids"]) == 42
    assert len(spec["seen_objects"]) == 11 and len(spec["unseen_objects"]) == 5
    assert set(spec["unseen_objects"]) == set(UNSEEN_OBJECTS)
    exs = [SimpleNamespace(row_index=i, class_name=names[i].split("'s ",1)[0],
                           pids=torch.tensor([i])) for i in range(len(names))]
    rows = [{"row_index": i, "class_name": e.class_name, "status": "ready"} for i,e in enumerate(exs)]
    kept, report, _ = filter_seen_examples(exs, {"rows":rows}, names)
    assert len(kept) == 74
    assert len(report["rows"]) == 74 and report["excluded_unseen_annotations"] == 42
    assert all(e.class_name not in UNSEEN_OBJECTS for e in kept)
    for e in kept:
        assert set(e.pids.tolist()).issubset(set(spec["seen_ids"]))
    # An unseen annotation mixing in a seen part ID is a taxonomy error.
    bad = list(exs)
    bad[-1] = SimpleNamespace(row_index=115, class_name="sheep", pids=torch.tensor([0]))
    try:
        filter_seen_examples(bad, {"rows": rows}, names)
    except ValueError:
        pass
    else:
        raise AssertionError("Foreign-object target did not fail")
    print("PASS: PP116 74/42 split, original 116 IDs unchanged, unseen rows excluded")

    torch.manual_seed(12)
    logits = torch.randn(4, 116, requires_grad=True)
    seen_ids = list(spec["seen_ids"])
    unseen_ids = list(spec["unseen_ids"])
    targets = torch.tensor([seen_ids[0], seen_ids[10], seen_ids[32], seen_ids[-1]])
    allowed = torch.zeros(116,dtype=torch.bool)
    allowed[seen_ids] = True
    new_targets = torch.tensor([seen_ids.index(int(x)) for x in targets])
    direct = F.cross_entropy(logits[:, seen_ids], new_targets)
    masked = F.cross_entropy(logits.masked_fill(~allowed[None,:], -torch.inf), targets)
    assert torch.allclose(direct, masked, atol=1e-6), (direct, masked)
    direct.backward()
    assert float(logits.grad[:, unseen_ids].abs().max()) == 0
    print("PASS: exact 74-way InfoNCE equals masked global; all unseen logit gradients are zero")

    # Emulate the output format of the unchanged official 116-way evaluation.
    text = 'per class results:\n| Class | IoU | Acc |\n'
    text += '\n'.join(f'| {name} | 12.0 | 13.0 |' for name in names)
    text += '\nSummary:\nmIoU of 850 test images: 12.0%\n'
    vals, recs = parse_official_output(text)
    assert len(recs) == 116
    assert abs(vals['seen74_miou']-12)<1e-8
    assert abs(vals['unseen42_miou']-12)<1e-8
    print("PASS: unchanged 116-class official log => seen74/unseen42/harmonic summary")
    print("SELF_TEST_PASS_PP116_SEEN_ONLY")


if __name__ == "__main__":
    main()

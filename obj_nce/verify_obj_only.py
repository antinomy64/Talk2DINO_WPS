#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only, no dataset required: check exact object-only NCE math, gradients, taxonomy."""
from pathlib import Path
from types import SimpleNamespace
import csv
import math
import torch
import torch.nn.functional as F

from . import train_pp116_obj_nce as m


def main():
    csv_path = Path(__file__).with_name('taxonomy_negative_sets_reference_116.csv')
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        entries = list(csv.DictReader(f))
    entries.sort(key=lambda r: int(r['part_id']))
    names = [e['positive_part'] for e in entries]
    assert len(names) == 116

    m._TAXONOMY_CLASS_NAMES[:] = names
    m._OBJECT_ONLY_NEG_MASK = m.build_object_only_negative_mask(names)
    m._TAXONOMY_NEG_MASK = m.build_taxonomy_negative_mask(names)
    mask = m._OBJECT_ONLY_NEG_MASK
    assert mask.shape == (116, 116)
    assert torch.equal(mask, mask.T)
    assert not bool(mask.diagonal().any())
    for i, row in enumerate(entries):
        expected = int(row['same_object_negative_count'])
        assert int(mask[i].sum()) == expected, (i, expected)
        for j in mask[i].nonzero(as_tuple=True)[0].tolist():
            assert names[i].split("'s ")[0].lower() == names[j].split("'s ")[0].lower()

    counts = mask.sum(1)
    isolated = [names[i] for i in (counts == 0).nonzero(as_tuple=True)[0].tolist()]
    assert isolated == ["tvmonitor's screen"], isolated
    assert int(counts.max()) == 13
    assert abs(float(counts.float().mean()) - 7.931034482758621) < 1e-7

    # 2 examples with different sets of present classes, including K=1.
    torch.manual_seed(123)
    v = F.normalize(torch.randn(3, 1, 768), dim=-1).requires_grad_(True)
    t = F.normalize(torch.randn(116, 768), dim=-1).requires_grad_(True)
    idx = {
        key: names.index(key) for key in ("cow's horn", "person's ear", "tvmonitor's screen")
    }
    pids = torch.tensor([[idx["cow's horn"]], [idx["person's ear"]], [idx["tvmonitor's screen"]]])
    args = SimpleNamespace(nce_variant='object_only', nce_temperature=0.07)
    losses = m.experimental_nce_losses(v, t, pids, None, None, args).reshape(-1)
    logits = v.detach().reshape(-1, 768) @ t.T / 0.07
    for row in range(3):
        pid = int(pids[row, 0])
        selected = [pid] + mask[pid].nonzero(as_tuple=True)[0].tolist()
        manual = torch.logsumexp(logits[row, selected], 0) - logits[row, pid]
        assert torch.allclose(losses[row], manual, atol=2e-6), (row, losses[row], manual)
    assert losses[-1].abs().item() == 0.0, 'Singleton object must have zero NCE loss'
    g_v, g_t = torch.autograd.grad(losses.sum(), (v, t), allow_unused=True)
    assert g_v is None, 'Stop-gradient through visual target must hold'
    assert g_t is not None and bool(torch.isfinite(g_t).all())
    assert float(g_t.abs().sum()) > 0

    args.nce_variant = 'global'
    global_loss = m.experimental_nce_losses(v.detach(), t, pids, None, None, args)
    expected_global = F.cross_entropy(logits, pids.reshape(-1), reduction='none').reshape(3, 1)
    assert torch.allclose(global_loss, expected_global)
    assert (global_loss.reshape(-1) + 1e-6 >= losses).all(), 'Masking negatives cannot increase CE'
    print('PASS: 116-class taxonomy and all 16 object groups')
    print('PASS: no cross-object negatives; correct counts and singleton behavior')
    print('PASS: object-only logsumexp equals direct masked cross entropy')
    print('PASS: visual stop-gradient; text receives gradients; global parity')
    print(f'MEAN NEGATIVES: {counts.float().mean().item():.6f}; MIN={counts.min().item()}; MAX={counts.max().item()}')
    print('OBJECT_ONLY_NCE_SELFTEST_PASS')


if __name__ == '__main__':
    main()

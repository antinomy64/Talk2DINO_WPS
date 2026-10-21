#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADE234 Soft Relative + Structure joint projector + orthogonal W experiment."""
from functools import partial
import train_relproto_soft_structure_ade234 as variant
import train_relproto_alignment_ade20kpart234 as original
import ovps_joint_projector_core as core


def main():
    parser = core.add_options(variant.build_parser())
    args = parser.parse_args()
    variant.validate_args(args)
    core.check_options(args)
    if args.self_test:
        variant.self_test()
        core.self_test_joint()
        return 0
    original.do_train = partial(core.train_joint, original=original,
                                variant=variant, dataset='ade234')
    original.train_and_eval(args)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

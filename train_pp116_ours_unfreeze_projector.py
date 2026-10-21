#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 SoftSingle (presence) 100-epoch joint projector + orthogonal W experiment."""
from functools import partial
import train_pp116_softsingle_dual as variant
import train_relproto_alignemt as original
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
    if args.part_setting != 'presence':
        raise ValueError('PP116 Ours baseline=36.75 uses --part_setting presence')
    variant._SETTING = args.part_setting
    original.prepare_examples = variant.prepare_examples_checked
    original.do_train = partial(core.train_joint, original=original,
                                variant=variant, dataset='pp116')
    original.train_and_eval(args)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

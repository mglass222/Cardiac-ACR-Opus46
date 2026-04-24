#!/usr/bin/env python
# coding: utf-8

"""
Unified CLI for the Cardiac-ACR pipeline.

Usage:
    python -m cardiac_acr preprocess {extract-patches,split}
    python -m cardiac_acr train        --backend {uni,resnet}
    python -m cardiac_acr evaluate     --backend {uni,resnet} [--checkpoint PATH]
    python -m cardiac_acr diagnose-wsi --backend {uni,resnet} [--checkpoint PATH]
    python -m cardiac_acr check-deps

Each subcommand delegates to the equivalent module entry point, which
can still be invoked directly (``python -m cardiac_acr.backends.uni.train``
etc.). The CLI exists so users don't have to remember every module
path.
"""

import argparse
import sys


def _preprocess(args):
    if args.stage == "extract-patches":
        from cardiac_acr.preprocessing import extract_patches
        extract_patches.main()
    elif args.stage == "split":
        from cardiac_acr.preprocessing import create_training_sets
        create_training_sets.main()


def _train(args):
    if args.backend == "uni":
        from cardiac_acr.backends.uni import train
        train.main()
    else:  # resnet
        from cardiac_acr.backends.resnet import train
        train.main()


def _evaluate(args):
    if args.backend == "uni":
        from cardiac_acr.backends.uni import evaluate
        evaluate.main()
    else:
        raise SystemExit(
            "evaluate: the resnet backend uses stats/ scripts instead — "
            "see python -m cardiac_acr.backends.resnet.stats.patch_level_stats"
        )


def _diagnose_wsi(args):
    from cardiac_acr.wsi import diagnose
    diagnose.run(args.backend, checkpoint_path=args.checkpoint)


def _check_deps(args):
    from cardiac_acr.utils import check_dependencies
    check_dependencies.main()


def _build_parser():
    p = argparse.ArgumentParser(prog="cardiac_acr")
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("preprocess", help="Run a preprocessing stage")
    pre.add_argument("stage",
                     choices=("extract-patches", "split"))
    pre.set_defaults(func=_preprocess)

    tr = sub.add_parser("train", help="Train the patch classifier")
    tr.add_argument("--backend", choices=("uni", "resnet"), required=True)
    tr.set_defaults(func=_train)

    ev = sub.add_parser("evaluate", help="Evaluate on the validation split")
    ev.add_argument("--backend", choices=("uni", "resnet"), required=True)
    ev.add_argument("--checkpoint", default=None)
    ev.set_defaults(func=_evaluate)

    dx = sub.add_parser("diagnose-wsi", help="Run WSI diagnosis on test slides")
    dx.add_argument("--backend", choices=("uni", "resnet"), required=True)
    dx.add_argument("--checkpoint", default=None)
    dx.set_defaults(func=_diagnose_wsi)

    cd = sub.add_parser("check-deps", help="Verify runtime dependencies")
    cd.set_defaults(func=_check_deps)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

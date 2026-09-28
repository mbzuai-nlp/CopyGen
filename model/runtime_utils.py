#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def apply_cuda_visible_devices_from_argv(argv: list[str], *, default: str | None = None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-visible-devices", default=None)
    args, _ = parser.parse_known_args(argv[1:])

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        return args.cuda_visible_devices

    if default is not None and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = default
        return default

    return os.environ.get("CUDA_VISIBLE_DEVICES")

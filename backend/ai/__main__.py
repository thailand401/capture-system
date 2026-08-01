"""Command-line entry point for the recognition engine.

Usage::

    python -m ai build                 # (re)build memory from dataset/
    python -m ai append <dir> <sign>   # append new samples for a sign id
    python -m ai recognize <image>     # run the full pipeline on an image
    python -m ai benchmark <image>     # run + print per-stage timings
    python -m ai promote               # promote eligible candidates
    python -m ai prune                 # remove redundant permanent vectors
    python -m ai stats                 # print memory statistics
    python -m ai export <path.zip>     # export the memory bundle
    python -m ai import <path.zip>     # import a memory bundle

Kept intentionally thin: it only parses arguments and delegates to the
factory-built objects.
"""

from __future__ import annotations

import argparse
import json
import sys

from ai.config import EngineConfig
from ai.factory import build_memory_manager, build_online_memory, build_pipeline
from ai.utils.logging import configure_logging, get_logger

logger = get_logger("cli")


def _cmd_build(_: argparse.Namespace, config: EngineConfig) -> int:
    count = build_memory_manager(config).rebuild()
    logger.info("build_done", vectors=count)
    return 0


def _cmd_append(args: argparse.Namespace, config: EngineConfig) -> int:
    added = build_memory_manager(config).append_directory(args.directory, args.sign_id)
    logger.info("append_done", added=added, sign_id=args.sign_id)
    return 0


def _cmd_recognize(args: argparse.Namespace, config: EngineConfig) -> int:
    result = build_pipeline(config).run(args.image)
    print(json.dumps([p.model_dump() for p in result.predictions], indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace, config: EngineConfig) -> int:
    result = build_pipeline(config).run(args.image)
    print(json.dumps(result.timings_ms, indent=2))
    return 0


def _online(config: EngineConfig):
    online = build_online_memory(config)
    online.ensure_ready()
    return online


def _cmd_promote(_: argparse.Namespace, config: EngineConfig) -> int:
    result = _online(config).promote()
    logger.info("promote_done", promoted=len(result.promoted_ids))
    print(json.dumps({"promoted": result.promoted_ids, "reasons": result.reasons}, indent=2))
    return 0


def _cmd_prune(_: argparse.Namespace, config: EngineConfig) -> int:
    removed = _online(config).prune()
    logger.info("prune_done", removed=removed)
    return 0


def _cmd_stats(_: argparse.Namespace, config: EngineConfig) -> int:
    print(json.dumps(_online(config).statistics().model_dump(), indent=2))
    return 0


def _cmd_export(args: argparse.Namespace, config: EngineConfig) -> int:
    path = _online(config).export(args.path)
    logger.info("export_done", path=str(path))
    return 0


def _cmd_import(args: argparse.Namespace, config: EngineConfig) -> int:
    _online(config).import_(args.path)
    logger.info("import_done", path=args.path)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="ai", description="Traffic Sign Recognition Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Rebuild memory from the dataset")

    append = sub.add_parser("append", help="Append new sample images for a sign id")
    append.add_argument("directory")
    append.add_argument("sign_id")

    recognize = sub.add_parser("recognize", help="Run the pipeline on an image")
    recognize.add_argument("image")

    benchmark = sub.add_parser("benchmark", help="Run the pipeline and print timings")
    benchmark.add_argument("image")

    sub.add_parser("promote", help="Promote eligible candidates into permanent memory")
    sub.add_parser("prune", help="Remove redundant permanent vectors")
    sub.add_parser("stats", help="Print memory statistics")

    export = sub.add_parser("export", help="Export the memory bundle to a .zip")
    export.add_argument("path")

    import_ = sub.add_parser("import", help="Import a memory bundle from a .zip")
    import_.add_argument("path")

    args = parser.parse_args(argv)
    config = EngineConfig()

    handlers = {
        "build": _cmd_build,
        "append": _cmd_append,
        "recognize": _cmd_recognize,
        "benchmark": _cmd_benchmark,
        "promote": _cmd_promote,
        "prune": _cmd_prune,
        "stats": _cmd_stats,
        "export": _cmd_export,
        "import": _cmd_import,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())

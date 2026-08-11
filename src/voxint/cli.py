"""Voxint CLI: submit, status, requeue, score."""

import argparse

from voxint import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxint", description="Voxint audio pipeline")
    parser.add_argument("--version", action="version", version=f"voxint {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

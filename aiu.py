"""Convenience launcher for AI ML Onboarding."""

from __future__ import annotations

import sys

from deep_agent import cli


def main(argv: list[str] | None = None) -> int:
    return cli.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())

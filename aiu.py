"""Convenience launcher for AI ML Onboarding."""

from __future__ import annotations

import sys

import ml_agent


def main(argv: list[str] | None = None) -> int:
    return ml_agent.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())

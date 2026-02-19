"""Entry point — python -m jace."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jace",
        description="JACE: Autonomous Control Engine",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration YAML file",
        default=None,
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the REST API server",
    )
    args = parser.parse_args()

    from jace.app import Application

    app = Application(config_path=args.config)
    try:
        asyncio.run(app.start(api=args.api))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

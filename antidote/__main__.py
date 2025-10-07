"""Antidote dev placeholder"""

import argparse
import pathlib
import logging
from .__version__ import version as __version__  # versioning is handled by setuptools-scm

import antidote.commands.tools
import antidote.commands.inference
import antidote.commands.train


def setup_logging(log_level) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_module_name(module) -> str:
    return pathlib.Path(module.__file__).stem


def main() -> None:
    print(f"Loading Antidote v{__version__}...")
    parser = argparse.ArgumentParser(
        prog="Antidote",
        description="A Neural network Trained In Deleterious ObjecT Elimination",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Antidote dev",  # + antidote.__version__
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    modules = [
        antidote.commands.tools,
        antidote.commands.inference,
        antidote.commands.train,
    ]

    subparsers = parser.add_subparsers(title="Choose a command", dest="command")
    subparsers.required = True

    for module in modules:
        module_name = get_module_name(module)
        this_parser = subparsers.add_parser(module_name, description=module.__doc__)
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Antidote with command: {args.command}")
    logger.debug(f"Arguments: {args}")

    args.func(args)


if __name__ == "__main__":
    main()

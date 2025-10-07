"""
Utility tools to process data related to Antidote.
"""

import argparse
import logging
import pathlib

from antidote.utils import plot_particle_overlaps
from antidote.utils import split_starfile
from antidote.utils import convert_cryosparc

logger = logging.getLogger(__name__)


def add_args(parser):
    """
    Add command line arguments to the parser. This function also specifies
    a series of subparsers that define arguments that are specific to each
    subcommand. This facilitates calling analysis scripts from the command line,
    for example:

    ~ antidote tools particle_overlaps --arg1 --arg2

    ...etc.

    Args:
    - parser (argparse.ArgumentParser): Argument parser object.

    Returns:
    - argparse.ArgumentParser: Parser with added arguments.
    """

    subparsers = parser.add_subparsers(dest="analysis_workflow")

    # particle-overlaps command
    particle_overlaps = subparsers.add_parser(
        "particle-overlaps",
        help="Match particles from different sources based on particle coordinates and micrograph IDs.",
    )
    particle_overlaps.add_argument(
        "-i",
        "--input-files",
        required=True,
        type=pathlib.Path,
        nargs=2,
        help="Path to two input STAR and/or cryoSPARC files to compare.",
    )

    particle_overlaps.add_argument(
        "-o",
        "--output",
        required=False,
        default="./particle_overlap_plot.png",  # will output to working directory if not specified
        type=pathlib.Path,
        help="Path to output.",
    )

    particle_overlaps.add_argument(
        "-t",
        "--tolerance",
        required=False,
        default=40,
        type=int,
        help="Tolerance for particle coordinate mismatches in pixels.",
    )

    # split-starfile command
    split_starfile = subparsers.add_parser(
        "split-starfile",
        help="Split a starfile with Antidote-derived labels into two starfiles above and below some threshold -t",
    )

    split_starfile.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to the input starfile",
    )

    split_starfile.add_argument(
        "-o",
        "--output-directory",
        required=False,
        type=pathlib.Path,
        help="Path to output.",
    )

    # putting these here instead of adding it to the main parser object makes more sense for the user.
    split_starfile.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag for overwriting the output file.",
    )

    split_starfile.add_argument(
        "--antidote-label-field-name",
        default="rlnHelicalTrackLength",
        type=str,
        help='The name of the field that contains the Antidote-derived labels. Default is "rlnHelicalTrackLength".',
    )

    split_starfile.add_argument(
        "-t",
        "--threshold",
        required=True,
        type=float,
        help="Threshold score for starfile split.",
    )

    # convert-cryosparc command
    convert_cryosparc = subparsers.add_parser(
        "convert-cryosparc",
        help="Convert a cryoSPARC job of .cs files to a .star file and symlink appropriate .mrc files.",
    )

    convert_cryosparc.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to the input cryoSPARC job directory.",
    )

    convert_cryosparc.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="Path to output folder (where the `relion` command will be run).",
    )

    convert_cryosparc.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag for overwriting the output file.",
    )

    convert_cryosparc.add_argument(
        "--no-inverty",
        action="store_true",
        default=False,
        help="Do not invert the y-axis of the particles. By default the inversion is done, due to the way cryoSPARC handles particles.",
    )

    return parser


def main(args):
    if args.analysis_workflow == "particle-overlaps":
        print("Generating particle overlap plot...")
        logger.info("Generating particle overlap plot...")
        plot_particle_overlaps.run(
            args.input_files[0],
            args.input_files[1],
            args.output,
            args.tolerance,
        )

    elif args.analysis_workflow == "split-starfile":
        print(f"Splitting {args.input} at a threshold of {args.threshold}...")
        logger.info(f"Splitting {args.input} at a threshold of {args.threshold}...")
        if args.output_directory is None:
            args.output_directory = args.input.parent

        split_starfile.run(args.input, args.output_directory, args.threshold, args.antidote_label_field_name, args.overwrite)

    elif args.analysis_workflow == "convert-cryosparc":
        print(f"Converting cryoSPARC job to star file and symlinked mrcs at {args.input}...")
        logger.info(f"Converting cryoSPARC job to star file and symlinked mrcs at {args.input}...")
        convert_cryosparc.run(args.input, args.output, args.overwrite, args.no_inverty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)

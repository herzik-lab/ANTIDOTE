"""
Run inference using an Antidote model.
"""

import argparse
from pathlib import Path
import starfile
import os
import logging
import copy

from antidote.reports import inference_report
from antidote.utils import get_recommended_threshold
from antidote.utils import class3D_builder
from antidote.utils import tensorflow_tools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger(__name__)
logging.getLogger("markdown_it").setLevel(logging.ERROR)


def add_args(parser):
    """
    Add command line arguments to the parser.

    Args:
    - parser (argparse.ArgumentParser): Argument parser object.

    Returns:
    - argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--antidote-label-field-name",
        default="rlnHelicalTrackLength",
        type=str,
        help='The name of the field in the output starfile that ANTIDOTE will use for particle predictions. Defaults to "rlnHelicalTrackLength" for convenient use with RELION.',
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to input folder containing the RELION Class3D job and *.star files.",
    )

    parser.add_argument(
        "--no-full-report",
        action="store_true",
        default=False,
        help="Do not generate the full inference report (which can take a long time).",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of cores to use for starfile parsing.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,  # default set in main()
        type=Path,
        help="Path to model.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Returns predictions that are a weighted ensemble of 2 antidote models. ",
    )

    parser.add_argument(
        "--min-iteration",
        type=int,
        default=2,
        help="The minimum iteration to consider during inference. The model must also accept this number of iterations. The recommended value is 2.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output folder. If not provided, a folder will be created in the same directory as the input folder.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag for overwriting the output file.",
    )

    parser.add_argument(
        "--secondary-model",
        type=Path,
        help="Path to secondary model for ensemble prediction.",
    )

    parser.add_argument(
        "--symmetry-expansion",
        type=int,
        default=1,
        help="Specify degree of symmetry expansion.",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=threshold_type,
        default=0.5,
        help="Cutoff threshold for accepted particles. Use 'adaptive' for adaptive threshold, or a float between 0 and 1. Default is 0.5.",
    )

    parser.add_argument(
        "--weight",
        type=float,
        help="Weight for the primary model in an ensemble prediction (between 0 and 1).",
    )

    return parser


def threshold_type(value):
    if value.lower() == "adaptive":
        return "adaptive"
    try:
        float_value = float(value)
        if 0 <= float_value <= 1:
            return float_value
        raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Threshold must be 'adaptive' or a float between 0 and 1. The default (0.5) is recommended for most cases."
        )


def run_inference(
    job_df,
    sf,
    model_path,
    label_field_name: str = "rlnHelicalTrackLength",
):
    """
    Run inference with Antidote on the input starfile

    Args:
    -   job_df (pd.DataFrame): input dataframe from a RELION Class3D job
    -   sf (starfile.StarFile): input starfile, to be modified in place with the predictions
    -   model_path (str): path to model
    -   label_field_name (str): The name of the field to use for labels in the output starfile

    Returns:
    -   None, writes a starfile with the antidote prediction mapped to label_field_name
    """
    print("\nRunning Antidote Inference...\n")

    model_path = Path(model_path)

    logger.info(f"Running inference with {model_path.name}... \n")

    model = tensorflow_tools.load_model(model_path)

    # run inference on input starfile using the model assigned above
    job_df[label_field_name] = model.predict(job_df)

    sf["particles"] = sf["particles"].merge(job_df[label_field_name], on="rlnImageName", how="outer").fillna(0)

    return sf


def run_ensemble(
    job_df,
    sf,
    model_path,
    secondary_model_path,
    ensemble_weight,
    label_field_name: str = "rlnHelicalTrackLength",
):
    """
    Run inference with Antidote on the input starfile using an ensemble model. This code is largely the
    same as the run_inference() function.

    Args:
    -   job_df (pd.DataFrame): input dataframe from a RELION Class3D job
    -   sf (starfile.StarFile): input starfile, to be modified in place with the predictions
    -   model_path (str): path to model
    -   secondary_model_path (str): path to secondary (weak) model
    -   ensemble_weight (float): A number between 0 and 1 that represents the primary (strong) model's
                                 weight in the ensemble.
    -   label_field_name (str): The name of the field to use for labels in the output starfile.

    Returns:
    -   None, writes a starfile with the antidote prediction mapped to label_field_name
    """
    print("\nRunning Antidote inference using an ensemble model...\n")

    model_path = Path(model_path)
    secondary_model_path = Path(secondary_model_path)
    # Check that the input is a valid RELION Class3D job and build a Class3D object
    logger.info(f"Loading {model_path.name} as the primary (strong) model... \n")
    model = tensorflow_tools.load_model(model_path)

    logger.info(f"Loading {secondary_model_path.name} as the secondary (weak) model... \n")
    secondary_model = tensorflow_tools.load_model(secondary_model_path)

    # run inference on input starfile using the models assigned above
    logger.info(f"Running inference with {model_path.name}... \n")
    pred1 = model.predict(job_df)

    logger.info(f"Running inference with {secondary_model_path.name}... \n")
    pred2 = secondary_model.predict(job_df)

    logger.info(
        f"Calculating weighted sum of predictions from {model_path.name} (weight = {ensemble_weight}) and {secondary_model_path.name} (weight = {(1 - ensemble_weight):.2f})... \n"
    )
    job_df[label_field_name] = ensemble_weight * pred1 + (1 - ensemble_weight) * pred2

    sf["particles"] = sf["particles"].merge(job_df[label_field_name], on="rlnImageName", how="outer").fillna(0)

    return sf


def main(args):
    """
    Executes inference and a reporting function using arguments provided on the CLI.
    """
    if args.model is None:
        # set base to top-level antidote directory
        project_root = Path(__file__).resolve().parent.parent
        args.model = project_root / "models" / "antidote_main"

    input_path = Path(args.input)

    # read the input starfile
    sf = starfile.read(f"{input_path}/run_it025_data.star")

    if args.antidote_label_field_name in sf["particles"].columns:
        raise ValueError(
            f"Label field name {args.antidote_label_field_name} already exists in the input starfile. Please choose a different name with --antidote-label-field-name."
        )

    # configure output file and dir
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_antidote_output"
    if not args.overwrite and output_path.exists():
        raise FileExistsError(f"Output directory {output_path} already exists. Use --overwrite to overwrite.")
    output_path.mkdir(parents=True, exist_ok=True)

    # warn the user in case of lack of --ensemble flag
    if not args.ensemble and (args.secondary_model is not None or args.weight is not None):
        logger.warning("--secondary-model and --weight are ignored unless --ensemble is set")

    # Check that the input is a valid RELION Class3D job and build a Class3D object
    job_path = class3D_builder.sanitize_input_path(args.input)
    inference_job = class3D_builder.RelionClass3DJob(
        job_path, max_workers=args.max_workers, min_iteration=args.min_iteration, symmetry_order=args.symmetry_expansion
    )
    job_df = inference_job.data

    if args.ensemble:
        if args.secondary_model is None or args.weight is None:
            parser.error("--ensemble requires --secondary-model and --weight")

        predictions_sf = run_ensemble(
            job_df,
            sf,
            args.model,
            args.secondary_model,
            args.weight,
            args.antidote_label_field_name,
        )

    else:
        predictions_sf = run_inference(
            job_df,
            sf,
            args.model,
            args.antidote_label_field_name,
        )

    full_predictions_path = output_path / "full_predictions.star"
    if full_predictions_path.exists():
        full_predictions_path.unlink()
    starfile.write(
        sf,
        full_predictions_path,
    )

    # Set the cutoff threshold
    if args.threshold is None:
        args.threshold = 0.5
    if args.threshold == "adaptive":
        args.threshold = get_recommended_threshold.run(predictions_sf, args.antidote_label_field_name)

    # apply cutoff threshold to output file and write it
    true_predictions_sf = copy.deepcopy(predictions_sf)
    true_predictions_sf["particles"] = true_predictions_sf["particles"][
        true_predictions_sf["particles"][args.antidote_label_field_name] >= args.threshold
    ]

    true_predictions_path = output_path / "true_predictions.star"
    logger.info(f"Writing {true_predictions_path} using a cutoff threshold of {args.threshold}...")

    if true_predictions_path.exists():
        true_predictions_path.unlink()
    starfile.write(
        true_predictions_sf,
        true_predictions_path,
    )

    # Generate an inference report and write the full starfile with all labels
    if args.ensemble:
        logger.warning(
            "Some details of the ensemble model will be missing from the inference report. The predictions histogram represents the final weighted sum values."
        )

    inference_report.generate(
        inference_job,
        predictions_sf,
        output_path,
        args.threshold,
        args.antidote_label_field_name,
        report_format="html",
        no_full_report=args.no_full_report,
    )

    logger.info("Done!")
    print(
        f"Antidote inference complete! Results written to {output_path}. See inference report {output_path}/antidote_full_report.html for details."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)

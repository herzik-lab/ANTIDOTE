"""
Takes a starfile containing Antidote-derived labels and splits it into two starfiles that contain all particles above
and below the threshold specified by the user.

Args:
-   input_path (pathlib.Path): The path to the starfile to be split.
-   output_dir (pathlib.Path): The directory that the output files will be written to. If the user doesn't specify this,
                               the directory is set to the same directory as the input file in tools.py.
-   threshold (float): The Antidote label that will be used to split the input starfile. For example, if the input is 0.5,
                       a starfile will be created containing particles given a label at or above 0.5, and another starfile
                       that contains particles given a label below 0.5.
-   antidote_label_field_name (str): The name of the Antidote label field in the input starfile.
-   overwrite (bool): Whether or not to overwrite output files if they already exist.

Returns:
-   None. For now, this writes two starfiles containing particles above or below the provided threshold.

Limitations:
The input file must have an Antidote label, which is specified by the user (default is rlnHelicalTrackLength).
"""

from pathlib import Path
import pandas as pd
import starfile


def split_sf(sf, threshold, antidote_label_field_name):
    sf_above = sf.copy()
    sf_below = sf.copy()

    sf_above["particles"] = sf_above["particles"][sf_above["particles"][antidote_label_field_name] >= threshold]

    sf_below["particles"] = sf_below["particles"][sf_below["particles"][antidote_label_field_name] < threshold]

    return sf_above, sf_below


def run(input_path: Path, output_dir: Path, threshold: float, antidote_label_field_name: str, overwrite: bool) -> None:
    sf = starfile.read(input_path)

    if (
        not pd.api.types.is_float_dtype(sf["particles"][antidote_label_field_name])
        or sf["particles"][antidote_label_field_name].min() < 0
        or sf["particles"][antidote_label_field_name].max() > 1
    ):
        raise ValueError(
            f"Field {antidote_label_field_name} does not contain valid ANTIDOTE labels (values between 0 and 1). Please ensure that the correct field is specified."
        )

    sf_above, sf_below = split_sf(sf, threshold, antidote_label_field_name)

    # set these paths explicitly so we can put them inside pre-Python 3.12 f-strings
    threshold_string = str(threshold).replace(".", "_")
    output_path_above = output_dir / (input_path.stem + f"_above_{threshold_string}.star")
    output_path_below = output_dir / (input_path.stem + f"_below_{threshold_string}.star")

    print(f"Writing {output_path_above}...")
    starfile.write(
        sf_above,
        output_path_above,
        overwrite=overwrite,
    )

    print(f"Writing {output_path_below}...")
    starfile.write(
        sf_below,
        output_path_below,
        overwrite=overwrite,
    )

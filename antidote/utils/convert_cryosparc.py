"""
Create .star file and necessary symlinks from a CryoSPARC job (NU-Refine, Ab-intio, Extract, Select 2D) in preparation for RELION 3D Classification. 
Ab-initio jobs are supported, but all particles (used, unused, regardless of class) will be included in the .star file.
Logic to support Hetero Refinement is included, but because of the nature of the multiple classes, it is not officially supported. 
"""

# TODO: Write end to end tests for all the different convert-cryosparc cases

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess
import starfile
import os

logger = logging.getLogger(__name__)


def run_csparc2star_and_symlink_mrcs(
    particles_file: Path,
    passthrough_particles_file: Path,
    star_file: Path,
    input_folder: Path,
    output_folder: Path,
    no_inverty: bool,
):
    args = ["csparc2star.py", particles_file, passthrough_particles_file, star_file]
    logger.info("Running csparc2star.py command: " + " ".join(map(str, args)))
    if not no_inverty:
        args.append("--inverty")
    exitcode = subprocess.call(args)

    if exitcode != 0:
        logger.error(
            f"csparc2star.py failed with exit code {exitcode}. Please ensure your input folder is a NU-Refine, Extract, or Select 2D job."
        )
        return

    # read the star file and get all the folders
    mrc_folders = set()
    star_data = starfile.read(star_file)
    for mrc_file in star_data["particles"]["rlnImageName"]:
        folder = os.path.dirname(mrc_file.split("@")[1])
        mrc_folders.add(folder)

    # rename all .mrc entries in the star file to .mrcs
    star_data["particles"]["rlnImageName"] = [
        mrc_file + "s" if mrc_file.endswith(".mrc") else mrc_file for mrc_file in star_data["particles"]["rlnImageName"]
    ]
    os.remove(star_file)
    starfile.write(star_data, star_file)

    # symlink all the contents of folder to the output folder (but not the folder itself)
    logger.info(f"Symlinking and renaming .mrc files to .mrcs in output folder {output_folder}")
    for folder in mrc_folders:
        folder_path: Path = Path(output_folder) / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)

        for file in Path(input_folder / ".." / folder).iterdir():
            # suffix should always be .mrcs
            filename = file.name
            if filename.endswith(".mrc"):
                filename = filename + "s"
            symlink_path = Path(folder_path / filename)
            if not symlink_path.exists():
                os.symlink(file, symlink_path)


def run(input_folder: str, output_folder: str, overwrite: bool, no_inverty: bool):
    """
    Convert a cryoSPARC job of .cs files to a .star file and symlink appropriate .mrc files.

    Args:
    -   input_folder (str): The path to the input cryoSPARC job directory. Should be a NU-Refine, Ab-Initio, Extract, or Select 2D job. Read access is required.
    -   output_folder (str): The path to output folder (where the `relion` command will be run). Read and write access is required.
    -   overwrite (bool): Flag for overwriting the output file.
    -   no_inverty (bool): Whether to invert the y-axis of the particles.
    """
    input_folder: Path = Path(input_folder)
    output_folder: Path = Path(output_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    if not output_folder.exists():
        logger.info(f"Output folder {output_folder} does not exist. Creating it...")
        output_folder.mkdir(parents=True, exist_ok=True)

    star_file = output_folder / f"{input_folder.name}.star"
    if star_file.exists():
        if overwrite:
            logger.info(f"Overwriting existing file {star_file}")
            os.remove(star_file)
        else:
            raise FileExistsError(f"Output file {star_file} already exists. Use --overwrite to overwrite it.")

    # discover particle files in cryoSPARC job directory for running csparc2star.py
    # an array and not just a single file to handle cases like Ab-Initio where there are multiple particle files
    passthrough_particles_files = []
    particles_files = []

    single_passthrough_particles_file = ""
    # for NU-Refine and Hetero Refinement jobs, we need to find the last iteration particles.cs file (JXXXX_), so this value will get
    # written to multiple times but only the last value will be used (which is the last iteration particles file
    # because we are sorting the files)
    single_particle_file = ""

    for file in sorted(input_folder.iterdir()):
        # if passthrough_particles_file gets assigned, don't even try to assign the same file to single_particle_file / particles_files
        # handles Hetero Refinement & Ab-Initio jobs
        if file.name.endswith("passthrough_particles_all_classes.cs") or file.name.endswith("passthrough_particles_unused.cs"):
            # don't add unused.cs if it's empty
            if file.name.endswith("unused.cs") and len(np.load(file)) == 0:
                continue
            passthrough_particles_files.append(file)
            continue
        # handles NU-Refine, Extract, and Select 2D jobs
        elif file.name.endswith("passthrough_particles.cs") or file.name.endswith("passthrough_particles_selected.cs"):
            single_passthrough_particles_file = file
            continue

        # handles Ab-Initio jobs
        if file.name.endswith("_final_particles.cs") or file.name.endswith("_final_particles_unused.cs"):
            # further filtering to not include by-class particle files
            if "_class_" not in file.name:
                # don't add unused.cs if it's empty
                if file.name.endswith("unused.cs") and len(np.load(file)) == 0:
                    continue
                particles_files.append(file)
        # handles NU-Refine, Hetero Refinement, Extract jobs, and Select 2D jobs
        elif file.name.endswith("particles.cs") or file.name == "particles_selected.cs":
            # further filtering to not include by-class particle files (for Hetero Refinement)
            if "_class_" in file.name and len(str(single_particle_file)) != 0 and "_class_" not in str(single_particle_file):
                continue
            single_particle_file = file

    if single_particle_file != "":
        assert (
            len(particles_files) == 0
        ), f"Found unexpected multiple particle files: {particles_files + [single_particle_file]}"
        particles_files.append(single_particle_file)

    if single_passthrough_particles_file != "":
        assert (
            len(passthrough_particles_files) == 0
        ), f"Found unexpected multiple passthrough particle files: {passthrough_particles_files + [single_passthrough_particles_file]}"
        passthrough_particles_files.append(single_passthrough_particles_file)

    if len(passthrough_particles_files) == 0 or len(particles_files) == 0:
        raise FileNotFoundError("Could not find the necessary files in the input folder.")

    logger.info(f"Discovered the following files for running csparc2star: {passthrough_particles_files} and {particles_files}")

    if len(particles_files) == 1:
        run_csparc2star_and_symlink_mrcs(
            particles_files[0],
            passthrough_particles_files[0],
            star_file,
            input_folder,
            output_folder,
            no_inverty,
        )
    # Handle multiple particle files (Ab-Initio)
    else:
        # TODO: ensure particles_files and passthrough_particles_files correspond to each other all the time
        for i, particles_file in enumerate(particles_files):
            run_csparc2star_and_symlink_mrcs(
                particles_file,
                passthrough_particles_files[i],
                output_folder / f"{input_folder.name}_{i}.star",
                input_folder,
                output_folder,
                no_inverty,
            )

        # TODO: Do further testing on this to ensure it works as expected
        # merge all the star files into a single one
        star_data = starfile.read(output_folder / f"{input_folder.name}_0.star")
        for i in range(1, len(particles_files)):
            star_data["particles"] = pd.concat(
                [star_data["particles"], starfile.read(output_folder / f"{input_folder.name}_{i}.star")["particles"]]
            )

        starfile.write(star_data, star_file)

        # remove the individual star files
        for i in range(len(particles_files)):
            os.remove(output_folder / f"{input_folder.name}_{i}.star")

    logger.info(f"Successfully converted cryoSPARC job to star file and symlinked mrcs at {output_folder}")

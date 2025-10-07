import concurrent.futures
import dill
import logging
import pandas as pd
from pathlib import Path
import re
import starfile
import sys
from typing import TypeVar

from antidote.utils import class3D_filter_features
from antidote.utils import class3D_normalize

RelionClass3DJob = TypeVar("RelionClass3DJob")
TrainingJob = TypeVar("TrainingJob", bound="RelionClass3DJob")

logger = logging.getLogger(__name__)


class RelionClass3DJob:
    """
    An object to represent a 3D Classification job from RELION.

    Args:
    -   input_path (pathlib.Path or str): The path to the RELION 3D Classification job. This path is generally of
                                          the format ..../Class3D/JobName/*.star. Attempts to resolve job name and
                                          starfiles are made if this structure is slightly different. Only the
                                          path up to JobName should be provided.

    -   max_workers (int or None): The number of threads to be used during starfile parsing, which is the most
                                   expensive step. The default value (None) uses all available threads. This
                                   argument is passed directly to the ProcessPoolExecutor object provided by
                                   concurrent.futures.

    -   label_func (optional callable or str): A lambda function or string that provides a function for assigning a
                                               label to the training data. This enables custom parsing of labeling
                                               logic in training sets. If a string is passed, the string is used to
                                               assign labels to True particles. If nothing is passed, particles
                                               containing the string "true" are labeled true.


    Returns:
    -   None. Creates a RelionClass3DJob object.

    Limitations:
    """

    def __init__(
        self,
        input_path,
        feature_index=None,
        features=None,
        labeling_func=None,
        max_workers=None,
        min_iteration=2,
        name=None,
        normalize=True,
        normalization_method=None,
        outlier_removal_method=None,
        remove_outliers=True,
        symmetry_order=1,  # no symmetry
        training=False,
    ) -> None:
        self.base_path: Path = None  # set from optimiser file
        self.data: pd.DataFrame = None  # final data for training or inference
        self.data_prenorm: pd.DataFrame = None  # engineered data before normalization
        self.data_raw: pd.DataFrame = None  # raw Class3D data
        self.e_step: int = None
        self._feature_index: str = feature_index  # index, default is rlnImageName
        self._features: list = features
        self._features_engineered: dict = None
        self.input_path: Path = Path(input_path).resolve()
        self.is_failure = False
        self.labeling_func: bytes = labeling_func  # needs to be deserialized with dill.load() before use
        self.min_iteration: int = min_iteration
        self.models_raw: pd.DataFrame = None  # parsed from model files
        self.name = input_path if name is None else name
        self.normalize: bool = normalize  # whether or not to apply normalization
        self._normalization_method: str = normalization_method
        self.num_classes: int = None  # set from model file
        self.num_iterations: int = None  # set from optimiser file
        self.num_particles: int = None  # set from parsed data
        (
            self.relion_version,
            self.relion_version_major,
        ) = self._set_relion_version_from_path()
        self.outliers: pd.DataFrame = None
        self._outlier_removal_method: str = outlier_removal_method
        self.remove_outliers: bool = remove_outliers  # whether or not to apply outlier removal
        self.symmetry: str = None  # parsed from job file
        self.symmetry_order: int = symmetry_order  # symmetry expansion value set by user
        self.tau_fudge_factor: float = None
        self.training: bool = training
        self.true_label_fraction: float = None
        self.working_path: Path = None

        # Construct RelionClass3DJob Object
        if any(self.input_path.glob("**/RELION_JOB_EXIT_FAILURE")):
            self.is_failure = True
            logger.warning(f"{input_path} contains a failed job")
        else:
            logger.info("Parsing optimiser files...")
            self._set_attributes_from_optimiser_file()
            logger.info("Parsing model files...")
            self._set_attributes_from_model_file()
            logger.info("Parsing job file...")
            self._set_attributes_from_job_file()
            self._set_attributes_from_model_file_classes(max_workers)
            self._set_attributes_from_data_file(max_workers)
            logger.info("Constructing RelionClass3DJob object...")
            self.run_filter_features()
            logger.info("Normalizing RelionClass3DJob data...")
            self.run_normalize()

    def __repr__(self) -> str:
        return repr(f"ANTIDOTE 3D Classification instance parsed from {self.input_path.name}")

    def _batch_executor_for_read_data_starfile(self, starfile_path: Path) -> pd.DataFrame:
        """
        Parses a data starfile and returns it as a pandas DataFrame with the iteration number using
        teamtomo's starfile module.

        Args:
        - starfile_path (pathlib.Path()): The full path to the data starfile.
        """
        starfile_od = starfile.read(starfile_path)
        starfile_df = starfile_od["particles"]
        starfile_df["Iteration"] = int(re.search(r"run_it(\d+)_.*\.star", str(starfile_path)).group(1))

        return starfile_df

    def _batch_executor_for_read_model_starfile(self, starfile_path: Path) -> pd.DataFrame:
        """
        Parses a model starfile and returns it as a pandas DataFrame with the iteration number using
        teamtomo's starfile module.

        Args:
        - starfile_path (pathlib.Path()): The full path to the data starfile.
        """
        starfile_od = starfile.read(starfile_path)
        starfile_df = starfile_od["model_classes"]

        starfile_df["Iteration"] = int(re.search(r"run_it(\d+)_.*\.star", str(starfile_path)).group(1))
        starfile_df["rlnClassNumber"] = starfile_df["rlnReferenceImage"].apply(
            lambda x: int(re.search(r"class(\d+)", x).group(1))
        )

        return starfile_df

    def _set_attributes_from_data_file(self, max_workers, data_file_suffix="data.star") -> None:
        """
        Parses data from data.star files and constructs a single dataframe representing these files.

        Args:
        -   path (pathlib.Path()): Path to 3D Class job provided by user.
        -   data_file_suffix (str): The suffix for data files generated by RELION.

        Returns:
        -   None. This method sets various attributes to this instance of a RelionClass3DJob object.
        """

        # get paths to all data files
        data_file_paths = [data_file_path for data_file_path in self.working_path.rglob("*" + data_file_suffix)]

        # Check that the input path actually contains the expected data
        assert data_file_paths, f"AssertionError: {self.input_path} does not contain any 3D Classification data."
        assert (
            len(data_file_paths) == self.num_iterations + 1
        ), f"AssertionError: {self.input_path} does not appear to contain {self.num_iterations} iterations of 3D Classification data."
        if self.min_iteration > self.num_iterations:
            logger.warning(
                "The minimum iteration cannot be larger than the number of iterations in the 3D classification. Using final iteration only."
            )
            self.min_iteration = self.num_iterations

        parsed_starfiles = []

        # parse starfiles in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._batch_executor_for_read_data_starfile, starfile_path): starfile_path
                for starfile_path in data_file_paths
            }

            # Initially print the progress bar
            print("Parsing data starfiles...")
            print_progress_bar(0, len(futures))

            num_parsed = 0
            for future in concurrent.futures.as_completed(futures):
                starfile_df = future.result()
                parsed_starfiles.append(starfile_df)

                num_parsed += 1
                print_progress_bar(num_parsed, len(futures))

            # Print a newline at the end to move to the next line
            print()

        self.data_raw = pd.concat(parsed_starfiles)
        self.num_particles = int(len(self.data_raw) / (self.num_iterations + 1))

    def _set_attributes_from_job_file(self, job_file_name="job.star") -> None:
        """
        Parses metadata from job.star based on the provided model_file_suffix.

        Args:
        -   path (pathlib.Path()): Path to 3D Class job provided by user.
        -   job_file_name (str): The filename for the job file generated by RELION. Data
                                 parsed from this file should not be iteration-dependent.

        Returns:
        -   None. This method sets various attributes to this instance of a
            RelionClass3DJob object.
        """
        job_file = None

        for path in self.input_path.rglob(job_file_name):
            job_file = starfile.read(path)
            break

        if job_file is None:
            logger.warning(f"No job file found for {self.input_path}")
            return

        job_sym = job_file["joboptions_values"].set_index("rlnJobOptionVariable")["rlnJobOptionValue"].get("sym_name")

        if job_sym is None:
            self.symmetry = "C1"
            logger.warning("Symmetry not found in job.star file")
        else:
            self.symmetry = job_sym

    def _set_attributes_from_model_file(self, model_file_suffix="it000_model.star") -> None:
        """
        Parses metadata from initial model starfile based on the provided model_file_suffix.

        Args:
        -   path (pathlib.Path()): Path to 3D Class job provided by user.
        -   model_file_suffix (str): The filename for the initial model file generated
                                     by RELION. Data parsed from this file should not be
                                     iteration-dependent.

        Returns:
        -   None. This method sets various attributes to this instance of a
            RelionClass3DJob object.
        """
        model_file = None

        for path in self.input_path.rglob("*" + model_file_suffix):
            model_file = starfile.read(path)
            break

        if model_file is None:
            logger.warning(f"No model file found for {self.input_path}")
            return

        self.tau_fudge_factor = model_file["model_general"]["rlnTau2FudgeFactor"]
        self.num_classes = model_file["model_general"]["rlnNrClasses"]

    def _set_attributes_from_model_file_classes(self, max_workers, model_file_suffix="model.star") -> None:
        """
        Parses data from model.star files and constructs a single dataframe representing these files.

        Args:
        -   path (pathlib.Path()): Path to 3D Class job provided by user.
        -   model_file_suffix (str): The suffix for model files generated by RELION.

        Returns:
        -   None. This method sets various attributes to this instance of a RelionClass3DJob object.
        """

        # get paths to all data files
        data_file_paths = [data_file_path for data_file_path in self.working_path.rglob("*" + model_file_suffix)]

        # Check that the input path actually contains the expected data
        assert data_file_paths, f"AssertionError: {self.input_path} does not contain any 3D Classification data."
        assert (
            len(data_file_paths) == self.num_iterations + 1
        ), f"AssertionError: {self.input_path} does not appear to contain {self.num_iterations} iterations of 3D Classification data."

        parsed_starfiles = []

        # parse starfiles in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._batch_executor_for_read_model_starfile, starfile_path): starfile_path
                for starfile_path in data_file_paths
            }

            # Initially print the progress bar
            print("Parsing model starfiles...")
            print_progress_bar(0, len(futures))

            num_parsed = 0
            for future in concurrent.futures.as_completed(futures):
                starfile_df = future.result()
                parsed_starfiles.append(starfile_df)

                num_parsed += 1
                print_progress_bar(num_parsed, len(futures))

            # Print a newline at the end to move to the next line
            print()

            self.models_raw = pd.concat(parsed_starfiles)

    def _set_attributes_from_optimiser_file(self, optimiser_file_suffix="it000_optimiser.star") -> None:
        """
        Parses metadata from initial optimiser starfile based on the provided optimiser_file_suffix. This
        function tries to set a working path (inferred from the location)

        Args:
        -   optimiser_file_suffix (str): The filename for the initial optimiser file generated
                                         by RELION. Data parsed from this file should not be
                                         iteration-dependent.

        Returns:
        -   None. This method sets various attributes to this instance of a
            RelionClass3DJob object.
        """

        # Find the optimiser file
        optimiser_file_path = next(self.input_path.rglob(f"*{optimiser_file_suffix}"), None)
        optimiser_file = starfile.read(optimiser_file_path) if optimiser_file_path else None

        if optimiser_file:
            self.working_path = optimiser_file_path.parent
            self.num_iterations = optimiser_file["rlnNumberOfIterations"]
            self.e_step = optimiser_file["rlnHighresLimitExpectation"]
        else:
            self.working_path = self.input_path
            logger.warning(f"No optimiser found for {self.input_path}")
            return

        # The remainder of this function tries to construct a base path from data in the optimiser file
        base_path = None
        rel_base_path = Path(optimiser_file["rlnOutputRootName"])

        # Pull the base path from the optimiser file if it is valid
        if rel_base_path.parent.exists():
            base_path = rel_base_path

        # Try to reconstruct the base path if a relative path is given in the optimiser file
        # Note that this path will only be valid if the Class3D dir also contains the source
        # data
        elif "Class3D" in rel_base_path.parts and "Class3D" in self.input_path.parts:
            path_from_root = self.input_path.parents[len(self.input_path.parents) - self.input_path.parts.index("Class3D") - 1]
            path_to_base = Path(*rel_base_path.parts[rel_base_path.parts.index("Class3D") + 1 :])
            base_path = path_from_root / path_to_base

        # If the synthetic base_path doesn't actually exist then don't set it as an attribute
        if base_path and base_path.parent.exists():
            self.base_path = base_path
        else:
            self.base_path = self.input_path

    def _set_relion_version_from_path(self, run_file="run.out") -> None:
        """
        Parses the default `run.out` logfile and pulls the RELION version.

        Args:
        -   path (pathlib.Path()): Path to 3D Class job provided by user.

        Returns:
        -   version (str): The full RELION version, stripped from run.out
        -   major_version (int): The RELION major version, as an int.
        """
        run_file_path = self.input_path / run_file
        version = "Version unknown"
        major_version = 0

        if run_file_path.exists():
            with open(run_file_path, "r") as file:
                for line in file:
                    if "RELION version" in line:
                        version = line.strip().lstrip("RELION version: ")
                        major_version_pattern = re.compile(r"(\d+)\.\d+")
                        match = major_version_pattern.search(version)
                        if match:
                            major_version = int(match.group(1))
                            break
                        else:
                            logger.warning("RELION version format is unrecognized")
                            break
                else:
                    logger.warning("RELION version not found in run.out")
        else:
            logger.warning(f"No run.out file found for {self.input_path}")

        return version, major_version

    @classmethod
    def as_training(cls, input_path, labeling_func="true", **kwargs) -> TrainingJob:
        """
        Denotes that this RelionClass3DJob will be used for training a new Antidote model,
        and exposes access to a few training-specific Class attributes.
        """
        if callable(labeling_func):
            pass
        elif isinstance(labeling_func, str):
            s = labeling_func  # need to redefine this to keep lambda logic
            labeling_func = lambda x: f"{s}" in x
        else:
            assert (
                False
            ), 'The labeling function is not valid. Provide either a callable function or a string to match to "true" particles.'

        # Pickle can't serialize lambda functions
        labeling_func = dill.dumps(labeling_func)

        return cls(input_path, training=True, labeling_func=labeling_func, **kwargs)

    @property
    def feature_index(self) -> str:
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: str) -> None:
        """
        Perform some basic type checking and rerun the feature filtering if
        a new feature_index is specified.
        """
        print("Feature index reset.")
        if isinstance(feature_index, str):
            feature_index = [feature_index]
        if not isinstance(feature_index, list) or not all(isinstance(item, str) for item in feature_index):
            raise TypeError("features/feature_index must be a list of strings")

        # The feature_index should not also be in the feature list
        if self._features and feature_index in self._features:
            self._features.remove(feature_index)

        self._feature_index = feature_index
        class3D_filter_features.run(self)
        class3D_normalize.run(self)

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, feature_list: list) -> None:
        print("Features reset.")
        # Make sure that the feature list is a list of strings
        if not isinstance(feature_list, list) or not all(isinstance(item, str) for item in feature_list):
            raise TypeError("features must be a list of strings.")

        if "Iteration" not in feature_list:
            logger.warning(
                "Removing Iterations from the feature list can break some "
                + "feature engineering and normalization approaches–proceed "
                + "with caution."
            )

        self._features = feature_list
        class3D_filter_features.run(self)
        class3D_normalize.run(self)

    @property
    def features_engineered(self) -> dict:
        return self._features_engineered

    @features_engineered.setter
    def features_engineered(self, features_engineered: dict) -> None:
        print("Engineered features reset.")
        if not isinstance(features_engineered, dict):
            raise TypeError("features_engineered must be a dict of feature_type:feature_name pairs")
        self._features_engineered = features_engineered
        class3D_filter_features.run(self)
        class3D_normalize.run(self)

    @property
    def normalization_method(self) -> str:
        return self._normalization_method

    @normalization_method.setter
    def normalization_method(self, method: str) -> None:
        print("Normalization method reset.")

        # Make sure that the method is valid
        if method not in ["Mean", "MinMax", "Robust", "Standard"]:
            raise ValueError("Normalization method must be 'Mean, 'MinMax', 'Robust, or 'Standard'.")

        self._normalization_method = method
        class3D_filter_features.run(self)
        class3D_normalize.run(self)

    @property
    def outlier_removal_method(self) -> str:
        return self._outlier_removal_method

    @outlier_removal_method.setter
    def outlier_removal_method(self, method: str) -> None:
        print("Outlier removal method reset.")

        # Make sure that the method is valid
        if method not in ["StdDev", "IsolationForest"]:
            raise ValueError("Normalization method must be 'StdDev' or 'IsolationForest'.")

        self._outlier_removal_method = method
        class3D_filter_features.run(self)
        class3D_normalize.run(self)

    def run_filter_features(self) -> RelionClass3DJob:
        """
        filter_features placeholder.
        """
        class3D_filter_features.run(self)

        return self

    def run_normalize(self) -> RelionClass3DJob:
        """
        normalize placeholder.
        """
        class3D_normalize.run(self)

        return self

    def reset(self) -> None:
        """
        Restore dynamic class properties to their defaults.
        """
        self.data = None
        self.outliers = None
        self._feature_index = None
        self._features = None
        self._features_engineered = None
        self.min_iteration = 2
        # self.normalize = True
        self._outlier_removal_method = None
        self._normalization_method = None
        # self.remove_outliers = True

        class3D_filter_features.run(self)
        class3D_normalize.run(self)


def print_progress_bar(iteration, total, width=30) -> None:
    """
    Mimics the TensorFlow progress bar style with no dependencies. Inspired by
    SO Question 3002085
    """
    percent = iteration / total
    if iteration == total:
        arrow = "=" * width
    else:
        arrow = "=" * int(round(percent * width) - 1) + ">"
    spaces = "." * (width - len(arrow))

    sys.stdout.write(f"\r{iteration}/{total} [{arrow + spaces}] - {percent*100:.2f}% complete")
    sys.stdout.flush()


def sanitize_input_path(input_path: Path) -> Path:
    """
    Checks that the input path is a RELION 3D Classification job and raises an error if it isn't.

    Returns:
    -   sanitized_path (Path): Path to a valid 3D Classification job.
    """

    input_path = Path(input_path).resolve()

    if any(input_path.glob("**/RELION_JOB_EXIT_FAILURE")):
        raise ValueError(f"'{input_path}' is a failed RELION job.")

    if input_path.is_file():
        input_path = input_path.parent

    if not any(input_path.glob("*.star")):
        raise ValueError(f"'{input_path}' does not contain .star files")

    return input_path

"""
Generate the dataframe containing features for training or inference. This is stored in the
RelionClass3DJob.data attribute. This module handles everything after data parsing and before
data normalization.
"""

import dill
import pandas as pd
import numpy as np
import pprint
import string
import random
import logging

from antidote.utils import handle_symmetry_expansion

logger = logging.getLogger(__name__)


def apply_freq_encoding(job, feature: str) -> pd.DataFrame:
    """
    Convert an input series containing arbitrary data (including ints/floats) and converts
    to a frequency-encoded series.

    Args:
    -   job (RelionClass3DJob): Passes through the parameters and data contained in the job, including number of iterations and the raw data
    -   feature (str): The name of the feature containing the data that will be mapped to source.
    """
    logger.debug(f"Starting frequency encoding for {feature}...")
    encoded_data = remove_and_pivot_on_iterations(
        job.data_raw[["rlnImageName", feature, "Iteration"]],
        job.feature_index,
        remove_initial_iterations=job.min_iteration,
    )

    for i in encoded_data:
        series = encoded_data[i]
        freq = series.groupby(series).size() / len(series)
        encoded_data[i] = series.map(freq)
        logger.debug(f"Frequency encoding completed for iteration {i[1]}")

    logger.debug(f"Frequency encoding completed for {feature}")
    return encoded_data


def apply_group(df: pd.DataFrame, length: int = 8):
    """
    Generates a random ID for this job that is assigned to a column in job.data. This is useful for cross-validation during training.
    """
    logger.debug(f"Generating random group ID of length {length}...")
    group = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
    df[("Group", "")] = group
    df[("Group", "")] = df[("Group", "")].astype("string")  # pandas str dtype
    logger.debug(f"Applied group ID {group}")

    return df


def apply_labels(df: pd.DataFrame, labeling_func: bytes) -> pd.DataFrame:
    """
    Apply labels using a lambda function
    """
    logger.debug("Applying labels...")

    labeling_func = dill.loads(labeling_func)

    if isinstance(df.columns, pd.MultiIndex):
        # Generate labels separately and concat them to prevent excessive memory frag
        labels = df.apply(lambda row: labeling_func(row.name), axis=1).to_frame()
        labels.columns = pd.MultiIndex.from_tuples([("Label", "")])
        df = pd.concat([df, labels], axis=1)
    else:
        labels = df.apply(lambda row: labeling_func(row.name))
        df = pd.concat([df, labels], axis=1)

    logger.debug(f"Applied labeling function using {str(labeling_func)}")

    return df


def apply_name(df, name: str):
    """
    Adds a "Name" field to job.data using a name specified by the user.
    """
    logger.debug(f"Adding 'Name' field with value: {name}")
    df[("Name", "")] = name
    df[("Name", "")] = df[("Name", "")].astype("string")  # pandas str dtype

    return df


def apply_sum_categorical_changes(job, feature: str) -> pd.DataFrame:
    """
    Convert a series of categorical data to a single feature reporting on the sum of their changes. For example,
    if the class number over four iterations of classification is 1, 1, 3, 1, this function should generate a single
    value of 2.

    This is a generalized replacement for the "jump count" engineered feature in the original antidote version.

    Args:
    -   job (RelionClass3DJob): Passes through the parameters and data contained in the job, including number of iterations and the raw data
    -   feature (str): The name of the feature containing the data that will be mapped to source.
    """
    logger.debug(f"Starting sum categorical changes for {feature}...")

    raw_data = remove_and_pivot_on_iterations(
        job.data_raw[["rlnImageName", feature, "Iteration"]],
        job.feature_index,
        remove_initial_iterations=job.min_iteration,
    )

    raw_data = raw_data[feature].diff(axis=1)
    raw_data = (raw_data > 0) | (raw_data < 0)

    count_col = raw_data.apply(np.sum, axis=1)
    result = pd.DataFrame(count_col, columns=pd.MultiIndex.from_tuples([(feature, "")]))

    logger.debug(f"Sum categorical changes completed for {feature}.")
    return result


def apply_sum_over_iteration_changes_encoding(job, feature: str) -> pd.DataFrame:
    """
    Reads over a feature column in job containing arbitrary data (including ints/floats) and sums the number of changes in the value between each iteration,
    Different from apply_sum_categorical_changes column where the total sum is stored in a series in the end, this method will track the changes between values
    on an iteration basis so the returned data frame contains a column for each iteration
    Commonly used for rlnClassNumber

    Args:
    -   job (RelionClass3DJob): Passes through the parameters and data contained in the job, including number of iterations and the raw data
    -   feature (str): The name of the feature containing the data that will be mapped to source.
    """
    logger.debug(f"Starting sum over iteration changes encoding for {feature}...")
    raw_data = remove_and_pivot_on_iterations(
        job.data_raw[["rlnImageName", feature, "Iteration"]],
        job.feature_index,
        remove_initial_iterations=job.min_iteration,
    )

    encoded_data = pd.DataFrame(index=raw_data.index)

    encoded_data[(feature, job.min_iteration)] = 0
    for i in range(job.min_iteration + 1, job.num_iterations + 1):
        compare_col = (raw_data[(feature, i)] != raw_data[(feature, i - 1)]).astype(int)
        encoded_data[(feature, i)] = compare_col + encoded_data[(feature, i - 1)]
        logger.debug(f"Processed iteration {i} for {feature}.")

    logger.debug(f"Sum over iteration changes encoding completed for {feature}.")
    return encoded_data


def apply_target_encoding(
    job,
    target_df: pd.DataFrame,
    source: str,
    target: str,
) -> pd.DataFrame:
    """
    Convert values in an input series to values in a target data structure. The target_df must have an
    Iteration feature.

    Args:
    -   job (RelionClass3DJob): Passes through the parameters and data contained in the job, including number of iterations and the raw data
    -   target_df (pd.DataFrame): The dataframe containing <source>, <target>, and "Iteration" features.
    -   source (str): The name of the feature containing the categorical data (to be mapped).
    -   target (str): The name of the feature containing the data that will be mapped to source.
    """
    logger.debug(f"Starting target encoding for {source} to {target}...")
    encoded_data = remove_and_pivot_on_iterations(
        job.data_raw[["rlnImageName", source, "Iteration"]],
        job.feature_index,
        remove_initial_iterations=job.min_iteration,
    )

    for i in encoded_data:
        source_data = encoded_data[i]
        iteration = source_data.name[1]
        t_df = target_df[target_df["Iteration"] == iteration]
        assert not t_df[source].duplicated().any(), f"Too many values in {source} to perform one-to-one mapping to {target}."

        encoded_series = source_data.map(t_df.set_index(source)[target])

        encoded_data[i] = encoded_series
        logger.debug(f"Target encoding completed for iteration {iteration}.")

    logger.debug(f"Target encoding completed for for {source} to {target}.")
    return encoded_data


def apply_zscore_encoding(job, feature: str) -> pd.DataFrame:
    """
    Convert an input series containing numerical data to a representation of those same
    data as Z-scores of the distribution of the input series.

    Args:
    -   job (RelionClass3DJob): Passes through the parameters and data contained in the job, including number of iterations and the raw data
    -   feature (str): The name of the feature containing the data that will be mapped to source.
    """
    logger.debug(f"Starting Z-score encoding for {feature}...")

    encoded_data = remove_and_pivot_on_iterations(
        job.data_raw[["rlnImageName", feature, "Iteration"]],
        job.feature_index,
        remove_initial_iterations=job.min_iteration,
    )

    for i in encoded_data:
        series = encoded_data[i]
        mean = series.mean()
        std = series.std()
        z_scores = series.map(lambda x: (x - mean) / std)

        encoded_data[i] = z_scores
        logger.debug(f"Z-score encoding completed for iteration {i[1]}.")

    logger.debug(f"Z-score encoding completed for {feature}.")
    return encoded_data


def calculate_true_label_fraction(df: pd.DataFrame) -> float:
    """
    Calculates the fraction of Trues in the "Labels" column in an input dataframe.
    """
    logger.debug("Calculating true label fraction...")
    if isinstance(df.columns, pd.MultiIndex):
        fraction_true = df[("Label", "")].mean() * 100
    else:
        fraction_true = df["Label"].mean() * 100

    logger.debug(f"True label fraction is {fraction_true:.2f}%")
    return fraction_true


def filter_features(df: pd.DataFrame, feature_index: list, features: list) -> pd.DataFrame:
    """
    Takes a dataframe and a list of column names and and returns a dataframe that only contains
    those features.
    """
    logger.debug(f"Extracting {', '.join(features)} from raw data...")
    return df[feature_index + features].copy()


def remove_and_pivot_on_iterations(
    df: pd.DataFrame,
    feature_index: list,
    remove_initial_iterations: int = 0,  # The default class definition for a RelionClass3DJob sets this to 2
    flatten: bool = False,
):
    """
    Takes a dataframe and pivots along an "Iteration" column to generate a MultiIndex dataframe.

    Args:
    -   df (pd.DataFrame): Input containing an Iteration feature.
    -   feature_index(list): The feature to be used as the index. This is usually rlnImageName. This is used right up
                             until training/inference.
    -   remove_initial_iterations: Initial iterations to drop. Early iterations don't appear to have useful information,
                                   it usually takes 2-3 iterations for the classification to begin to converge.
    -   flatten(bool): Flattens the MultiIndex header down to a single index for each feature_iteration. This will break
                       downstream feature engineering and shouldn't be used unless the downstream effects are characterized.
    """
    logger.debug(f"Removing first {remove_initial_iterations} iterations...")
    # Remove first n iterations. In RELION, it000 and it001 usually look odd.
    df = df[df["Iteration"] >= remove_initial_iterations]

    logger.debug("Pivoting dataframe to iteration-wise multiindex headers...")
    # Pivot to multi-index df
    df = df.pivot(index=feature_index[0], columns="Iteration")

    if flatten:
        logger.warning("Flattening multiindex to single header...")
        # Match RELION's iteration formatting
        df.columns = [f"{col[0]}_it{col[1]:03d}" for col in df.columns]

    return df


def run(job) -> pd.DataFrame:
    """
    Generate the dataframe containing features for training or inference. This is stored in the
    RelionClass3DJob.data attribute. When ingesting a Class3D dataset, the features attribute
    of the RelionClass3D object should not be populated–this function will provide a default.
    The user also has the option of passing an explicit feature list, either by calling this
    function or by populating the features attribute (and then calling this function).

    The run() function constructs the job.data object feature-by-feature. First, class-
    agnostic features are generated–unwanted data from the raw_data attribute are removed, feature
    columns for frequency encoding are created,

    Args:
    -   job (RelionClass3DJob): The object representing the Class3D dataset to be filtered.
    -   features (list): A list containing the features to be included in the final df.

    Returns:
    -   df (pd.DataFrame): A dataframe containing the process feature data.
    """
    logger.debug("Starting feature generation process")

    # Configure default job attributes if necessary
    if job.feature_index is None:
        logger.debug("No index specified. Setting default index...")
        job._feature_index = ["rlnImageName"]
    logger.debug(f"Index set to {job.feature_index}.")

    if job.features is None:
        logger.debug("No base features specified. Setting default features...")
        features = [
            "rlnLogLikeliContribution",
            "rlnMaxValueProbDistribution",
            "rlnNrOfSignificantSamples",
            "Iteration",
        ]
        job._features = features
    logger.debug(f"Base features set to {job.features}.")

    if job.features_engineered is None:
        logger.debug("No engineered features specified. Setting default engineered features...")
        features_engineered = {
            "FreqEnc": ["rlnClassNumber"],
            "ModelTargetEnc": [
                # ("rlnClassNumber", "rlnClassDistribution"), duplicate of the frequency encoding
                ("rlnClassNumber", "rlnEstimatedResolution"),
                ("rlnClassNumber", "rlnOverallFourierCompleteness"),
            ],
            "SumCategoricalChanges": ["rlnClassNumber"],
            "SumOverIterationChangesEnc": ["rlnClassNumber"],
            "ZScoreEnc": [
                "rlnLogLikeliContribution",
                "rlnMaxValueProbDistribution",
                "rlnNrOfSignificantSamples",
            ],
        }
        job._features_engineered = features_engineered
    logger.debug(f"Engineered features set to \n{pprint.pformat(job.features_engineered)}")

    # Update particle names for symmetry expansion
    logger.debug("Updating particle names for symmetry expansion...")
    job.data_raw = handle_symmetry_expansion.encode(job.data_raw, job.feature_index, job.symmetry_order, job.num_iterations)

    # remove unwanted features from the dataframe
    job.data = filter_features(job.data_raw, job.feature_index, job.features)
    # Break out to iterations.
    if "Iteration" in job.features:
        job.data = remove_and_pivot_on_iterations(
            job.data,
            job.feature_index,
            remove_initial_iterations=job.min_iteration,
        )
    else:
        logger.warning('"Iteration" is not in the feature list–skipping pivot.')

    # Storing columns in memory prevents DataFrame fragmentation
    new_columns = {}

    for encoding in job.features_engineered:
        for feature in job.features_engineered[encoding]:
            if isinstance(feature, tuple):
                source, target = feature[0], feature[1]
                f = f"{feature[0]}_to_{feature[1]}"
            else:
                f = feature

            match encoding:
                case "FreqEnc":
                    encoded_data = apply_freq_encoding(job, feature)
                    for i in range(job.min_iteration, job.num_iterations + 1):
                        new_columns[(str(f + "_" + encoding), i)] = encoded_data[(feature, i)]

                case "SumOverIterationChangesEnc":
                    encoded_data = apply_sum_over_iteration_changes_encoding(job, feature)
                    for i in range(job.min_iteration, job.num_iterations + 1):
                        new_columns[(str(f + "_" + encoding), i)] = encoded_data[(feature, i)]

                case "ModelTargetEnc":
                    encoded_data = apply_target_encoding(job, job.models_raw, source, target)
                    for i in range(job.min_iteration, job.num_iterations + 1):
                        new_columns[(str(f + "_" + encoding), i)] = encoded_data[(source, i)]

                case "SumCategoricalChanges":
                    encoded_data = apply_sum_categorical_changes(job, feature)
                    new_columns[(str(f + "_" + encoding), "")] = encoded_data[(feature, "")]

                case "ZScoreEnc":
                    encoded_data = apply_zscore_encoding(job, feature)
                    for i in range(job.min_iteration, job.num_iterations + 1):
                        new_columns[(str(f + "_" + encoding), i)] = encoded_data[(feature, i)]

                case _:
                    raise ValueError(f"{encoding} is not a valid encoding strategy.")

    new_data = pd.DataFrame(new_columns)
    job.data = pd.concat([job.data, new_data], axis=1)

    # Add training data labels
    if job.training is True:
        job.data = apply_labels(job.data, job.labeling_func)
        # job.data = apply_group(job.data)
        job.true_label_fraction = calculate_true_label_fraction(job.data)
        # if job.name:
        #     job.data = apply_name(job.data, job.name)
        # else:
        #     job.data = apply_name(job.data, job.input_path.name)

    # reorder the features of the data frame
    job.data = job.data.reindex(sorted(job.data.columns), axis=1)

    return job.data

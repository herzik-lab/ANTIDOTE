"""
Normalizes the data in a RelionClass3DJob Object
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class MeanScaler:
    """
    Minimal mean scaler in the style of sklearn.
    """

    def _fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.ptp = np.ptp(X, axis=0)
        return self

    def _transform(self, X):
        return (X - self.mean) / self.ptp

    def fit_transform(self, X):
        return self._fit(X)._transform(X)


def apply_normalization(
    df: pd.DataFrame,
    scaler: str = "Mean",
    columnwise: bool = False,
    training: bool = False,
) -> pd.DataFrame:
    """
    Apply sklearn scalers to data on a columnwise or feature-wise basis.
    """
    logger.info(f"Normalizing data with {scaler}Scaler {'columnwise' if columnwise else 'featurewise'}...")
    match scaler:
        case "Mean":
            s = MeanScaler()
        case "MinMax":
            s = MinMaxScaler(feature_range=(0, 1))
        case "Robust":
            s = RobustScaler()
        case "Standard":
            s = StandardScaler()
        case _:
            raise ValueError(f"{scaler} is not implemented as a valid scaling option in Antidote.")

    # columns for MultiIndex dfs are not updated when a df is sliced, which can produce unexpected behavior
    df.columns = df.columns.remove_unused_levels()
    preserve_index = df.index

    # handle non-numeric data in the input df
    if training:
        preserve_nonnumeric = df.select_dtypes(exclude=[np.number])
        df = df.drop(preserve_nonnumeric, axis=1)
        df.columns = df.columns.remove_unused_levels()

    if columnwise:
        scaled_data = s.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    else:
        df_scaled = pd.DataFrame(index=preserve_index, columns=df.columns)
        for col in df.columns.levels[0]:
            feature = df[col]
            flat = feature.values.flatten().reshape(-1, 1)
            scaled = s.fit_transform(flat)
            if isinstance(feature, pd.Series):
                feature_scaled = pd.DataFrame(scaled.reshape(feature.shape), index=preserve_index)
                # Pandas seems to treat a 1D df as a series here, so col needs to be reset
                col = (col, "")
            else:
                feature_scaled = pd.DataFrame(
                    scaled.reshape(feature.shape),
                    index=preserve_index,
                    columns=feature.columns,
                )

            df_scaled[col] = feature_scaled

    df_scaled.index = preserve_index
    if training:
        df_scaled = pd.concat([preserve_nonnumeric, df_scaled], axis=1)

    return df_scaled


def apply_outlier_removal(
    df,
    method: str = "StdDev",
    sigma: float = 3.0,
    iso_forest_true_label_fraction: float = 0.03,
    iso_forest_fraction: float = 0.02,
    training: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select and run outlier removal method. This function handles input data to only pass the appropriate data types (all numeric data with
    one label feature for two-pass outlier removal when relevant) and returns two distinct dataframes with inliers and outliers.
    """
    logger.info(f"Applying {method} outlier removal to data...")

    # separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number])
    non_numeric_columns = df.select_dtypes(exclude=[np.number])

    # reintroduce T/F labels for two-pass outlier removal
    if training:
        numeric_columns[("Label", "")] = non_numeric_columns.pop(("Label", ""))

    match method:
        case "StdDev":
            inliers_numeric, outliers_numeric = remove_outliers_via_stddev(numeric_columns, sigma=sigma, training=training)

        case "IsolationForest":
            inliers_numeric, outliers_numeric = remove_outliers_via_iforest(
                numeric_columns,
                iso_forest_true_label_fraction=iso_forest_true_label_fraction,
                iso_forest_fraction=iso_forest_fraction,
                training=training,
            )

    # merge inlier and outlier dfs with labels and drop the rows with no corresponding data
    inliers = pd.concat([non_numeric_columns, inliers_numeric], axis=1).dropna(axis=0)
    outliers = pd.concat([non_numeric_columns, outliers_numeric], axis=1).dropna(axis=0)

    return inliers, outliers


def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the following potential issues in the input data:
        - RELION "-nan" values that are occasionally introduced during target encoding to classifications in which one class has collapsed.
        - RELION "inf" values that are occasionally introduced during target encoding to classifications in which one class has collapsed.
        - General check for NaN values.
    """

    # Check for object dtype indicative of mixed data / "-nan"
    for col in df.columns:
        # This is inefficient but will rarely be run
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                # Calculate the minimum non-NaN value in the column
                if isinstance(df[df[col].name[0]], pd.Series):
                    min_value = df[df[col].name[0]].min()
                elif isinstance(df[df[col].name[0]], pd.DataFrame):
                    min_value = pd.to_numeric(df[df[col].name[0]].min(), errors="coerce")
                    min_value = min_value.min()

                # Replace NaN with the minimum value
                df[col].fillna(min_value, inplace=True)
            warnings.warn(f"Found non-numeric values in {col[0]}  - did 3D Classification converge correctly?")

    # Check for inf
    for feature in df.columns.get_level_values(0).unique():
        # np.inf only works with numeric data, so we need to check if dtypes are numeric to handle "name" and "group" fields in training data
        # The parent column df[feature] always has an "object" dtype (non-numeric), so we need to check all sub-columns
        if isinstance(df[feature], pd.DataFrame) and all(
            pd.api.types.is_numeric_dtype(df[feature][col]) for col in df[feature].columns
        ):
            numeric_check = True
        elif isinstance(df[feature], pd.Series) and pd.api.types.is_numeric_dtype(df[feature]):
            numeric_check = True
        else:
            numeric_check = False

        if numeric_check:
            if np.isinf(df[feature]).any().any():
                x_inf = df[feature].replace([np.inf, -np.inf], np.nan)
                min_value = x_inf.min() if isinstance(x_inf, pd.Series) else x_inf.min().min()
                max_value = x_inf.max() if isinstance(x_inf, pd.Series) else x_inf.max().max()

                # inplace = True here does not work likely due to MultiIndex
                mi = df[feature].replace([np.inf, -np.inf], [max_value, min_value])
                df[feature] = mi
                warnings.warn(f"Found infinite values in {feature} - did 3D Classification converge correctly?")

    if df.isnull().values.any():
        raise ValueError("Job data contains NaNs and cannot be normalized.")

    return df


def remove_outliers_via_iforest(
    df: pd.DataFrame,
    iso_forest_true_label_fraction: float = 0.03,
    iso_forest_fraction: float = 0.02,
    training: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes outliers from an input dataframe using sklearn's IsolationForest function.

    If this is a training job, this function will also apply an initial first-pass outlier removal step on data labeled
    "True". Note that the second pass of outlier detection is performed on all particles in the dataset, including the
    particles labeled as outliers in the initial pass on "True" particles only.

    The default "auto" approach to contamination provided by sklearn tends to remove a large amount of the input data. This
    behavior can be further explored in the future.
    """
    # Initial mask that will be updated with labels for outliers across each outlier detection run
    mask = pd.Series([True] * len(df), index=df.index)

    if training:
        # remove "False" values (and resulting NaNs) and label columns to prevent iforest from operating on these values
        df_true = df[df["Label"]].drop(columns=["Label"]).dropna()
        clf_true = IsolationForest(contamination=iso_forest_true_label_fraction)
        mask_true = clf_true.fit_predict(df_true)

        # Apply boolean filter to iforest's output and update mask with bitwise and
        condition = mask_true == 1
        mask[df_true.index] &= condition

    # Outlier detection on the entire dataset
    if training:
        df_full = df.drop(columns=["Label"])  # do not consider label columns
    else:
        df_full = df

    clf_full = IsolationForest(contamination=iso_forest_fraction)
    mask_full = clf_full.fit_predict(df_full)

    # Apply boolean filter to iforest's output and update mask with bitwise AND
    condition = mask_full == 1
    mask &= condition

    inliers = df[mask]
    outliers = df[~mask]

    return inliers, outliers


def remove_outliers_via_stddev(
    df: pd.DataFrame, sigma: float = 3.0, training: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes outliers from an input dataframe that lie above some standard deviation sigma.

    If this is a training job, an initial first pass of outlier detection will be performed on the particles labeled "True".
    The second pass will be performed on all particles. If the data are unlabeled (if this is an inference job), only a single
    pass of outlier detection will be performed.
    """
    # Initial "True" mask to be updated with or without the True-only classifier
    mask = pd.Series([True] * len(df), index=df.index)

    if training:
        # remove "False" values (and resulting NaNs) and label columns
        df_true = df[df["Label"]].drop(columns=["Label"]).dropna()
        std_true = df_true.std()
        mean_true = df_true.mean()
        mask_true = (abs(df_true - mean_true) <= sigma * std_true).all(axis=1)
        mask[df_true.index] &= mask_true

    # Outlier detection on the entire dataset
    if training:
        df_full = df.drop(columns=["Label"])  # do not consider label columns
    else:
        df_full = df

    std = df_full.std()
    mean = df_full.mean()
    mask_full = (abs(df_full - mean) <= sigma * std).all(axis=1)
    mask &= mask_full

    inliers = df[mask]
    outliers = df[~mask]

    return inliers, outliers


def run(job) -> pd.DataFrame:
    """
    Normalize the data in a RelionClass3DJob Object.
    """
    # sorting yields a marginal performance increase
    job.data = job.data.sort_index(axis=1)
    job.data_prenorm = job.data.copy()

    # Configure default outlier removal approach
    if job.remove_outliers and job.outlier_removal_method is None:
        job._outlier_removal_method = "IsolationForest"

    # Configure default normalization method if necessary
    if job.normalize and job.normalization_method is None:
        job._normalization_method = "Mean"

    job.data = clean_input_data(job.data)

    # Remove outliers and assign outliers to self.outliers
    if job.remove_outliers:
        (
            job.data,
            job.outliers,
        ) = apply_outlier_removal(
            job.data,
            method=job.outlier_removal_method,
            training=job.training,
        )

    # Normalize the data
    if job.normalize:
        job.data = apply_normalization(job.data, scaler=job.normalization_method, training=job.training)

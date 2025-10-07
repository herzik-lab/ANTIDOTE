import logging
import pandas as pd

logger = logging.getLogger(__name__)


def decode(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Removes the symmetry encoding suffix introduced by encode() by performing simple pattern matching.

    Limitations:
    This isn't designed to work on a column in a multiindex DataFrame (the symmetry expansion suffix
    should never be in the columns).
    """
    if isinstance(column_name, list):  # ANTIDOTE stores all features as a list
        column_name = column_name[0]

    pattern = r"_SE\d+"

    if column_name in df.columns:
        df.loc[:, column_name] = df[column_name].str.replace(pattern, "", regex=True)
    elif column_name == df.index.name:
        df.index = df.index.to_series().str.replace(pattern, "", regex=True)
    else:
        raise ValueError(f"{column_name} not found in target DataFrame.")

    return df


def encode(
    df: pd.DataFrame,
    column_name: str,
    symmetry_order: int,
    num_iterations: int = 1,
    strict: bool = True,
    encoding_suffix: str = "_SE",
) -> pd.DataFrame:
    """
    This function handles symmetry-encoded particles by appending a suffix to the name field early
    in the data processing pipeline. It seems to be the case that symmetry encoded particles are always
    grouped and in order when generated in RELION, so the suffix is applied with this assumption in mind.
    If the index group does not contain duplicates, this function should not affect data processing.

    Generally, it is expected that this function will operate on the 'rlnImageName' field, which is the
    default feature_index value. Since this field is used as the index for most of the data processing
    in ANTIDOTE, duplicate values (within each iteration) result in errors that should be fixable by
    appending a unique identifier (in this case "_SE" and an integer value) to each entry in the
    rlnImageName field.

    Args:
    -   df (pd.DataFrame): A pandas dataframe containing the column to be encoded.
    -   column_name (str): The name of the column containing duplicated entries.
    -   symmetry_order (int): The degree of symmetry, which is equal to the number of duplicate particles
                              that will be created by RELION. For example, C2 symmetry would be 2, D3
                              symmetry would be 12, and so on.
    -   strict (bool): Confirm that the symmetry order matches the actual number of duplicated rows in the
                       dataframe.

    Returns:
    -  df (pd.DataFrame): A pandas dataframe containing the encoded column

    Limitations:
    This function assumes that symmetry expansion introduces duplicated particles that are introduced to
    the original starfile in a consistent order.
    """
    if isinstance(column_name, list):  # ANTIDOTE stores all features as a list
        column_name = column_name[0]

    copy_groups = df.groupby(column_name)

    if strict:
        assert not any(
            copy_groups.size() > (symmetry_order * (num_iterations + 1))
        ), f"There are duplicate particle names in the input starfile. Has symmetry expansion been applied to this dataset? If so, run ANTIDOTE with --symmetry-expansion to set the symmetry order, which appears to be {int(copy_groups.size().mean()/num_iterations)} for this dataset."
        assert all(
            copy_groups.size() == (symmetry_order * (num_iterations + 1))
        ), f"Particles do not appear to have symmetry expansion of order {symmetry_order}"

    # Don't add suffix if there are no duplicates
    if symmetry_order > 1:
        logger.info("Handling symmetry expansion...")
        encoding_value = (copy_groups.cumcount() % symmetry_order + 1).astype(str)
        df.loc[:, column_name] = df[column_name] + encoding_suffix + encoding_value

    return df

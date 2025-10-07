import pandas as pd


def df_metadata(job) -> pd.DataFrame:
    """
    Collects meta data of RelionClass3D job.

    Args:
    - job (class3D_builder.RelionClass3DJob): job to collect data from

    Returns:
    - extracted cols: columns of data frame containing meta data for job.

    """
    total_true_particles = job.data[("Label", "")].sum()
    total_false_particles = len(job.data[("Label", "")]) - total_true_particles
    extract_cols = {
        "Name": [job.name],
        "Path": [job.input_path],
        "num_classes": [job.num_classes],
        "min_iteration": [job.min_iteration],
        "num_iterations": [job.num_iterations],
        "num_particles": [job.num_particles],
        "e_step": [job.e_step],
        "tau_fudge_factor": [job.tau_fudge_factor],
        "true_label_fraction": [job.true_label_fraction],
        "num_true_labels": [total_true_particles],
        "num_false_labels": [total_false_particles],
    }
    return pd.DataFrame.from_dict(extract_cols)


def df_mean(job) -> pd.DataFrame:
    """
    Collects means of features in job.data.
    Separates by True and False particles.

    Args:
    - job (class3D_builder.RelionClass3DJob): job to collect data from

    Returns:
    - mean_rows (pd.DataFrame): two rows of data frame(true & false particles)
      containing means for features in job.data.

    """

    true_df = job.data[job.data["Label"] == True]
    false_df = job.data[job.data["Label"] == False]
    df_categories = [true_df, false_df]

    mean_rows = pd.DataFrame()

    for df in df_categories:
        columns = df[[i for i in df.columns if df.dtypes[i] is float or i == ("Name", "")]].columns

        mean = df[columns].groupby("Name").mean()

        if mean_rows.empty:
            mean["Label"] = True
            mean_rows = mean.copy()
        else:
            mean["Label"] = False
            mean_rows = pd.concat([mean_rows, mean])

    return mean_rows


def df_std(job) -> pd.DataFrame:
    """
    Collects standard deviation values of features in job.data.
    Seperates by True and False particles.

    Args:
    - job (class3D_builder.RelionClass3DJob): job to collect data from

    Returns:
    - std_rows (pd.DataFrame): two rows of data frame(true & false particles)
      containing standard deviation values for features in job.data.

    """

    df_categories = []
    true_df = job.data[job.data["Label"] == True]
    false_df = job.data[job.data["Label"] == False]
    df_categories = [true_df, false_df]

    std_rows = pd.DataFrame()

    for df in df_categories:
        columns = df[[i for i in df.columns if isinstance(df.dtypes[i], float) or i == ("Name", "")]].columns

        std = df[columns].groupby("Name").std()

        if std_rows.empty:
            std["Label"] = True
            std_rows = std.copy()
        else:
            std["Label"] = False
            std_rows = pd.concat([std_rows, std])

    return std_rows

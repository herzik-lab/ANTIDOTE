import datetime
import holoviews as hv
from holoviews import opts
import KDEpy
import logging
import numpy as np
import pandas as pd
import panel as pn
from pathlib import Path
from panel.widgets.tables import Tabulator
import starfile
import warnings

from antidote.utils.class3D_builder import RelionClass3DJob

hv.extension("bokeh")
pn.extension("tabulator")
# The data snapshot tables stick out and are not full-width when closed
pn.extension(
    raw_css=[
        """
        :host {
            margin-left: 0px;
            margin-right: 0px;
        }

        .bk-panel-models-layout-Column {
            padding-left: 5px;
            padding-right: 5px;    
        }

        .bk-panel-models-layout-Card.accordion {
            width: 100vw;
        }
        """
    ]
)

logger = logging.getLogger(__name__)

"""
Build a report for an inference job.
"""


def plot_dataframe_snapshot(
    df: pd.DataFrame, selected_particles: pd.Index = None, key: str = None, suffix: str = "", sample_size: int = 20
) -> Tabulator:
    """
    Generate a Tabulator object for an input dataframe. This tends to slow down the report's loading speed,
    we can decide whether or not to keep this in later.
    """
    if selected_particles is None:
        df_sample = df.sample(n=sample_size)
    else:
        if key is None:
            df_sample = df[df.index.isin(selected_particles)].sample(n=sample_size)
        else:
            df_sample = df[df[key].isin(selected_particles)]
            df_sample = df_sample.groupby(key).apply(lambda x: x.sample(1)).reset_index(drop=True)

    if key is None:
        df_sample = df_sample.sort_index()
    else:
        df_sample = df_sample.sort_values(by=key)

    if not isinstance(df_sample.columns, pd.MultiIndex):
        return pn.widgets.Tabulator(
            df_sample, pagination="local", page_size=10, layout="fit_data_table", disabled=True, styles={"width": "90%"}
        ), (df_sample.index if key is None else df_sample[key])

    # subindices have to be strings and column names have to be unique
    df_sample.columns = pd.MultiIndex.from_tuples([(f"{x[0]}{suffix}", str(x[1])) for x in df_sample.columns])

    unique_columns_mapping = {}
    groups = {}
    for col in df_sample.columns:
        unique_col = "_".join(filter(None, col))  # Unique flattened name
        display_col = col[1] if col[1] else col[0]  # Readable name for display
        unique_columns_mapping[unique_col] = display_col
        top_level = col[0]
        if top_level in groups:
            groups[top_level].append(unique_col)
        else:
            groups[top_level] = [unique_col]

    # Apply unique columns to the DataFrame
    df_flat = df_sample.copy()
    df_flat.columns = [unique_col for unique_col in unique_columns_mapping.keys()]

    return pn.widgets.Tabulator(
        df_flat,
        # groups=groups, # performance is slightly better without grouping
        pagination="local",
        page_size=10,
        layout="fit_data_table",
        disabled=True,
    ), (df_sample.index if key is None else df_sample[key])


def plot_prediction_distribution_with_threshold(
    job: RelionClass3DJob, sf: dict, recommended_threshold: float, label_field_name: str
) -> hv.Overlay:
    """
    Plot the distribution of predictions from antidote and overlay the KDE and
    recommended threshold.

    # TODO
    -   Adjust spike height to highest value in KDE

    Args:
    Returns:
    -   prediction_plot_html (str): A string representation of the plot for direct
                                    insertion into the HTML report.
    """
    pred = np.array(sf["particles"][label_field_name])
    x, y = KDEpy.FFTKDE(kernel="gaussian", bw="ISJ").fit(pred).evaluate()
    f, e = np.histogram(pred, int(np.round(np.sqrt(pred.size))))

    spike = hv.Spikes(([recommended_threshold], 250), vdims="height", label="Threshold").opts(color="yellow", line_width=2)
    curve = hv.Curve((x, y * (f.max() / y.max())), label="KDE").opts(height=600, line_width=1, color="red")
    hist = hv.Histogram((f, e), label="Histogram").opts(tools=["hover"], responsive=True, height=600)

    overlay = (hist * curve * spike).opts(
        opts.Overlay(
            show_legend=True,
            title=f"ANTIDOTE inference score distribution",
            xlabel="Predicted Score",
            ylabel="Count",
        )  # Showing the legend for the overlay
    )

    return overlay


def plot_prediction_by_class(job: RelionClass3DJob, sf: dict, threshold: float, label_field_name: str):
    particles = sf["particles"][[label_field_name, "rlnClassNumber"]]
    particles["prediction"] = (particles[label_field_name] >= threshold).map(
        {True: "True Particles", False: "False Particles"}
    )
    particles = particles.groupby(["rlnClassNumber", "prediction"]).size().reset_index(name="count")
    particles.sort_values(by="prediction", ascending=False, inplace=True)
    bars = hv.Bars(particles, kdims=["rlnClassNumber", "prediction"])

    bars.opts(
        title="ANTIDOTE predictions by RELION 3D Class",
        ylabel="Count",
        xlabel="Relion 3D Class Number",
        height=600,
        responsive=True,
        multi_level=False,
        stacked=True,
        tools=["hover"],
    )

    return bars


def plot_violins(df: pd.DataFrame) -> hv.HoloMap:
    """
    Builds a subindex-specific set of violin plots for a multiindex df.
    """
    holomap = hv.HoloMap(kdims="Feature")
    ylim_dict = {}

    for feature in df.columns.levels[0]:
        feature_data = df[feature]

        # Apply the custom conversion function element-wise with logging
        if isinstance(feature_data, pd.Series):
            logger.debug(f"Skipping feature, does not contain values for each iteration: {feature}")
            continue

        feature_data = feature_data.applymap(lambda x: to_numeric_with_logging(x, feature))

        # Drop NaN values resulting from failed conversions
        feature_data = feature_data.dropna()

        # Calculate global min and max for each feature for framing violins
        min_val = feature_data.min().min() if isinstance(feature_data, pd.DataFrame) else feature_data.min()
        max_val = feature_data.max().max() if isinstance(feature_data, pd.DataFrame) else feature_data.max()
        ylim_dict[feature] = (min_val, max_val)

        sub_columns = feature_data.columns if isinstance(feature_data, pd.DataFrame) else [None]
        violin_plots = []

        for sub_column in sub_columns:
            data_column = feature_data if sub_column is None else feature_data[sub_column]
            if np.issubdtype(data_column.dtype, np.number):
                label = f"{sub_column}" if sub_column is not None else "All Iterations"
                violin_plot = hv.Violin(data_column, label=label).opts(
                    ylim=ylim_dict[feature],
                    xlabel=feature,
                    ylabel="Value",
                    responsive=True,
                    framewise=True,
                    shared_axes=False,
                    height=600,
                    cmap="Category20",
                )
                violin_plots.append(violin_plot)

        if violin_plots:
            combined_plot = hv.Overlay(violin_plots).opts(legend_position="right", framewise=True, shared_axes=False)
            holomap[feature] = combined_plot

    return holomap


def make_pdf(job: RelionClass3DJob, sf, threshold, label_field_name) -> None:
    """
    Build the PDF file that will represent the inference report. This can just be the HTML report
    with the matplotlib back end.
    """
    hv.extension("bokeh")
    prediction_plot = plot_prediction_distribution_with_threshold(job, sf, threshold, label_field_name)
    pass


def make_markdown(job: RelionClass3DJob, sf: dict, threshold: float, label_field_name: str, block: str) -> pn.pane.Markdown:
    """
    Panel report framework
    """
    match block:
        case "title":
            true_particles = sf["particles"][sf["particles"][label_field_name] >= threshold]

            md = pn.pane.Markdown(
                f"""

                # ANTIDOTE Inference Report for {job.input_path.name}
                Report generated from inference on {job.input_path} performed on {datetime.datetime.now()}.

                ## Class3D Parameters

                * RELION Version: {job.relion_version}
                * Particles: {job.num_particles}
                * Iterations: {job.num_iterations}
                * Classes: {job.num_classes}
                * Tau Fudge Factor: {job.tau_fudge_factor}

                ## Results Overview

                * Inference Particles: {(job.num_particles - len(job.outliers.index)) if isinstance(job.outliers, pd.DataFrame) else job.num_particles}
                * Inference Threshold: {threshold}
                * Inference predicted true particles: {len(true_particles)}
                * Inference predicted false particles: {job.num_particles - len(true_particles) - (len(job.outliers.index) if isinstance(job.outliers, pd.DataFrame) else 0)}
                * Outliers removed (not passed into inference, considered false particles): {len(job.outliers.index) if isinstance(job.outliers, pd.DataFrame) else 0}
                * Normalization Method: {job.normalization_method}
                * Raw Features: {", ".join(str(i).removeprefix('rln') for i in job.features)}

                """
            )

        case "violins_prenorm":
            md = pn.pane.Markdown(
                """
                ## Data Distributions

                ### Engineered Data

                The following distributions represent all native and engineered features that go into ANTIDOTE before normalization. These features are generated from 3D Classification metadata.
                """
            )

        case "violins_norm":
            md = pn.pane.Markdown(
                f"""
                ### Normalized Data

                The following distributions represent all native and engineered features that go into ANTIDOTE. These are the engineered features plotted above after {job.normalization_method} normalization.
                """
            )

        case "data_snapshot_inference":
            md = pn.pane.Markdown(
                f"""

                ## Data Snapshots

                The following tables provide snapshots into the distribution of the data used by ANTIDOTE. Each table contains a random sample of data from ANTIDOTE's data processing pipeline.

                ### Inference Data

                The following data are used by ANTIDOTE directly for inference. These include the raw and engineered features pulled from RELION 3D Classification metadata, All features are normalized with {job.normalization_method}.
                """
            )

        case "data_snapshot_prenorm":
            md = pn.pane.Markdown(
                f"""

                ### Engineered Data

                The following data are the ANTIDOTE inference data without the application of {job.normalization_method}Scaler normalization.
                """
            )

        case "data_snapshot_raw":
            md = pn.pane.Markdown(
                f"""

                ### Raw Data

                The following data are the ANTIDOTE inference data without the application of {job.normalization_method}Scaler normalization or feature engineering. These data are parsed directly from RELION 3D Classification data starfiles.
                """
            )

    return md


def to_numeric_with_logging(x: str, feature_name: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        logger.warning(f"Non-numeric value encountered in feature '{feature_name}': {x}")
        return np.nan


def save_report(
    report: pn.Column,
    output_path: Path,
    report_format: str,
    overwrite: bool,
    no_open_report: bool,
    embed: bool,
    progress: bool,
    report_prefix: str,
) -> None:
    match report_format:
        case "html":
            report_path = output_path / f"antidote_{report_prefix}_report.html"
            logger.info(f"Writing {report_prefix} inference report to {report_path.resolve()}...")
            report.save(report_path, title="ANTIDOTE Inference Report", overwrite=overwrite, embed=embed, progress=progress)
            if not no_open_report:
                import webbrowser

                webbrowser.open(report_path.resolve().as_uri())
        case "pdf":
            warnings.warn("PDF report has not been implemented yet")
        case _:
            warnings.warn(f"{report_format} is not a valid report format, please choose html or pdf")


def generate(
    job: RelionClass3DJob,  # RelionClass3DJob object
    sf: dict,  # full output starfile object from ANTIDOTE
    output_path: Path,
    threshold: float,
    label_field_name: str,
    report_format: str = "html",
    no_full_report: bool = False,
    no_open_report: bool = False,
    overwrite: bool = True,
) -> None:
    start_time = datetime.datetime.now()
    logger.info("Generating inference report...")
    valid_col = lambda col: isinstance(col, tuple) and (
        (isinstance(col[1], int) and col[1] in [2, 5, 10, 25]) or (col[1] == "")
    )
    job_data_reduced_df = job.data.loc[:, [col for col in job.data.columns if valid_col(col)]]
    job_data_prenorm_reduced_df = job.data_prenorm.loc[:, [col for col in job.data_prenorm.columns if valid_col(col)]]
    logger.debug(f"Dataframes reduce, time elapsed: {datetime.datetime.now() - start_time}")
    # Construct report components
    md_block_title = make_markdown(job, sf, threshold, label_field_name, block="title")
    logger.debug(f"Made markdown, time elapsed: {datetime.datetime.now() - start_time}")

    prediction_by_class_plot = plot_prediction_by_class(job, sf, threshold, label_field_name)
    logger.debug(f"Made prediction by class plot, time elapsed: {datetime.datetime.now() - start_time}")
    prediction_plot = plot_prediction_distribution_with_threshold(job, sf, threshold, label_field_name)
    logger.debug(f"Made prediction plot, time elapsed: {datetime.datetime.now() - start_time}")

    md_block_data_snapshot_inference = make_markdown(job, sf, threshold, label_field_name, block="data_snapshot_inference")
    logger.debug(f"Made markdown for data snapshot, time elapsed: {datetime.datetime.now() - start_time}")
    data_snapshot, selected_particles = plot_dataframe_snapshot(job_data_reduced_df)
    logger.debug(f"Made data snapshot, time elapsed: {datetime.datetime.now() - start_time}")

    md_block_data_snapshot_prenorm = make_markdown(job, sf, threshold, label_field_name, block="data_snapshot_prenorm")
    logger.debug(f"Made markdown for data snapshot (prenorm), time elapsed: {datetime.datetime.now() - start_time}")
    data_prenorm_snapshot, _ = plot_dataframe_snapshot(job_data_prenorm_reduced_df, selected_particles)
    logger.debug(f"Made data snapshot (prenorm), time elapsed: {datetime.datetime.now() - start_time}")

    md_block_data_snapshot_raw = make_markdown(job, sf, threshold, label_field_name, block="data_snapshot_raw")
    logger.debug(f"Made markdown for data snapshot (raw), time elapsed: {datetime.datetime.now() - start_time}")
    data_raw_snapshot, _ = plot_dataframe_snapshot(job.data_raw, selected_particles=selected_particles, key="rlnImageName")
    logger.debug(f"Made data snapshot (raw), time elapsed: {datetime.datetime.now() - start_time}")

    # initial report, faster to generate w/o violins
    summary_report = pn.Column(
        md_block_title,
        prediction_plot,
        prediction_by_class_plot,
        pn.Accordion(
            ("Inference Data Snapshot", pn.Column(md_block_data_snapshot_inference, data_snapshot)),
            ("Pre-Normalized Data Snapshot", pn.Column(md_block_data_snapshot_prenorm, data_prenorm_snapshot)),
            ("Raw Data Snapshot", pn.Column(md_block_data_snapshot_raw, data_raw_snapshot)),
        ),
        sizing_mode="stretch_width",
        styles={"padding-left": "5%", "padding-right": "5%"},
    )
    logger.debug(f"Made summary report, time elapsed: {datetime.datetime.now() - start_time}")
    save_report(
        summary_report,
        output_path,
        report_format,
        overwrite,
        no_open_report,
        embed=False,
        progress=False,
        report_prefix="summary",
    )

    if not no_full_report:
        logger.info(
            "Now generating full report, feel free to view the summary report while you wait (this might take a minute)..."
        )

        md_block_violins_prenorm = make_markdown(job, sf, threshold, label_field_name, block="violins_prenorm")
        logger.debug(f"Made markdown for violins, time elapsed: {datetime.datetime.now() - start_time}")
        md_block_violins_norm = make_markdown(job, sf, threshold, label_field_name, block="violins_norm")
        logger.debug(f"Made markdown for violins, time elapsed: {datetime.datetime.now() - start_time}")

        def create_violins(
            data,
            start_time,
        ):
            violins_holomap = plot_violins(data)
            logger.debug(f"Made violin holomap, time elapsed: {datetime.datetime.now() - start_time}")
            violins = pn.panel(violins_holomap, widget_location="top_right")
            logger.debug(f"Made violins, time elapsed: {datetime.datetime.now() - start_time}")
            return violins

        violins_prenorm = create_violins(job.data_prenorm, start_time)
        violins_norm = create_violins(job.data, start_time)

        full_report = pn.Column(
            md_block_title,
            prediction_plot,
            prediction_by_class_plot,
            pn.Accordion(
                (
                    "Data Distributions",
                    pn.Column(md_block_violins_prenorm, violins_prenorm, md_block_violins_norm, violins_norm),
                ),
                ("Inference Data Snapshot", pn.Column(md_block_data_snapshot_inference, data_snapshot)),
                ("Pre-Normalized Data Snapshot", pn.Column(md_block_data_snapshot_prenorm, data_prenorm_snapshot)),
                ("Raw Data Snapshot", pn.Column(md_block_data_snapshot_raw, data_raw_snapshot)),
            ),
            sizing_mode="stretch_width",
            styles={"padding-left": "5%", "padding-right": "5%"},
        )
        logger.debug(f"Made full report, time elapsed: {datetime.datetime.now() - start_time}")
        save_report(
            full_report,
            output_path,
            report_format,
            overwrite,
            no_open_report,
            embed=True,
            progress=False,
            report_prefix="full",
        )

    logger.debug(f"Inference report generation complete, time elapsed: {datetime.datetime.now() - start_time}")

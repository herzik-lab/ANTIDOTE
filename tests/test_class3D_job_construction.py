from pathlib import Path
import pytest

from antidote.utils import class3D_builder


@pytest.fixture(scope="module")
def job():
    """
    Checks Class3D parsing and label application.
    """
    input_path = Path(__file__).parent / "data/minimal_job"

    job = class3D_builder.RelionClass3DJob.as_training(
        input_path,
        labeling_func="2@calibration",
        remove_outliers=False,
        normalize=False,
    )

    return job


def test_starfile_accession(job) -> None:
    """
    Checks accession of metadata from data, model, and optimiser starfiles and population of RelionClass3DJob attributes
    """
    assert job.num_particles == 100
    assert job.num_iterations == 25
    assert job.num_classes == 2
    assert job.tau_fudge_factor == 4.0
    assert job.true_label_fraction == 10.0


def test_encoding_nans(job) -> None:
    # Check that encoding works
    assert not job.data.isnull().values.any()


def test_class3D_properties(job) -> None:
    """
    Test that job.features, job.feature_index, job.features_engineered reset correctly.
    """
    pass


@pytest.mark.parametrize("method, outliers", [("IsolationForest", 3), ("StdDev", 26)])
def test_outlier_detection(job, method, outliers) -> None:
    """
    Test outlier removal approaches.
    """
    job.remove_outliers = True
    job.outlier_removal_method = method

    assert len(job.outliers) == outliers

    job.remove_outliers = False
    job.reset()


@pytest.mark.parametrize("method", ["Mean", "MinMax", "Standard", "Robust"])
def test_normalization(job, method) -> None:
    """
    Test normalization approaches.
    """
    job.normalize = True
    job.normalization_method = method

    assert not job.data.isnull().values.any()

    job.normalize = False
    job.reset()

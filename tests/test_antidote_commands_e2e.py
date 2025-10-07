import pytest
import os
import subprocess
import starfile
import shutil

TEST_FOLDER = "./tests/data/minimal_job"
OUTPUT_FOLDER_PREFIX = "./tests/data/minimal_job_output_"
EXISTING_LABEL_TEST_FOLDER = "./tests/data/minimal_job_existing_label"
EXISTING_LABEL_OUTPUT_FOLDER_PREFIX = "./tests/data/minimal_job_existing_label_output_"


@pytest.fixture(autouse=True)
def setup_teardown(request):
    output_folder = OUTPUT_FOLDER_PREFIX + request.node.name
    existing_label_output_folder = EXISTING_LABEL_OUTPUT_FOLDER_PREFIX + request.node.name
    shutil.rmtree(output_folder, ignore_errors=True)
    shutil.rmtree(existing_label_output_folder, ignore_errors=True)
    yield
    shutil.rmtree(output_folder, ignore_errors=True)
    shutil.rmtree(existing_label_output_folder, ignore_errors=True)


def check_antidote_inference_run(output_folder, label_field_name="rlnHelicalTrackLength", no_report=False):
    full_sf = starfile.read(f"{output_folder}/full_predictions.star")
    full_sf_particle_rlnImageName = full_sf["particles"]["rlnImageName"]
    full_sf_particle_rlnImageName_set = set(full_sf_particle_rlnImageName)
    assert len(full_sf_particle_rlnImageName) == 100
    assert len(full_sf["particles"][label_field_name]) == 100
    assert full_sf["particles"][label_field_name].min() >= 0 and full_sf["particles"][label_field_name].max() <= 1

    true_sf = starfile.read(f"{output_folder}/true_predictions.star")
    true_sf_particle_rlnImageName = true_sf["particles"]["rlnImageName"]
    true_sf_particle_rlnImageName_set = set(true_sf_particle_rlnImageName)
    assert true_sf_particle_rlnImageName_set.issubset(full_sf_particle_rlnImageName_set)

    assert len(true_sf_particle_rlnImageName) < 100 and len(true_sf_particle_rlnImageName) > 0

    if not no_report:
        assert os.stat(f"{output_folder}/antidote_summary_report.html").st_size > 0


def test_basic_antidote_inference_full_run(request):
    output_folder = OUTPUT_FOLDER_PREFIX + request.node.name
    cmd = f"antidote inference --input {TEST_FOLDER} --output {output_folder} --no-open-report"
    subprocess.check_call(cmd, shell=True)

    check_antidote_inference_run(output_folder)


def test_basic_antidote_split_starfile_run(request):
    output_folder = OUTPUT_FOLDER_PREFIX + request.node.name
    cmd = f"antidote inference --input {TEST_FOLDER} --output {output_folder} --no-full-report"
    subprocess.check_call(cmd, shell=True)

    cmd = f"antidote tools split-starfile --input {output_folder}/full_predictions.star --output-directory {output_folder} --threshold 0.7"
    subprocess.check_call(cmd, shell=True)

    full_sf = starfile.read(f"{output_folder}/full_predictions.star")
    above_sf = starfile.read(f"{output_folder}/full_predictions_above_0_7.star")
    below_sf = starfile.read(f"{output_folder}/full_predictions_below_0_7.star")

    assert set(above_sf["particles"]["rlnImageName"]).isdisjoint(set(below_sf["particles"]["rlnImageName"]))
    assert set(above_sf["particles"]["rlnImageName"]).union(set(below_sf["particles"]["rlnImageName"])) == set(
        full_sf["particles"]["rlnImageName"]
    )
    assert above_sf["particles"]["rlnHelicalTrackLength"].min() >= 0.7
    assert below_sf["particles"]["rlnHelicalTrackLength"].max() < 0.7


def test_antidote_inference_overwrite(request):
    output_folder = OUTPUT_FOLDER_PREFIX + request.node.name
    cmd = f"antidote inference --input {TEST_FOLDER} --output {output_folder} --no-full-report"
    subprocess.check_call(cmd, shell=True)

    cmd_to_fail = f"antidote inference --input {TEST_FOLDER} --output {output_folder}"
    result = subprocess.run(cmd_to_fail.split(), capture_output=True, text=True)
    assert result.returncode == 1
    assert "already exists. Use --overwrite to overwrite." in result.stderr

    cmd = f"antidote inference --input {TEST_FOLDER} --output {output_folder} --no-full-report --overwrite"
    subprocess.check_call(cmd, shell=True)

    check_antidote_inference_run(output_folder, no_report=True)

    assert not os.path.exists(f"{output_folder}/full_predictions.star~")
    assert not os.path.exists(f"{output_folder}/true_predictions.star~")


def test_antidote_inference_existing_antidote_label_field_name(request):
    existing_label_output_folder = EXISTING_LABEL_OUTPUT_FOLDER_PREFIX + request.node.name
    cmd_to_fail = (
        f"antidote inference --input {EXISTING_LABEL_TEST_FOLDER} --output {existing_label_output_folder} --no-full-report"
    )
    result = subprocess.run(cmd_to_fail.split(), capture_output=True, text=True)
    assert result.returncode == 1
    assert "Please choose a different name with --antidote-label-field-name." in result.stderr

    cmd = f"antidote inference --input {EXISTING_LABEL_TEST_FOLDER} --output {existing_label_output_folder} --antidote-label-field-name rlnAntidotePrediction --no-full-report"
    subprocess.check_call(cmd, shell=True)

    check_antidote_inference_run(existing_label_output_folder, label_field_name="rlnAntidotePrediction", no_report=True)

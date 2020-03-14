import filecmp
import glob
import os
import shutil
import sys
from pathlib import Path
import json

import pytest
import pandas as pd
sys.path.insert(0, 'cognival')
sys.path.insert(0, 'cognival/significance_testing')

from significance_testing.statisticalTesting import extract_results
from testing_helpers import bonferroni_correction, test_significance as significance_test, save_scores

# Note: filecmp's diff_files attribute only lists differing files existing in both paths!

@pytest.fixture
def mapping_dict():
    with open('tests/reference/test_extract_results_results/mapping_1.json') as f:
        return json.load(f)


@pytest.fixture
def refdir():
    return Path('tests/reference/test_extract_results_results')


@pytest.fixture
def refdir_eye_tracking():
    return Path('tests/reference/test_extract_results_results/experiments/eye-tracking')


@pytest.fixture
def refdir_eeg():
    return Path('tests/reference/test_extract_results_results/experiments/eeg')


@pytest.fixture
def refdir_fmri():
    return Path('tests/reference/test_extract_results_results/experiments/fmri')


@pytest.fixture
def outdir():
    res_path = Path('tests/output/results/tmp')
    try:
        shutil.rmtree('tests/output/results')
    except FileNotFoundError:
        pass
    os.makedirs(res_path)
    os.makedirs(res_path / 'eeg' / 'eeg_zuco' / 'ALL_DIM' / 'glove.6B.50' / '1')
    os.makedirs(res_path / 'fmri'/ 'fmri_pereira-1' / 'ALL_DIM' / 'glove.6B.50' / '1')
    os.makedirs(res_path / 'eye-tracking' / 'eye-tracking_zuco' / 'nFixations' / 'glove.6B.50'/ '1')
    yield Path('tests/output/results')
    shutil.rmtree('tests/output/results')


@pytest.fixture
def result_dir():
    res_path = Path('tests/reference/test_extract_results_results/experiments')
    return res_path


def test_extract_results_eye_tracking(mapping_dict, refdir_eye_tracking, result_dir, outdir):
    extract_results(1,
                    'gaze',
                    "eye-tracking_zuco_nFixations_glove.6B.50",
                    mapping_dict,
                    result_dir,
                    outdir)
    assert not filecmp.dircmp(refdir_eye_tracking, outdir / 'tmp' / 'eye-tracking').diff_files


def test_extract_results_eeg(mapping_dict, refdir_eeg, result_dir, outdir):
    extract_results(1,
                    'eeg',
                    'eeg_zuco_ALL_DIM_glove.6B.50',
                    mapping_dict,
                    result_dir,
                    outdir)
    assert not filecmp.dircmp(refdir_eeg, outdir / 'tmp' / 'eeg').diff_files


def test_extract_results_fmri(mapping_dict, refdir_fmri, result_dir, outdir):
    extract_results(1,
                    'fmri',
                    'fmri_pereira-1_ALL_DIM_glove.6B.50',
                    mapping_dict,
                    result_dir,
                    outdir)
    assert not filecmp.dircmp(refdir_fmri, outdir / 'tmp' / 'fmri').diff_files


def test_informal_bonferroni_correction_1():
    alpha = bonferroni_correction(0.01, 1)
    assert alpha == 0.01


def test_informal_bonferroni_correction_5():
    alpha = bonferroni_correction(0.01, 5)
    assert alpha == 0.002


def test_informal_significance_test_eeg(refdir):
    baseline_file = refdir / 'sig_test_results' / 'eeg' / '1' /  'baseline_scores_eeg_zuco_ALL_DIM_glove.6B.50.txt'
    model_file = refdir / 'sig_test_results' / 'eeg' / '1' / 'embeddings_scores_eeg_zuco_ALL_DIM_glove.6B.50.txt'
    significant, pval, name = significance_test(str(baseline_file), str(model_file), 0.01, 'Wilcoxon')
    assert significant == True
    assert pval == pytest.approx(5.3231470579107915e-06 , rel=10e-50, abs=10e-50)
    assert name == 'eeg_zuco_ALL_DIM_glove.6B.50'


def test_informal_significance_test_fmri(refdir):
    baseline_file = refdir / 'sig_test_results' / 'fmri' / '1' /  'baseline_scores_fmri_pereira-1_ALL_DIM_glove.6B.50.txt'
    model_file = refdir / 'sig_test_results' / 'fmri' / '1' / 'embeddings_scores_fmri_pereira-1_ALL_DIM_glove.6B.50.txt'
    significant, pval, name = significance_test(str(baseline_file), str(model_file), 0.01, 'Wilcoxon')
    assert significant == False
    assert pval == pytest.approx(0.85824021397514, rel=10e-3, abs=10e-3)
    assert name == 'fmri_pereira-1_ALL_DIM_glove.6B.50'


def test_informal_significance_test_eye_tracking(refdir):
    baseline_file = refdir / 'sig_test_results' / 'eye-tracking' / '1'  / 'baseline_scores_eye-tracking_zuco_nFixations_glove.6B.50.txt'
    model_file = refdir / 'sig_test_results' / 'eye-tracking' / '1' / 'embeddings_scores_eye-tracking_zuco_nFixations_glove.6B.50.txt'
    significant, pval, name = significance_test(str(baseline_file), str(model_file), 0.01, 'Wilcoxon')
    assert significant == True
    assert pval == pytest.approx(3.6664582563418876e-06, rel=10e-10, abs=10e-10)
    assert name == 'eye-tracking_zuco_nFixations_glove.6B.50'
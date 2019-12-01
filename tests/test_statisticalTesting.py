import filecmp
import glob
import os
import shutil
import sys
from pathlib import Path

import pytest
import pandas as pd
sys.path.insert(0, '..')
sys.path.insert(0, '../significance_testing')

from significance_testing.statisticalTesting import extract_results
from testing_helpers import bonferroni_correction, test_significance as significance_test, save_scores

# Note: filecmp's diff_files attribute only lists differing files existing in both paths!

@pytest.fixture
def refdir_gaze():
    return Path('reference/sig_scores/gaze')


@pytest.fixture
def refdir_eeg():
    return Path('reference/sig_scores/eeg')


@pytest.fixture
def refdir_fmri():
    return Path('reference/sig_scores/fmri')


@pytest.fixture
def outdir():
    res_path = Path('output/results/tmp')
    try:
        shutil.rmtree('output/results')
    except FileNotFoundError:
        pass
    os.makedirs(res_path)
    os.mkdir(res_path / 'eeg')
    os.mkdir(res_path / 'fmri')
    os.mkdir(res_path / 'gaze')
    yield Path('output/results')
    shutil.rmtree('output/results')


@pytest.fixture
def result_dir():
    res_path = Path('output/sig_test_results')
    try:
        shutil.rmtree(res_path)
    except FileNotFoundError:
        pass
    os.makedirs(res_path)
    os.mkdir(res_path / 'eeg')
    os.mkdir(res_path / 'fmri')
    os.mkdir(res_path / 'gaze')
    yield res_path
    shutil.rmtree(res_path)


def test_extract_results_gaze(refdir_gaze, result_dir, outdir):
    extract_results('gaze',
                    'glove-50',
                    'random-embeddings-50',
                    result_dir / 'gaze',
                    outdir)
    assert not filecmp.dircmp(refdir_gaze, outdir / 'tmp' / 'gaze').diff_files


def test_extract_results_eeg(refdir_eeg, result_dir, outdir):
    extract_results('eeg',
                    'glove-50',
                    'random-embeddings-50',
                    result_dir / 'eeg',
                    outdir)
    assert not filecmp.dircmp(refdir_eeg, outdir / 'tmp' / 'eeg').diff_files


def test_extract_results_fmri(refdir_fmri, result_dir, outdir):
    extract_results('fmri',
                    'glove-50',
                    'random-embeddings-50',
                    result_dir / 'fmri',
                    outdir)
    assert not filecmp.dircmp(refdir_fmri, outdir / 'tmp' / 'fmri').diff_files


def test_informal_bonferroni_correction_eeg(refdir_eeg):
    hypotheses = [1 for filename in os.listdir(refdir_eeg) if 'embeddings_' in filename]
    alpha = bonferroni_correction(0.01, len(hypotheses))
    assert alpha == 0.01


def test_informal_bonferroni_correction_fmri(refdir_fmri):
    hypotheses = [1 for filename in os.listdir(refdir_fmri) if 'embeddings_' in filename]
    alpha = bonferroni_correction(0.01, len(hypotheses))
    assert alpha == 0.01


def test_informal_bonferroni_correction_gaze(refdir_gaze):
    hypotheses = [1 for filename in os.listdir(refdir_gaze) if 'embeddings_' in filename]
    alpha = bonferroni_correction(0.01, len(hypotheses))
    assert alpha == 0.002


def test_informal_significance_test_eeg(refdir_eeg):
    baseline_file = refdir_eeg / 'baseline_scores_zuco-eeg_ALL_DIM_glove-50_1.0.txt'
    model_file = refdir_eeg / 'embeddings_scores_zuco-eeg_ALL_DIM_glove-50_1.0.txt'
    output_str, pval, name = significance_test(str(baseline_file), str(model_file), 0.01)
    assert output_str == "Wilcoxon\n\nTest result is significant with p-value: 9.847862463890885e-47\n"
    assert pval == pytest.approx(9.847862463890885e-47, rel=10e-50, abs=10e-50)
    assert name == 'embeddings_scores_zuco-eeg_ALL_DIM_glove-50_1'


def test_informal_significance_test_fmri(refdir_fmri):
    baseline_file = refdir_fmri / 'baseline_scores_wehbe-100-1_ALL_DIM_glove-50_32.0.txt'
    model_file = refdir_fmri / 'embeddings_scores_wehbe-100-1_ALL_DIM_glove-50_32.0.txt'
    output_str, pval, name = significance_test(str(baseline_file), str(model_file), 0.01)
    assert output_str == "Wilcoxon\n\nTest result is not significant with p-value: 0.08923550925843837\n"
    assert pval == pytest.approx(0.08923550925843837, rel=10e-3, abs=10e-3)
    assert name == 'embeddings_scores_wehbe-100-1_ALL_DIM_glove-50_32'


def test_informal_significance_test_gaze(refdir_gaze):
    baseline_file = refdir_gaze / 'baseline_scores_zuco-gaze_FFD_glove-50_22.0.txt'
    model_file = refdir_gaze / 'embeddings_scores_zuco-gaze_FFD_glove-50_22.0.txt'
    output_str, pval, name = significance_test(str(baseline_file), str(model_file), 0.01)
    assert output_str == "Wilcoxon\n\nTest result is significant with p-value: 4.3008834197837275e-09\n"
    assert pval == pytest.approx(4.3008834197837275e-09, rel=10e-10, abs=10e-10)
    assert name == 'embeddings_scores_zuco-gaze_FFD_glove-50_22'
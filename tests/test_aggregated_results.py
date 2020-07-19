import filecmp
import sys
from pathlib import Path

import pytest
import pandas as pd
sys.path.insert(0, 'cognival')
sys.path.insert(0, 'cognival/significance_testing')

from significance_testing.aggregated_gaze_results import extract_results_gaze
from significance_testing.aggregated_eeg_results import extract_results as extract_results_eeg
from significance_testing.aggregated_fmri_results import extract_results as extract_results_fmri
from significance_testing.aggregate_significance import *

@pytest.fixture
def refdir():
    return Path('tests/reference/sig_scores')

@pytest.fixture
def refdir_eye_tracking():
    return Path('tests/reference/sig_scores/eye-tracking')

@pytest.fixture
def refdir_eeg():
    return Path('tests/reference/sig_scores/eeg')

@pytest.fixture
def refdir_fmri():
    return Path('tests/reference/sig_scores/fmri')
    

def test_extract_results_eye_tracking(refdir_eye_tracking):
    with open(refdir_eye_tracking / 'options1.0.json') as f:
        combinations = json.load(f)
    
    fold_errors, results_lists, avg_results = extract_results_gaze(combinations)
    assert avg_results['glove-50'] == pytest.approx(0.00668823196591811, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.008006483877673488, abs=1e-3, rel=1e-3)


def test_extract_results_eeg(refdir_eeg):
    with open(refdir_eeg / 'options1.0.json') as f:
        combinations = json.load(f)

    fold_errors, results_lists, avg_results = extract_results_eeg(combinations)
    assert avg_results['glove-50'] == pytest.approx(0.008593254540586731, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.010960347359505416, abs=1e-3, rel=1e-3)


def test_extract_results_fmri(refdir_fmri):
    with open(refdir_fmri / 'options1.0.json') as f:
        combinations = json.load(f)

    fold_erorrs, results_lists, avg_results = extract_results_fmri(combinations)
    assert avg_results['glove-50'] == pytest.approx(0.00668823196591811, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.003556954166535499, abs=1e-3, rel=1e-3)


def test_aggregate_signi_fmri(refdir):
    significance = aggregate_signi_fmri(refdir, 1, 'Wilcoxon', ['glove-50'])
    assert significance == {'glove-50': '0/1'}


def test_aggregate_signi_eeg(refdir):
    significance = aggregate_signi_eeg(refdir, 1, 'Wilcoxon', ['glove-50'])
    assert significance == {'glove-50': '1/1'}


def test_aggregate_signi_eye_tracking(refdir):
    significance = aggregate_signi_gaze(refdir, 1, 'Wilcoxon', ['glove-50'])
    assert significance == {'glove-50': '1/1'}

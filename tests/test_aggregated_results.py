import filecmp
import sys
from pathlib import Path

import pytest
import pandas as pd
sys.path.insert(0, '..')
sys.path.insert(0, '../significance_testing')

from significance_testing.aggregated_gaze_results import extract_results_gaze, extract_results_gaze_all
from significance_testing.aggregated_eeg_results import extract_results as extract_results_eeg
from significance_testing.aggregated_fmri_results import extract_results as extract_results_fmri
from significance_testing.aggregate_significance import *

@pytest.fixture
def refdir():
    return Path('reference/sig_scores')

@pytest.fixture
def refdir_gaze():
    return Path('reference/sig_scores/gaze')

@pytest.fixture
def refdir_eeg():
    return Path('reference/sig_scores/eeg')

@pytest.fixture
def refdir_fmri():
    return Path('reference/sig_scores/fmri')
    

def test_extract_results_gaze(refdir_gaze):
    avg_results = extract_results_gaze(refdir_gaze / 'options1.0.json',
                                       baselines=['random-embeddings-50'],
                                       embeddings=['glove-50'])
    assert avg_results['glove-50'] == pytest.approx(0.00668823196591811, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.008006483877673488, abs=1e-3, rel=1e-3)


#def test_extract_results_gaze_all():
#    avg_results = extract_results_gaze_all()


def test_extract_results_eeg(refdir_eeg):
    avg_results = extract_results_eeg(refdir_eeg / 'options1.0.json')
    assert avg_results['glove-50'] == pytest.approx(0.008593254540586731, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.010960347359505416, abs=1e-3, rel=1e-3)


def test_extract_results_fmri(refdir_fmri):
    avg_results = extract_results_fmri(refdir_fmri / 'options1.0.json')
    assert avg_results['glove-50'] == pytest.approx(0.00668823196591811, abs=1e-3, rel=1e-3)
    assert avg_results['random-embeddings-50'] == pytest.approx(0.003556954166535499, abs=1e-3, rel=1e-3)

def test_aggregate_signi_fmri_only_1000_voxels_True(refdir):
    significance = aggregate_signi_fmri(refdir, 'Wilcoxon', ['glove-50'], only_1000_voxels=True)
    assert significance == {'glove-50': '0/0'}

def test_aggregate_signi_fmri_only_1000_voxels_False(refdir):
    significance = aggregate_signi_fmri(refdir, 'Wilcoxon', ['glove-50'], only_1000_voxels=False)
    assert significance == {'glove-50': '0/1'}

def test_aggregate_signi_eeg(refdir):
    significance = aggregate_signi_eeg(refdir, 'Wilcoxon', ['glove-50'])
    assert significance == {'glove-50': '1/1'}

def test_aggregate_signi_gaze(refdir):
    significance = aggregate_signi_gaze(refdir, 'Wilcoxon', ['glove-50'])
    assert significance == {'glove-50': '1/1'}
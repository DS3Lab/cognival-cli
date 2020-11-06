import filecmp
import sys

from pathlib import Path, PosixPath

import numpy as np
import pytest
import pandas as pd
import json
import shutil
import pickle
sys.path.insert(0, 'cognival')

from lib_nubia.commands.process import *
from tests.process_fixtures import *

def test_filter_config(fixture_filter_config, results_filter_config):
    assert filter_config(**fixture_filter_config) == results_filter_config


def test_cumulate_random_emb_results(fixture_cumulate_random_embeddings, results_cumulate_random_embeddings):
    # Note: Does not test logging
    results = cumulate_random_emb_results(**fixture_cumulate_random_embeddings)
    for result, reference in zip(results, results_cumulate_random_embeddings):
        try:
            np.testing.assert_almost_equal(result.to_numpy(), reference.to_numpy(), decimal=3)
        except AttributeError:
            np.testing.assert_almost_equal(result, reference, decimal=3)


def test_write_random_emb_results(fixture_write_random_emb_results, results_write_random_emb_results):
    os.makedirs('tests/results', exist_ok=True)
    results = write_random_emb_results(**fixture_write_random_emb_results)
    del results['cum_rand_logging']['folds']
    np.testing.assert_almost_equal(results['cum_rand_word_error'][1:][:, 1:],
                                   results_write_random_emb_results['cum_rand_word_error'][1:][:, 1:],
                                   decimal=3)


def test_process_and_write_results(fixture_process_and_write_results, results_process_and_write_results):
    assert process_and_write_results(**fixture_process_and_write_results) == results_process_and_write_results


def test_insert_config_dict(fixture_insert_config_dict, results_insert_config_dict):
    assert insert_config_dict(**{**fixture_insert_config_dict}) == results_insert_config_dict


def test_resolve_cog_emb(fixture_resolve_cog_emb, results_resolve_cog_emb):
    assert resolve_cog_emb(**fixture_resolve_cog_emb) == results_resolve_cog_emb


# TODO: Requires mocking of configuration editor
#def test__edit_config(fixture_):
#    assert _edit_config()


def test_update_emb_config(fixture_update_emb_config, results_update_emb_config):
    assert update_emb_config(**fixture_update_emb_config) == results_update_emb_config


def test_generate_random_df(fixture_generate_random_df, results_generate_random_df):
    result = generate_random_df(**fixture_generate_random_df)
    result = result.set_index('word').to_numpy()
    np.testing.assert_almost_equal(result, results_generate_random_df)


def test_populate_cog_source(fixture_populate_cog_source, results_populate_cog_source):
    assert populate(**fixture_populate_cog_source) == results_populate_cog_source


def test_populate_embedding(fixture_populate_embedding, results_populate_embedding):
    assert populate(**fixture_populate_embedding) == results_populate_embedding


def test_populate_embedding_w_random(fixture_populate_embedding_w_random, results_populate_embedding_w_random):
    assert populate(**fixture_populate_embedding_w_random) == results_populate_embedding_w_random

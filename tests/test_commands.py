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

from lib_nubia.commands.commands import *
from tests.commands_fixtures import *

# TODO: Export config dynamically, test as such
#def test_run(fixture_run, results_run):
#    #This is a minimal test, only checking that a run finishes, i.e., run_id is incremented
#    assert run(**fixture_run) == results_run


#def test_list_configs(fixture_list_configs, results_list_configs):
#    assert list_configs(**fixture_list_configs) == results_list_configs


#def test_list_embeddings(fixture_list_embeddings, results_list_embeddings):
#    assert list_embeddings(**fixture_list_embeddings) == results_list_embeddings


#def test_list_cognitive_sources(results_import_cognitive_sources, results_list_cognitive_sources):
#    assert list_cognitive_sources(results_import_cognitive_sources) == results_list_cognitive_sources


def test_config_open(fixture_config_open, results_config_open):
    assert config_open(**fixture_config_open)[1] == results_config_open


#def test_config_show(fixture_config_show, results_config_show):
#    assert config_show(**fixture_config_show) == results_config_show


#def test_config_show_details(fixture_config_show_details, results_config_show_details):
#    assert config_show(**fixture_config_show_details) == results_config_show_details

#TODO: Cannot test the following without mocking

#def test_config_experiment(fixture_config_experiment, results_config_experiment):
#    assert config_experiment(**fixture_config_experiment) == results_config_experiment


#def test_config_experiment_w_random(fixture_config_experiment_w_random, results_config_experiment_w_random):
#    assert config_experiment(**fixture_config_experiment_w_random) == results_config_experiment_w_random

#TODO: Add tests for deletion

def test_config_delete_individual_experiment(fixture_config_delete_individual_experiment, results_config_delete_individual_experiment):
    assert config_delete(**fixture_config_delete_individual_experiment) == results_config_delete_individual_experiment


def test_config_delete_remove_cognitive_source(fixture_config_delete_remove_cognitive_source, results_config_delete_remove_cognitive_source):
    assert config_delete(**fixture_config_delete_remove_cognitive_source) == results_config_delete_remove_cognitive_source


def test_config_delete_remove_cognitive_source(fixture_config_delete_remove_cognitive_source, results_config_delete_remove_cognitive_source):
    assert config_delete(**fixture_config_delete_remove_cognitive_source) == results_config_delete_remove_cognitive_source


def test_config_delete_individual_experiment(fixture_config_delete_individual_experiment, results_config_delete_individual_experiment):
    assert config_delete(**fixture_config_delete_individual_experiment) == results_config_delete_individual_experiment


def test_config_delete_embedding_from_all_sources(fixture_config_delete_embedding_from_all_sources, results_config_delete_embedding_from_all_sources):
    assert config_delete(**fixture_config_delete_embedding_from_all_sources) == results_config_delete_embedding_from_all_sources


def test_remove_dangling_emb_random(fixture_remove_dangling_emb_random, results_remove_dangling_emb_random):
    assert remove_dangling_emb_random(**fixture_remove_dangling_emb_random) == results_remove_dangling_emb_random


def test_significance(fixture_significance, results_significance):
    results = list(significance('test', **fixture_significance))
    assert results[0] == '{}'
    results = json.loads(results[1])['hypotheses']
    results_significance = results_significance['hypotheses']
    for key in results.keys():
        try:
            assert pytest.approx(results[key]) == results_significance[key]
        except TypeError:
            assert results[key] == results_significance[key]


def test_aggregate(fixture_aggregate, results_aggregate):
    results = json.loads(next(aggregate('test', **fixture_aggregate)))
    results_aggregate = json.loads(results_aggregate)
    for key in results.keys():
        try:
            assert pytest.approx(results[key]) == results_aggregate[key]
        except TypeError:
            assert results[key] == results_aggregate[key]


def test_aggregate_large_example(fixture_aggregate_large_example, results_aggregate_large_example):
    results = json.loads(next(aggregate('eeg_reftest_prerelease', **fixture_aggregate_large_example)))
    for key in results.keys():
        try:
            assert pytest.approx(results[key]) == results_aggregate_large_example[key]
        except TypeError:
            assert results[key] == results_aggregate_large_example[key]


def test_update_vocabulary(fixture_update_vocabulary, results_update_vocabulary):
    assert update_vocabulary(**fixture_update_vocabulary) == results_update_vocabulary


# TODO: Requires mocking to test
#def test_import_cognitive_sources(fixture_import_cognitive_sources, results_import_cognitive_sources):
#    assert import_cognitive_sources(**fixture_import_cognitive_sources) == results_import_cognitive_sources


#def test_import_embeddings(fixture_import_embeddings, results_import_embeddings):
#    assert import_embeddings(**fixture_import_embeddings) == results_import_embeddings


def test_import_random_baselines(fixture_import_random_baselines, results_import_random_baselines):
    assert import_random_baselines(**fixture_import_random_baselines) == results_import_random_baselines

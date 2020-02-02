#!/usr/bin/env python3

# Derived from: TODO
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import asyncio
import collections
import csv
import copy
import itertools
import json
import gzip
import os
import pprint
import sys
import re
import requests
import signal
import socket
import shutil
import subprocess
import time
import typing
import zipfile

from datetime import datetime
from distutils.sysconfig import get_python_lib
from pathlib import Path
from subprocess import Popen, PIPE

import gdown
from joblib import Parallel, delayed
from natsort import natsorted
from nubia import command, argument, context
import numpy as np
import pandas as pd
import tabulate
import tableformatter as tform

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import input_dialog, button_dialog, yes_no_dialog, message_dialog, radiolist_dialog, ProgressBar
from pygments import highlight
from pygments.lexers import MarkdownLexer
from pygments.formatters import TerminalFormatter
from termcolor import cprint, colored

from cog_evaluate import run as run_serial
from cog_evaluate_parallel import run_parallel as run_parallel
from handlers.file_handler import write_results, write_options, update_version
from handlers.data_handler import chunk
from handlers.binary_to_text_conversion import bert_to_text, elmo_to_text

from utils import generate_df_with_header, word2vec_bin_to_txt

#sys.path.insert(0, 'significance_testing/')
from significance_testing.statisticalTesting import extract_results as st_extract_results
from significance_testing.aggregated_eeg_results import extract_results as agg_eeg_extr_results
from significance_testing.aggregated_fmri_results import extract_results as agg_fmri_extr_results
from significance_testing.aggregated_gaze_results import extract_results_gaze as agg_et_extr_results
from significance_testing.aggregate_significance import aggregate_signi_eeg, aggregate_signi_fmri
from significance_testing.aggregate_significance import aggregate_signi_gaze as aggregate_signi_et
from significance_testing.testing_helpers import bonferroni_correction, test_significance

from lib_nubia.prompt_toolkit_table import *
from lib_nubia.commands import messages

# Silence TF 2.0 deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local imports

from .form_editor import ConfigEditor, config_editor
from .process import (_filter_config,
                      cumulate_random_emb_results,
                      write_random_emb_results,
                      process_and_write_results,
                      insert_config_dict,
                      resolve_cog_emb,
                      _edit_config,
                      update_emb_config,
                      generate_random_df)
                      
from .utils import (tupleit,
                   _open_config,
                   _open_cog_config,
                   _check_cog_installed,
                   _check_emb_installed,
                   _save_cog_config,
                   _save_config,
                   DisplayablePath,
                   download_file,
                   AbortException,
                   chunked_list_concat_str,
                   field_concat,
                   chunks,
                   page_list)

from .templates import (WORD_EMB_CONFIG_FIELDS,
                        MAIN_CONFIG_TEMPLATE,
                        COGNITIVE_CONFIG_TEMPLATE,
                        EMBEDDING_PARAMET_TEMPLATE,
                        EMBEDDING_CONFIG_TEMPLATE)

# Pretty warnings
import warnings

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return colored('Warning: {}\n'.format(msg), 'yellow')

warnings.formatwarning = custom_formatwarning

NUM_BERT_WORKERS = 1
COGNIVAL_SOURCES_URL = 'https://drive.google.com/uc?id=1pWwIiCdB2snIkgJbD1knPQ6akTPW_kx0'

@command
class Run:
    """Execute experimental runs based on specified configuration and constraints
    [Sub-commands]
    - experiment: Run parallelized evaluation of single, selected or all
                  combinations of embeddings and cognitive sources.
    - experiment_serial: Non-parallelized variant, only for debugging purposes.
―
    """
    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command
    @argument("configuration", type=str, description="Name of configuration", positional=True)
    @argument("embedding", type=str, description="Single embedding")
    @argument("embeddings", type=list, description="List of embeddings")
    @argument("cognitive_source", type=str, description="Single cognitive source")
    @argument("cognitive_sources", type=list, description="List of cognitive sources")
    @argument("cognitive_feature", type=str, description="Single cognitive feature")
    @argument("cognitive_features", type=list, description="List of cognitive features")
    @argument("random_baseline", type=bool, description="Whether to compute random baseline(s) corresponding to specified embedding")
    def experiment_serial(self,
               configuration,
               embedding=None,
               embeddings=None,
               cognitive_source=None,
               cognitive_sources=None,
               cognitive_feature='ALL',
               cognitive_features=None,
               random_baseline=False):
        '''
        Run serial evaluation of single, selected or all combinations of embeddings and cognitive sources.
        Only exists for debugging purposes.
        '''
        cprint('Warning: experiment_serial is intended for debugging purposes only, use "run experiment p=1 ..." for single core processing', 'magenta')
        ctx = context.get_context()
        resources_path = ctx.resources_path

        parametrization = _filter_config(configuration,
                                         embedding,
                                         embeddings,
                                         cognitive_source,
                                         cognitive_sources,
                                         cognitive_feature,
                                         cognitive_features,
                                         random_baseline)
        
        if not parametrization:
            cprint("Aborting ...", "red")
            return
        
        config_dict, embeddings_list, emb_to_random_dict, cog_sources_list, cog_source_to_feature = parametrization

        for emb, cog in itertools.product(embeddings_list, cog_sources_list):
            for feat in cog_source_to_feature[cog]:
                cprint('Evaluating {}/{}/{} ...'.format(cog, emb, feat), 'green')
                start_time = datetime.now()
                truncate_first_line = config_dict['wordEmbConfig'][emb]['truncate_first_line']
                # Get random embeddings, check if not yet evaluated
                rand_emb = emb_to_random_dict.get(emb, None)
                results = run_serial(config_dict,
                                     emb,
                                     rand_emb,
                                     cog,
                                     feat,
                                     truncate_first_line)
                process_and_write_results(results, rand_emb, config_dict)
                time_taken = datetime.now() - start_time
                cprint('Finished after {} seconds'.format(time_taken), 'yellow')
                
        # Increment version
        config_dict['version'] += 1

        _save_config(config_dict, configuration, resources_path)
        cprint('All done.', 'green')
    
    @command
    @argument("configuration", type=str, description="Configuration for experimental runs", positional=True)
    @argument("processes", type=int, description="No. of CPU cores used for parallelization")
    @argument("embedding", type=str, description="Single embedding")
    @argument("embeddings", type=list, description="List of embeddings")
    @argument("cognitive_source", type=str, description="Single cognitive source")
    @argument("cognitive_sources", type=list, description="List of cognitive sources")
    @argument("cognitive_feature", type=str, description="Single cognitive feature")
    @argument("cognitive_features", type=list, description="List of cognitive features")
    @argument("random_baseline", type=bool, description="Compute random baseline(s) corresponding to specified embedding")
    def experiment(self,
                 configuration,
                 embedding=None,
                 embeddings=None,
                 cognitive_source=None,
                 cognitive_sources=None,
                 cognitive_feature='ALL',
                 cognitive_features=None,
                 processes=None,
                 random_baseline=True):
        '''
        Run parallelized evaluation of single, selected or all combinations of embeddings and cognitive sources.
        Only exists for debugging purposes.
        '''
        ctx = context.get_context()
        resources_path = ctx.resources_path

        parametrization = _filter_config(configuration,
                                         embedding,
                                         embeddings,
                                         cognitive_source,
                                         cognitive_sources,
                                         cognitive_feature,
                                         cognitive_features,
                                         random_baseline)
        
        if not parametrization:
            cprint("Aborting ...", "red")
            return
        
        config_dict, embeddings_list, emb_to_random_dict, cog_sources_list, cog_source_to_feature = parametrization


        # TODO: Check whether all installed!
        results_dict = run_parallel(config_dict,
                                    emb_to_random_dict,
                                    embeddings_list,
                                    cog_sources_list,
                                    cog_source_to_feature,
                                    cpu_count=processes)
        
        for modality, results in results_dict.items():
            run_stats = {}
            for id_, (result_proper, result_random, rand_emb, options) in enumerate(zip(results["proper"],
                                                                                 results["random"],
                                                                                 results["rand_embeddings"],
                                                                                 results["options"])):
                process_and_write_results(result_proper, result_random, rand_emb, config_dict, options, id_, run_stats)
            write_options(config_dict, modality, run_stats)

        # Increment version
        config_dict['version'] += 1

        _save_config(config_dict, configuration, resources_path)


@command
def list_configs():
    '''
    List available configurations with general parameters.
―
    '''
    ctx = context.get_context()
    resources_path = ctx.resources_path
    general_param_dicts = []
    for entry in os.scandir(resources_path):
        if entry.name.endswith('config.json'):
            with open(entry) as f:
                config = entry.name.split('_')[0]
                if not config == 'reference': 
                    config_dict = json.load(f)
                    general_params = {k:v for k, v in config_dict.items() if k not in ['cogDataConfig', 'wordEmbConfig', 'randEmbConfig', 'randEmbSetToParts']}
                    general_params['config'] = config
                    general_param_dicts.append(general_params)

    fgrid = tform.AlternatingRowGrid()
    cols = ['config'] + [k for k in general_param_dicts[0].keys() if not k == 'config']
    rows = [[colored(x['config'], attrs=['bold'], color='yellow')] + [v for k, v in x.items() if not k == 'config'] for x in general_param_dicts]
    formatted_table = tform.generate_table(rows=rows,
                                            columns=cols,
                                            grid_style=fgrid)
    cprint('List of available configurations. Note that the reference configuration is read-only and not listed.', 'green')
    print(formatted_table)

@command
@argument("configuration", type=str, description="Name of configuration", positional=True)
@argument("details", type=bool, description="Whether to show details for all cognitive sources. Ignored when cognitive_source is specified.")
@argument("cognitive_source", type=str, description="Cognitive source for which details should be shown")
@argument("hide_random", type=str, description="Hide random embeddings from Word embedding specifics")
def show_config(configuration, details=False, cognitive_source=None, hide_random=True):
    '''
    Display an overview for the given configuration.
―
    '''
    ctx = context.get_context()
    resources_path = ctx.resources_path

    config_dict = _open_config(configuration, resources_path, quiet=True, protect_reference=False)
    if not config_dict:
        return

    fgrid = tform.FancyGrid()
    altgrid = tform.AlternatingRowGrid()
    if not cognitive_source:
        cprint("Note: Use 'edit-config open {}' to edit the general properties of this configuration.".format(configuration), attrs=["bold"], color="green")
        general = [(k, v) for k, v in config_dict.items() if not k in ['cogDataConfig', 'wordEmbConfig', 'randEmbConfig', 'randEmbSetToParts']]
        print()
        cprint('General properties', attrs=['bold', 'reverse'], color='green')
        formatted_table = tform.generate_table(rows=[[x[1] for x in general]],
                                            columns=[x[0] for x in general],
                                            grid_style=fgrid,
                                            #row_tagger=row_stylist,
                                            transpose=False)
        print(formatted_table)
        print()
        experiment_rows = [chunked_list_concat_str(list(config_dict['cogDataConfig']), 4)]
        experiment_rows += [chunked_list_concat_str(list(['{} ({})'.format(k, v['random_embedding'] if 'random_embedding' in v and v['random_embedding'] else 'None') \
                                                            for k, v in config_dict['wordEmbConfig'].items()]), 2)]
        experiment_rows = [experiment_rows]

        cprint('Experiment properties', attrs=['bold', 'reverse'], color='cyan')
        formatted_table = tform.generate_table(rows=experiment_rows,
                                            columns=[colored('Cognitive sources', attrs=['bold']),
                                                     colored('Embeddings (Rand. emb.)', attrs=['bold'])],
                                            grid_style=fgrid,
                                            #row_tagger=row_stylist,
                                            transpose=True)

        print(formatted_table)
    
    if cognitive_source:
        cognitive_sources = [cognitive_source]
    elif details:
        cognitive_sources = list(config_dict['cogDataConfig'].keys())
    else:
        cognitive_sources = []

    if cognitive_sources:
        cprint("Note: Use 'edit-config experiment configuration={} cognitive-sources=[{}] single-edit=True' to edit the properties "
               "of the specified cognitive source(s) and associated embedding specifics.".format(configuration, ', '.join(cognitive_sources)), attrs=["bold"], color="green")

    for cognitive_source in cognitive_sources:
        cprint('{}\n'.format(cognitive_source), attrs=['bold', 'reverse'], color='yellow')
        try:
            cog_source_config_dict = config_dict['cogDataConfig'][cognitive_source]
        except KeyError:
            cprint('Cognitive source {} not registered in configuration {}, aborted ...'.format(cognitive_source, configuration), 'red')
            return

        cog_source_properties = [(k, field_concat(v)) for k, v in cog_source_config_dict.items() if k != 'wordEmbSpecifics']
        cprint('Cognitive source properties ({})'.format(cognitive_source), attrs=['bold', 'reverse'], color='green')
        formatted_table = tform.generate_table(rows=[[', '.join(x[1]) if isinstance(x[1], list) else x[1] for x in cog_source_properties]],
                                            columns=[x[0] for x in cog_source_properties],
                                            grid_style=fgrid,
                                            #row_tagger=row_stylist,
                                            transpose=False)
        print(formatted_table)
        print()
        cprint('Word embedding specifics', attrs=['bold', 'reverse'], color='cyan')
        word_emb_specifics = cog_source_config_dict['wordEmbSpecifics']
        if hide_random:
            word_emb_specifics = {k:v for k, v in word_emb_specifics.items() if not k.startswith('random')}

        df = pd.DataFrame.from_dict(word_emb_specifics).transpose()
        df.reset_index(inplace=True)
        df = df.rename(columns={"index": "word_embedding",
                                "activations":"activations (l.)",
                                "batch_size": "batch_size (l.)",
                                "epochs": "epochs (l.)",
                                "layers": "layers (nested l.)",
                                "validation_split": "validation_split (l.)"})
        df = df.applymap(field_concat)
        formatted_table = tform.generate_table(df,
                                            grid_style=altgrid,
                                            transpose=False) 
        print(formatted_table)


@command
class EditConfig:
    """Generate or edit configuration files for experimental combinations (cog. data - embedding type)
    [Sub-commands] 
    - open: Opens existing or creates new configuration edits its general properties.
    - populate: Populates specified configuration with empty templates or default
                configurations for some or all installed cognitive sources.
    - experiment: Edit configuration of single, multiple or all combinations of embeddings
                  and cognitive sources of specified configuration.
    - delete: Remove cognitive sources or experiments (cog.source - embedding combinations) from specified configuration
              or delete entire configuration.
―
    """

    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command
    @argument('configuration', type=str, description='Name of configuration', positional=True)
    @argument('overwrite', type=str, description='Name of configuration')
    def open(self, configuration, overwrite=False):
        '''
        Creates empty configuration file from template.
―
        '''
        ctx = context.get_context()
        resources_path = ctx.resources_path
        create = False

        if os.path.exists(resources_path / '{}_config.json'.format(configuration)):
            if overwrite:
                create = yes_no_dialog(title='Configuration {} already exists.'.format(configuration),
                            text='You have specified to overwrite the existing configuration. Are you sure?').run()
                if not create:
                    cprint('Aborted ...', 'red')
                    return
            else:
                create = False
        else:
            create = True

        if create:
            config_dict = copy.deepcopy(MAIN_CONFIG_TEMPLATE)
            _edit_config(config_dict, configuration)
        else:
            config_dict = _open_config(configuration, resources_path)
            if not config_dict:
                return
            _edit_config(config_dict, configuration)    
    
    @command
    @argument('configuration', type=str, description='Name of configuration file', positional=True)
    @argument('rand_embeddings', type=bool, description='Include random embeddings True/False')
    @argument('modalities', type=list, description="Groups of cog. sources to install.")
    @argument('cognitive_sources', type=list, description="Either list of cognitive sources or ['all'] (default).")
    @argument('embeddings', type=list, description="Either list of embeddings or ['all'] (default)")
    def populate(self, configuration, rand_embeddings, modalities=None, cognitive_sources=['all'], embeddings=['all'], mode="reference", quiet=False):
        '''
        Populates configuration with templates for some or all installed cognitive sources.
―
        '''
        # TODO: Finish; Show dialog for each source, buttons "Use defaults" (filling form), "Save & Next", "Abort"
        ctx = context.get_context()
        resources_path = ctx.resources_path
        embedding_registry = ctx.embedding_registry

        config_dict = _open_config(configuration, resources_path, quiet=quiet)
        if not config_dict:
            return
        cog_config_dict = _open_cog_config(resources_path)

        reference_dict = None

        try:
            cognitive_sources, embeddings = resolve_cog_emb(modalities,
                                                            cognitive_sources,
                                                            embeddings,
                                                            config_dict,
                                                            cog_config_dict,
                                                            embedding_registry,
                                                            scope="all")
        except AbortException:
            return

        if not config_dict:
            cprint('populate: Configuration {} does not yet exist!'.format(configuration), 'red')
            return

        if mode == 'reference':
            reference_path = resources_path / 'reference_config.json'
            with open(reference_path) as f:
                reference_dict = json.load(f)
        
        cog_sources_config = _open_cog_config(resources_path)
        if not cog_sources_config['cognival_installed']:
            cprint("Error: CogniVal cognitive vectors unavailable!", "red")
            raise AbortException
        else:
            cog_sources_config = cog_sources_config["sources"] 

        # Add cognitive source dicts
        for csource in cognitive_sources:
            if not csource in config_dict['cogDataConfig']:
                if mode == 'reference':
                    key = csource
                    try:
                        # Populate from reference config
                        config_dict['cogDataConfig'][key] = copy.deepcopy(reference_dict['cogDataConfig'][key])
                        config_dict['cogDataConfig'][key]['wordEmbSpecifics'] = {}
                    except KeyError:
                        cprint('Source {} not a CogniVal default source, looking up installed cog. sources ...'.format(csource), 'yellow')
                        try:
                            # Populate from installed cognitive sources
                            modality, csource_suff = csource.split('_')
                            csource_dict = cog_config_dict['sources'][modality][csource_suff]
                            cdc_dict = {"dataset": str(Path('cognitive_sources') / modality / csource_dict['file']),
                                        "modality": modality,
                                        "features": csource_dict['features'] if not csource_dict['features'] == 'single' else ['ALL_DIM'],
                                        "type": 'single_output' if csource_dict['dimensionality'] > 1 else 'multivariate_output'
                                        }
                            config_dict['cogDataConfig'][key] = copy.deepcopy(cdc_dict)
                            config_dict['cogDataConfig'][key]['wordEmbSpecifics'] = {}
                        except KeyError:
                            cprint('Cognitive source {} is unknown, adding empty ...'.format(csource), 'red')
                            config_dict['cogDataConfig'][key] = copy.deepcopy(COGNITIVE_CONFIG_TEMPLATE)
                            config_dict['cogDataConfig'][key]['wordEmbSpecifics'] = {}

                elif mode == 'empty':
                    config_dict['cogDataConfig'][key] = copy.deepcopy(COGNITIVE_CONFIG_TEMPLATE)
                    
        # Add embedding experiments dicts
        # Dynamic; check if installed
        for csource in config_dict['cogDataConfig']:
            for emb in embeddings:
                if not emb in config_dict['cogDataConfig'][csource]['wordEmbSpecifics']:
                    if not embedding_registry['proper'][emb]['installed']:
                        warnings.warn('Skipping {} ... (not installed)'.format(emb), UserWarning)
                        continue
                    
                    insert_config_dict(config_dict, reference_dict, mode, csource, emb, emb)
                    
                if rand_embeddings or config_dict['randEmbConfig']:
                    rand_emb = embedding_registry['proper'][emb]['random_embedding']
                    if rand_emb:
                        emb_part_list = ['{}_for_{}'.format(rand_emb_part, emb) for rand_emb_part in embedding_registry['random_multiseed'][rand_emb]['embedding_parts']]
                        for rand_emb_part in emb_part_list:
                            insert_config_dict(config_dict, reference_dict, mode, csource, rand_emb_part, emb)

        # Add embedding configurations dicts
        for emb in embeddings:
            emb_key = 'wordEmbConfig'
            emb_dict = copy.deepcopy(embedding_registry['proper'][emb])
            config_dict[emb_key][emb] = copy.deepcopy({k:v for k, v in emb_dict.items() if k in WORD_EMB_CONFIG_FIELDS})
            config_dict[emb_key][emb]['path'] = str(Path('embeddings') / embedding_registry['proper'][emb]['path'] / embedding_registry['proper'][emb]['embedding_file'])
            if rand_embeddings or config_dict['randEmbConfig']:
                config_dict[emb_key][emb]['random_embedding'] = '{}_for_{}'.format(config_dict[emb_key][emb]['random_embedding'], emb)
                rand_emb = embedding_registry['proper'][emb]['random_embedding']
                if rand_emb:
                    emb_dict = copy.deepcopy(embedding_registry['random_multiseed'][rand_emb])
                    emb_part_list = natsorted(list(embedding_registry['random_multiseed'][rand_emb]['embedding_parts']))
                    config_dict['randEmbSetToParts']['{}_for_{}'.format(rand_emb, emb)] = ['{}_for_{}'.format(rand_emb_part, emb) for rand_emb_part in emb_part_list]
                    for rand_emb_part in emb_part_list:
                        config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)] = copy.deepcopy({k:v for k, v in emb_dict.items() if k in WORD_EMB_CONFIG_FIELDS})
                        config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)]['path'] = str(Path('embeddings') / config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)]['path'] / '{}.txt'.format(rand_emb_part))
                else:
                    cprint('Embedding {} has no associated random embedding, skipping ...'.format(emb), 'yellow')

        _save_config(config_dict, configuration, resources_path, quiet=quiet)

    @command
    @argument('configuration', type=str, description='Name of configuration file', positional=True)
    @argument('modalities', type=list, description="Groups of cog. sources to install.")
    @argument('cognitive_sources', type=list, description="Either list of cognitive sources or ['all'] (default).")
    @argument('embeddings', type=list, description="Either list of embeddings or ['all'] (default)")
    @argument('rand_embeddings', type=bool, description='Include random embeddings True/False')
    @argument('single_edit', type=bool, description='Whether to edit embedding specifics one by one or all at once.')
    @argument('edit_cog_source_params', type=bool, description='Whether to edit parameters of the specified cognitive sources.')
    def experiment(self,
                   configuration,
                   rand_embeddings,
                   modalities=None,
                   cognitive_sources=['all'],
                   embeddings=['all'],
                   populate=True,
                   single_edit=False,
                   edit_cog_source_params=False):
        '''
        Edit configuration of single, multiple or all combinations of embeddings and cognitive sources.
        '''
        # TODO: Don't open configuration twice, pass dictionary to populate if some non-existent
        ctx = context.get_context()
        embedding_registry = ctx.embedding_registry
        resources_path = ctx.resources_path

        main_conf_dict = _open_config(configuration, resources_path)
        cog_data_config_dict = _open_cog_config(resources_path)

        if not main_conf_dict:
            return
        
        config_dicts = []

        edit_all_embeddings = embeddings[0] == 'all'

        try:
            cognitive_sources, embeddings = resolve_cog_emb(modalities,
                                                            cognitive_sources,
                                                            embeddings,
                                                            main_conf_dict,
                                                            cog_data_config_dict,
                                                            embedding_registry,
                                                            scope="all")
        except AbortException:
            return

        cog_data_config_dict = main_conf_dict['cogDataConfig']
        
        for csource in cognitive_sources:            
            if csource not in cog_data_config_dict:
                if populate:
                    cprint('Source {} not yet registered, creating ...'.format(csource), 'yellow')
                    try:
                        self.populate(configuration, rand_embeddings=rand_embeddings, cognitive_sources=[csource], embeddings=[], quiet=True)
                    except AbortException:
                        return
                    main_conf_dict = _open_config(configuration, resources_path, quiet=True)
                    cog_data_config_dict = main_conf_dict['cogDataConfig']
                else:
                    continue

            # If populate is False, edit only existing configurations vs. populating missing ones
            if edit_all_embeddings and not populate:
                embeddings = list(cog_data_config_dict[csource]["wordEmbSpecifics"].keys())

            if edit_cog_source_params:
                # Run config editor for cognitive source
                config_patch = config_editor("cognitive",
                                            cog_data_config_dict[csource],
                                            [],
                                            [csource],
                                            singleton_params=['dataset', 'type', 'modality'],
                                            skip_params=['wordEmbSpecifics'])

                if config_patch:
                    cog_data_config_dict[csource].update(config_patch)
                    _save_config(main_conf_dict, configuration, resources_path)
                else:
                    return

            for emb in embeddings:
                if not emb in cog_data_config_dict[csource]["wordEmbSpecifics"]:
                    cprint('Experiment {} / {} not yet registered, adding empty template ...'.format(csource, emb), 'yellow')
                    try:
                        self.populate(configuration, rand_embeddings=rand_embeddings, cognitive_sources=[csource], embeddings=[emb], quiet=True)
                    except AbortException:
                        return
                    main_conf_dict = _open_config(configuration, resources_path, quiet=True)
                    cog_data_config_dict = main_conf_dict['cogDataConfig']

                emb_config = cog_data_config_dict[csource]["wordEmbSpecifics"][emb]
                config_dicts.append(emb_config)
            
        if config_dicts:            
            if single_edit:
                for idx, (emb, cdict) in enumerate(zip(embeddings, config_dicts)):
                    config_template = copy.deepcopy(config_dicts[idx])
                    # Run editor for cognitive source/embedding experiments
                    config_patch = config_editor("embedding_exp",
                                                 config_template,
                                                 [emb],
                                                 cognitive_sources,
                                                 singleton_params=['cv_split', 'validation_split'])
                    if not config_patch:
                        return
                    else:
                        update_emb_config(emb, cdict, config_patch, rand_embeddings, main_conf_dict, embedding_registry)
                        _save_config(main_conf_dict, configuration, resources_path)
                        
            else:
                config_template = copy.deepcopy(config_dicts[0])
                # Generate template for multi-editing
                if len(config_dicts) > 1:
                    config_aggregated = collections.defaultdict(set)
                    for key in config_template:
                        for cdict in config_dicts:
                            config_aggregated[key].add(tupleit(cdict[key]))
                    
                    for key, values in config_aggregated.items():
                        if len(values) > 1:
                            config_template[key] = '<multiple values>'
                        else:
                            config_template[key] = values.pop()
                
                # Run editor for cognitive source/embedding experiments
                config_patch = config_editor("embedding_exp",
                                            config_template,
                                            embeddings,
                                            cognitive_sources,
                                            singleton_params=['cv_split', 'validation_split'])
                if not config_patch:
                    return
                else:
                    for emb, cdict in zip(embeddings, config_dicts):
                        update_emb_config(emb, cdict, config_patch, rand_embeddings, main_conf_dict, embedding_registry)

                _save_config(main_conf_dict, configuration, resources_path)

    @command
    @argument('configuration', type=str, description='Name of configuration file', positional=True)
    @argument('modalities', type=list, description="Groups of cog. sources to install.")
    @argument('cognitive_sources', type=list, description="Either list of cognitive sources or ['all'] (default).")
    @argument('embeddings', type=list, description="Either list of embeddings or ['all'] (default)")
    def delete(self, configuration, modalities=None, cognitive_sources=None, embeddings=None):        
        '''
        Remove cognitive sources or experiments (cog.source - embedding combinations) from specified configuration or
        delete entire configuration.
        '''
        ctx = context.get_context()
        embedding_registry = ctx.embedding_registry
        resources_path = ctx.resources_path

        main_conf_dict = _open_config(configuration, resources_path)
        cog_data_config_dict = _open_cog_config(resources_path)

        if not cognitive_sources and not modalities and not embeddings:
            delete_config = button_dialog(title='Deletion',
                                text='You have not specified cognitive sources and embeddings for removal. Do you wish to delete configuration "{}"?'.format(configuration),
                                buttons=[
                                        ('No', False),
                                        ('Yes', True),
                                        ]).run()
            if delete_config:
                os.remove(resources_path / '{}_config.json'.format(configuration))
                cprint('Deleting configuration "{}" ...'.format(configuration), 'yellow')
                return
            else:
                return
        
        elif not cognitive_sources and not modalities:
            all_sources = button_dialog(title='Deletion',
                                text='You have not specified cognitive sources or groups. Do you wish to remove the specified embeddings for all sources?',
                                buttons=[
                                        ('No', False),
                                        ('Yes', True),
                                        ]).run()
            if all_sources:
                cognitive_sources = ['all']

        elif not embeddings:
            embeddings = []
        
        if not main_conf_dict:
            return
        
        try:
            cognitive_sources, embeddings = resolve_cog_emb(modalities,
                                                            cognitive_sources,
                                                            embeddings,
                                                            main_conf_dict,                                                            
                                                            cog_data_config_dict,
                                                            embedding_registry,
                                                            scope="config")
        except AbortException:
            return

        cog_data_config_dict = main_conf_dict['cogDataConfig']
        
        # Remove experiments
        if embeddings:
            cprint("Removing experiments ...", "magenta")
            for csource in cognitive_sources:            
                for emb in embeddings:
                    if emb in cog_data_config_dict[csource]["wordEmbSpecifics"]:
                        rand_emb = main_conf_dict["wordEmbConfig"][emb]["random_embedding"]
                        if rand_emb:
                            cprint ("Deleting {}/{} and associated random embedding set {}/{}...".format(csource, emb, csource, rand_emb), 'green')
                        else:
                            cprint ("Deleting {}/{} ...".format(csource, emb), 'green')
                        del cog_data_config_dict[csource]["wordEmbSpecifics"][emb]
                        if rand_emb:
                            for rand_emb_part in main_conf_dict["randEmbSetToParts"][rand_emb]:
                                del cog_data_config_dict[csource]["wordEmbSpecifics"][rand_emb_part]
                    else:
                        cprint ("Combination {}/{} not found, skipping ...".format(csource, emb), 'yellow')
            
            # Remove source if empty
            if not cog_data_config_dict[csource]["wordEmbSpecifics"]:
                cprint("Deleting now empty source {} ...", 'yellow')
                del cog_data_config_dict[csource]

        # Remove complete cognitive source
        else:
            cprint("Removing cognitive sources ...", "magenta")
            for csource in cognitive_sources:            
                if csource in cog_data_config_dict:
                    cprint ("Deleting {} ...".format(csource), 'green')
                    del cog_data_config_dict[csource]
    
        _save_config(main_conf_dict, configuration, resources_path)


@command()
@argument('embeddings',
          type=str,
          description='Name of embeddings that have been registered (not necessarily installed).',
          positional=True)
@argument('no_embeddings',
          type=int,
          description='Number of random embeddings to be generated (and across which performance will later be averaged).')
@argument('seed_func',
          type=str,
          description='Number of random embeddings to be generated (and across which performance will later be averaged).')
def generate_random(embeddings, no_embeddings=10, seed_func='exp_e_floored'):
    """
    Generate random embeddings for specified proper embeddings.
―
    """
    ctx = context.get_context()
    resources_path = ctx.resources_path
    embeddings_path = ctx.embeddings_path

    if embeddings.startswith('random'):
        cprint('✗ Reference embedding must be non-random! Aborting ...'. format(embeddings), 'red')
        return
    
    ctx = context.get_context()
    emb_properties = ctx.embedding_registry['proper'].get(embeddings, None)
    if not emb_properties:
        cprint('✗ No specifications set for embeddings {}! Install custom embeddings or register them manually. Aborting ...'. format(embeddings), 'red')
        return

    emb_dim = emb_properties['dimensions']

    available_dims = set()
    for _, parameters in ctx.embedding_registry['random_multiseed'].items():
        if parameters['installed']:
            available_dims.add(parameters['dimensions'])

    # Obtain seed values from non-linear function
    if seed_func == 'exp_e_floored':
        seeds = [int(np.floor((k+1)**np.e)) for k in range(no_embeddings)]
    else:
        raise NotImplementedError('Only floor(x**e) (exp_e_floored) currently implemented')
    
    rand_emb_name = 'random-{}-{}'.format(emb_dim, len(seeds))
    
    # Generate random embeddings if not already present
    if emb_dim not in available_dims:
        cprint('No pre-existing random embeddings of dimensionality {}, generating ...'.format(emb_dim), 'yellow')

        with open(resources_path / 'standard_vocab.txt') as f:
            vocabulary = f.read().split('\n')
        cprint('Generating {}-dim. random embeddings using standard CogniVal vocabulary ({} tokens)...'.format(emb_dim, len(vocabulary)), 'yellow')

        # Generate random embeddings
        rand_emb_keys = ['{}_{}_{}'.format(rand_emb_name, idx+1, seed) for idx, seed in enumerate(seeds)]
        path = embeddings_path / 'random_multiseed' / '{}_dim'.format(emb_dim) / '{}_seeds'.format(len(seeds))
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        with ProgressBar() as pb:
            # TODO: Make n_jobs parametrizable
            Parallel(n_jobs=(os.cpu_count()-1))(delayed(generate_random_df)(seed, vocabulary, emb_dim, emb_file, path) for seed, emb_file in pb(list(zip(seeds, rand_emb_keys))))
                
        ctx.embedding_registry['random_multiseed'][rand_emb_name] = {'url': 'locally generated',
                                                                    'dimensions': emb_dim,
                                                                    'path':str(path),
                                                                    'embedding_parts':{},
                                                                    'installed': True,
                                                                    'chunked': False,
                                                                    'associated_with':[embeddings]}
        for rand_emb_key in rand_emb_keys:
            ctx.embedding_registry['random_multiseed'][rand_emb_name]['embedding_parts'][rand_emb_key] = '{}.txt'.format(rand_emb_key)
        
    else:
        if not embeddings in ctx.embedding_registry['random_multiseed'][rand_emb_name]['associated_with']:
            cprint('Random embeddings of dimensionality {} already present, associating ...'.format(emb_dim), 'green')
            ctx.embedding_registry['random_multiseed'][rand_emb_name]['associated_with'].append(embeddings)
        else:
            cprint('Random embeddings of dimensionality {} already present and associated.'.format(emb_dim), 'green')
    
    # Associate random embeddings with proper embeddings
    emb_properties['random_embedding'] = rand_emb_name
    cprint('✓ Generated random embeddings (Naming scheme: random-<dimensionality>-<no. seeds>-<#seed>-<seed_value>)', 'green')
    ctx.save_configuration()

@command
@argument('configuration', type=str, description='Name of configuration file', positional=True)
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('alpha', type=str, description='Alpha value')
@argument('test', type=str, description='Significance test')
def sig_test(configuration, modalities=['eye-tracking', 'eeg', 'fmri'], alpha=0.01, test='Wilcoxon'):    
    '''
    Test significance of results in the given modality and produced based on the specified configuration.
―
    '''
    ctx = context.get_context()
    resources_path = ctx.resources_path

    config_dict = _open_config(configuration, resources_path)
    if not config_dict:
        return
    
    out_dir = Path(config_dict["PATH"]) / config_dict["outputDir"]

    if not os.path.exists(out_dir):
        cprint('Output path {} associated with configuration "{}" does not exist. Have you already performed experimental runs?'.format(out_dir, configuration), "red")
        return

    # Get mapping of previous version (current not yet executed)
    version = config_dict['version'] - 1

    with open(out_dir / 'mapping_{}.json'.format(version)) as f:
        mapping_dict = json.load(f)
    
    for modality in modalities:
        cprint('\n[{}]\n'.format(modality.upper()), attrs=['bold'], color='green')
        experiments_dir = out_dir / 'experiments'
        sig_test_res_dir = out_dir / 'sig_test_results' / modality
        report_dir = out_dir / 'reports' / modality
        
        # Erase previously generated report files and significance test files
        if os.path.exists(report_dir / '{}.json'.format(test)):
            os.remove(report_dir / '{}.json'.format(test))
        if os.path.exists(sig_test_res_dir):
            shutil.rmtree(sig_test_res_dir)
        
        os.makedirs(sig_test_res_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        datasets = [k for (k, v) in config_dict["cogDataConfig"].items() if 'modality' in v and v['modality'] == modality]
        
        emb_bl_pairs =[]
        for k, v in config_dict["wordEmbConfig"].items():
            if v['random_embedding']:
                emb_bl_pairs.append((k, v['random_embedding']))
            else:
                cprint('Embedding {} has no associated random embeddings, no significance test possible, skipping ...', 'yellow')

        embeddings, baselines = zip(*emb_bl_pairs)

        for ds in datasets:
            for feat in config_dict["cogDataConfig"][ds]["features"]:
                for embed in embeddings:
                    experiment = '{}_{}_{}'.format(ds, feat, embed)
                    try:
                        st_extract_results(version, modality, experiment, mapping_dict, experiments_dir, sig_test_res_dir)
                    except KeyError:
                        pass

        hypotheses = sum([1 for filename in os.listdir(sig_test_res_dir) if 'embeddings_' in filename])
        
        # Skip modality if no hypotheses found ...
        if not hypotheses:
            cprint('No hypotheses, skipping ...', 'yellow')
            continue

        cprint('Number of hypotheses: {}'.format(hypotheses), attrs=['bold'])
        bf_corr_alpha = bonferroni_correction(alpha, hypotheses)
        print('α (initial/after bonferroni correction): {} / {}'.format(alpha, bf_corr_alpha))

        report = report_dir / '{}.json'.format(test)
        results = collections.defaultdict(dict)
        results['bonferroni_alpha'] = bf_corr_alpha

        cprint('\n[Significance tests]:', attrs=['bold'])

        for filename in os.listdir(sig_test_res_dir):
            if not 'baseline' in filename:
                experiment = re.sub(r'embeddings_scores_(.*?).txt', r'\1', filename)
                embedding = mapping_dict[experiment]['embedding']
                with open(experiments_dir / mapping_dict[experiment]['proper'] / '{}.json'.format(embedding)) as f:
                    result_json = json.load(f)
                avg_mse = result_json['AVERAGE_MSE']
                feature = mapping_dict[experiment]['feature']

                model_file = sig_test_res_dir / filename
                baseline_file = sig_test_res_dir / 'baseline_{}'.format(filename.partition('_')[2])
                significant, pval, name = test_significance(baseline_file, model_file, bf_corr_alpha, test)
                
                results['hypotheses'][name] = {'p_value': pval,
                                                'alpha': alpha,
                                                'significant': significant,
                                                'wordEmbedding': embedding,
                                                'feature': feature,
                                                'AVERAGE_MSE': avg_mse}

                results['hypotheses'][name]['feature'] = feature

                if significant:
                    out_str, color = ': significant (p = {:1.3e})'.format(pval), 'green'
                else:
                    out_str, color = ': not significant (p = {:1.3e})'.format(pval), 'red'
                
                cprint('    - {}'.format(name), attrs=['bold'], color=color, end='')
                cprint(out_str, color)

        with open(report, 'w') as fp:
            json.dump(dict(results), fp, indent=4)

@command
@argument('configuration', type=str, description='Name of configuration file', positional=True)
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('test', type=str, description='Significance test')
def aggregate(configuration,
              modalities=['eye-tracking', 'eeg', 'fmri'],
              test="Wilcoxon"):    
    '''
    Test significance of results in the given modality and produced based on the specified configuration.
―
    '''
    ctx = context.get_context()
    resources_path = ctx.resources_path
    config_dict = _open_config(configuration, resources_path)

    if not config_dict:
        return

    # Get mapping of previous version (current not yet executed)
    version = config_dict['version'] - 1
    out_dir = Path(config_dict['PATH']) / config_dict['outputDir']

    with open(out_dir / 'mapping_{}.json'.format(version)) as f:
        mapping_dict = json.load(f)

    report_dir = out_dir / 'reports'
    
    if not os.path.exists(out_dir):
        cprint('Output path {} associated with configuration "{}" does not exist. Have you already performed experimental runs?'.format(out_dir, configuration), "red")
        return

    extract_results = {'eeg': agg_eeg_extr_results,
                       'fmri': agg_fmri_extr_results,
                       'eye-tracking': agg_et_extr_results}

    aggregate_significance = {'eeg': aggregate_signi_eeg,
                              'fmri': aggregate_signi_fmri,
                              'eye-tracking': aggregate_signi_et}
    
    modality_to_experiments = collections.defaultdict(list)
    
    options_dicts = []
    for modality in modalities:
        try:
            with open(Path(out_dir) / 'experiments' / modality / 'options_{}.json'.format(version)) as f:
                options_dict = json.load(f)
                options_dicts.append(options_dict)
        except FileNotFoundError:
            cprint("No results for modality {}, skipping ...".format(modality), "yellow")
            options_dicts.append(None)

        for experiment, properties in mapping_dict.items():
            if properties['modality'] == modality:
                modality_to_experiments[modality].append(properties['embedding'])

    emb_bl_pairs =[]

    for k, v in config_dict["wordEmbConfig"].items():
        if v['random_embedding']:
            emb_bl_pairs.append((k, v['random_embedding']))
        else:
            cprint('Embedding {} has no associated random embeddings, no significance test possible, skipping ...', 'yellow')

    embeddings, baselines = zip(*emb_bl_pairs)
    
    for modality, options_dict in zip(modalities, options_dicts):
        cprint('\n[{}]\n'.format(modality.upper()), attrs=['bold'], color='green')
    
        if not options_dict:
            continue

        results = extract_results[modality](options_dict)

        significance = aggregate_significance[modality](report_dir,
                                                        test,
                                                        modality_to_experiments[modality])
        
        df_rows = []
        df_rows_cli = []

        for emb, base in zip(embeddings, baselines):
            try:
                if modality == 'eye-tracking':
                    avg_base = results[base]
                    avg_emb = results[emb]
                else:
                    avg_base = results[base]
                    avg_emb = results[emb]
                df_rows_cli.append({'Embedding':emb, 'Ø MSE Baseline':avg_base, 'Ø MSE Proper':avg_emb, 'Significance':  colored(significance[emb], 'yellow')})
                df_rows.append({'Embedding':emb, 'Ø MSE Baseline':avg_base, 'Ø MSE Proper':avg_emb, 'Significance': significance[emb]})
            except KeyError:
                pass

        df_cli = pd.DataFrame(df_rows_cli)
        df = pd.DataFrame(df_rows)
        df.set_index('Embedding', drop=True, inplace=True)
        df.to_json(report_dir / modality / 'aggregated_scores.json')
        print(tabulate.tabulate(df_cli, headers="keys", tablefmt="fancy_grid", showindex=False))

@command
@argument('configuration', type=str, description='Name of configuration file')
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('test', type=str, description='Significance test')
def report(configuration, modalities=['eye-tracking', 'eeg', 'fmri'], test="Wilcoxon"):
    '''
    (not yet implemented)
―
    '''
    pass


@command
class Install:
    """Install CogniVal cognitive vectors, default embeddings (proper and random) and custom embeddings.
    [Sub-commands]
    - cognitive_sources: Install the entire batch of preprocessed CogniVal and other cognitive sources.
    - embedding: Download and install a default embedding (by name) or custom embedding (from URL)
―
    """

    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command(aliases=["cognitive"])
    @argument('source', type=str, description='Cognitive source')
    def cognitive_sources(self, source='cognival'):
        """
        Download the entire batch of preprocessed CogniVal cognitive sources.
        """
        ctx = context.get_context()
        cognival_path = ctx.cognival_path
        resources_path = ctx.resources_path

        if source == 'cognival': 
            url = COGNIVAL_SOURCES_URL
        else:
            url = None

        basepath = cognival_path / 'cognitive_sources'
        cog_config = _open_cog_config(resources_path)

        if source == 'cognival':
            if not cog_config['cognival_installed']:
                fullpath = basepath / 'cognival_vectors.zip'
                cprint("Retrieving CogniVal cognitive sources ...", "yellow")
                gdown.cached_download(url, path=str(fullpath), quiet=False, postprocess=gdown.extractall)
                cognival_dir = basepath / 'cognival-vectors'
                subdirs = os.listdir(cognival_dir)
                for subdir in subdirs:
                    shutil.move(str(cognival_dir / subdir), basepath)
                os.remove(basepath / 'cognival_vectors.zip')
                for path in ['__MACOSX', '.DS_Store', 'cognival-vectors']:
                    try:
                        shutil.rmtree(basepath / path)
                    except NotADirectoryError:
                        os.remove(basepath / path)
                cog_config['cognival_installed'] = True
            else:
                cprint("CogniVal sources already present!", "green")
                return 
        else:
            if not cog_config['cognival_installed']:
                cprint('Please install CogniVal source before installing custom cognitive sources (run this command without argument).', 'yellow')
                return

            # Specify path
            message_dialog(title='Cognitive source registration',
                           text='Custom cognitive sources MUST conform to the CogniVal format (space-separated, columns word, feature\n'
                                'or dimension columns (named e[i])) '
                                'and be put manually in the corresponding directory (cognitive_sources/<modality>/) after running this assistant.\n'
                                'The specified name ({}) must match the corresponding text file! \n'
                                'Multi-hypothesis (multi-file) sources must currently be added manually. '.format(source)).run()


            modality = radiolist_dialog(title='Cognitive source registration',
                                        text='Specify cognitive source modality:',
                                        values=[('eeg', 'EEG (Electroencephalography)'),
                                                ('eye-tracking', 'Eye-Tracking'),
                                                ('fmri', 'fMRI (Functional magnetic resonance imaging)'),
                                                ]).run()

            if modality is None:
                return

            message_dialog(title='Cognitive source registration',
                           text='Please ensure that the file has the following path and name after installation:\n'
                                'cognitive_sources/{}/{}.txt'.format(modality, source)).run()
            
            dimensionality = input_dialog(title='Cognitive source registration',
                                    text='Specify the dimensionality of the cognitive source. Leave empty if each dimension constitutes a separate \n'
                                         'feature. Multi-dimensional multi-feature sources are not supported.').run()
            
            if dimensionality is None:
                return
            
            if not dimensionality:
                dimensionality = 1

            dimensionality = int(dimensionality)

            features = input_dialog(title='Cognitive source registration',
                                    text='If the source has multiple features, specify below, separated by comma. Leave empty otherwise.').run()
            
            if features is None:
                return
            
            if not features:
                features = 'single'
            else:
                features = [x.strip() for x in features.split(',')]

            # Prefix modality
            cog_config['sources'][modality][source] = {'file': '{}.txt'.format(source),
                                                       'features': features,
                                                       'dimensionality': dimensionality,
                                                       'hypothesis_per_participant': False}
            index = cog_config['index']
            index.append(source)
            cog_config['index'] = natsorted(list(set(index)))
        
        cprint("Completed installing cognitive sources ({})".format(source), "green")

        paths = DisplayablePath.make_tree(basepath, max_len=10, max_depth=3)

        for path in paths:
            cprint(path.displayable(), 'cyan')

        _save_cog_config(cog_config, resources_path)

    @command()
    @argument('x', type=str, description='Force removal and download', positional=True) #choices=list(CTX.embedding_registry['proper']))
    @argument('force', type=bool, description='Force removal and download')
    def embedding(self, x, force=False, log_only_success=False, are_set=False, associate_rand_emb=None):
        """
        Download and install a default embedding (by name) or custom embedding (from URL)
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        embeddings_path = ctx.embeddings_path

        if not are_set:
            associate_rand_emb = yes_no_dialog(title='Random embedding generation',
                                    text='Do you wish to compare the embeddings with random embeddings of identical dimensionality? \n').run()
        local = False                                   
        
        # Download all embeddings
        if x == 'all':
            for emb in ctx.embedding_registry['proper']:
                self.embedding(emb,
                               are_set=True,
                               associate_rand_emb=associate_rand_emb)

            # Download random embeddings
            if ctx.debug:
                for rand_emb in ctx.embedding_registry['random_static']:
                    self.embedding(rand_emb, log_only_success=True)
            return

        # Download all static random embeddings
        elif x == 'all_random':
            if ctx.debug:
                for rand_emb in ctx.embedding_registry['random_static']:
                    self.embedding(rand_emb, log_only_success=True)
                folder = ctx.embedding_registry['random_static'][name]['path']
            else:
                cprint('Error: Random embeddings must be generated using "generate_random"', 'red')
                return

        # Download a set of static random embeddings
        elif x.startswith('random'):
            if ctx.debug:
                name = x
                url = ctx.embedding_registry['random_static'][x]['url']
                path = 'random_static'
                folder = ctx.embedding_registry['random_static'][name]['path']
            else:
                cprint('Error: Random embeddings must be generated using "generate_random"', 'red')
                return

        # Download a set of default embeddings
        elif x in ctx.embedding_registry['proper']:
            if x.startswith('bert'):
                cprint('Warning:', attrs=['bold', 'reverse'], color='red', end='')
                print(' ', end='')
                cprint('BERT conversion is extremely memory-intensive. 16GB of RAM or more (depending on the embedding size) highly recommended. Press Ctrl-C to abort.', attrs=['reverse'], color='yellow')
            name = x
            url = ctx.embedding_registry['proper'][x]['url']
            # Always assume subdirectories to be part of the archive!
            path = ctx.embedding_registry['proper'][x]['path'].split('/')[0]
            folder = ctx.embedding_registry['proper'][name]['path']
            emb_dim = ctx.embedding_registry['proper'][name]['dimensions']

        # Download custom embedding via URL
        else:
            if os.path.exists(x):
                local = True
            elif not x.startswith('http') and x is not None:
                cprint("Specified value is neither a default embedding, valid URL or path of an existing file, aborting ...", "red")
                list_embeddings()
                return

            url = x
            
            name = input_dialog(title='Embedding registration',
                                text='You have provided a custom embedding URL/file path ({}). Please make sure that\n'
                                     'all of the following criteria are met:\n\n'
                                     '- The URL is either a direct HTTP(S) link to the file or a Google Drive link. \n'
                                     '- The file is either a ZIP archive, gzipped file or usable as-is (uncompressed).\n\n'
                                     'Other modes of hosting and archival are currently NOT supported and will cause the installation to fail.\n'
                                     'In those instances, please manually download and extract the files in the "embeddings"'
                                     'directory and \nregister them in "resources/embedding_registry.json"\n\n'
                                     'Please enter a short name for the embeddings:'.format(url),
                                ).run()
            if not name:
                name = 'my_custom_embeddings'
                cprint('No name specified, using "my_custom_embeddings" ...', 'yellow')

            main_emb_file = input_dialog(title='Embedding registration',
                        text='Specify the main embedding file. This information is usually available from the supplier.\n'
                             'If not available, you can leave this information empty and manually edit resources/embeddings2url.json\n'
                             'after the installation.').run()

            if not main_emb_file:
                main_emb_file = url.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
                cprint('No main embedding file specified, inferring "{}" (only works for .gz compressed single files!)'.format(main_emb_file), 'yellow')

            path = input_dialog(title='Embedding registration',
                                text='Optionally specify the directory name for the embeddings. The path may specify more details. \n'
                                     'If no string is specified, the name of the main embedding file (without extension) is used instead.'.format(url)).run()


            if path == '':
                folder = path = main_emb_file.rsplit('.', maxsplit=1)[0]
            elif not path:
                cprint('Aborted.', 'red')
                return

            emb_dim = input_dialog(title='Embedding registration',
                        text='Please specify embedding dimensionality:').run()
            
            if not emb_dim:
                cprint('Aborted.', 'red')
                return
            
            try:
                emb_dim = int(emb_dim)
            except ValueError:
                cprint('Error: {} is not a valid embedding dimensionality, aborting ...'.format(emb_dim), 'red')
                return

            emb_binary = yes_no_dialog(title='Embedding registration',
                                       text='Is the embedding file binary? If "No" is chosen, the file \n'
                                        'will be treated as text (only-space separated formats supported).').run()

            emb_binary_format = None
            truncate_first_line = False
            if emb_binary:
                emb_binary_format = radiolist_dialog(title='Embedding registration',
                                               text='Choose the binary format (switch to buttons using <Tab>). Note that unlisted formats (e.g. ELMo) cannot be processed automatically.',
                                               values=[('word2vec', 'word2vec-compliant (e.g. fasttext. Requires gensim)'),
                                                       ('bert', 'BERT-compliant (requires bert-as-service)')]).run()

                if not emb_binary_format:
                    cprint('Aborted.', 'red')
                    return
            else:
                truncate_first_line = yes_no_dialog(title='Embedding registration',
                                                    text='Is the embedding prefixed by a header row or row specifying dimensionality?')

            chunk_embeddings = button_dialog(title='Embedding registration',
                                            text='Large embeddings can be memory-intensive. Do you wish to segment the embeddings into chunks?\n'
                                                  'The original file will be retained.',
                                            buttons=[
                                                    ('No', False),
                                                    ('Yes', True),
                                                    ]).run()

            chunk_number = 0
            if chunk_embeddings:
                chunk_number = input_dialog(title='Embedding registration',
                                             text='Please specify the number of chunks. If left empty, the value is set to 4.').run()
                try:
                    if chunk_number:
                        chunk_number = int(chunk_number)
                except ValueError:
                    cprint('Invalid value, aborting ...', 'red')
                    return
                
                if not chunk_number:
                    chunk_number = 4

            ctx.embedding_registry['proper'][name] = {'url': url,
                                                 'dimensions': emb_dim,
                                                 'path': path,
                                                 'embedding_file': main_emb_file if not emb_binary else '{}.txt'.format(main_emb_file.rsplit('.', maxsplit=1)[0]),
                                                 'installed': False,
                                                 'binary': emb_binary,
                                                 'binary_format': emb_binary_format,
                                                 'binary_file': main_emb_file if emb_binary else None,
                                                 'truncate_first_line': True if emb_binary_format == 'word2vec' else truncate_first_line,
                                                 'chunked': chunk_embeddings,
                                                 'chunk_number': chunk_number,
                                                 'chunked_file': main_emb_file.rsplit('.', maxsplit=1)[0] if chunk_embeddings else None,
                                                 'chunk_ending': '.txt',
                                                 'random_embedding': None}
            
            # Make mapping for current session (loaded from config upon next start)
            ctx.path2embeddings[folder] = [name]

        # Check if embeddings already installed
        if x in ctx.embedding_registry['proper'] and ctx.embedding_registry['proper'][x]['installed'] and not force:
            if not log_only_success:
                cprint('Embedding {} already installed. Use "force" to override'.format(name), 'yellow')
            return

        if url:
            fname = url.split('/')[-1]
            fpath = embeddings_path / fname
            fpath_extracted = embeddings_path / path

            try:
                shutil.rmtree(fpath_extracted)
            except FileNotFoundError:
                pass

            cprint('Downloading and installing:', 'yellow', end =' ') 
            cprint('{}'.format(name), 'yellow', attrs=['bold'])
            # Google Drive downloads
            if 'drive.google.com' in url:
                gdown.download(url, str(embeddings_path / 'gdrive_embeddings.dat'), quiet=False)
                try:
                    with zipfile.ZipFile(embeddings_path / 'gdrive_embeddings.dat', 'r') as zip_ref:
                        zip_ref.extractall(fpath_extracted) 
                except zipfile.BadZipFile:
                    # Assume gzipped bin (requires manually creating path and setting filename)
                    os.mkdir(fpath_extracted)
                    with gzip.open(embeddings_path / 'gdrive_embeddings.dat', 'rb') as f_in:
                        with open(fpath_extracted / '{}.bin'.format(path), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                os.remove(embeddings_path / 'gdrive_embeddings.dat')
            # Normal HTTP downloads
            else:
                os.makedirs(fpath_extracted)
                # TODO: Test this
                if local:
                    shutil.copy(url, fpath_extracted / fname)
                else:
                    download_file(url, fpath)
                    if fname.endswith('zip'):
                        with zipfile.ZipFile(fpath, 'r') as zip_ref:
                            zip_ref.extractall(fpath_extracted)
                        os.remove(fpath)
                    elif fname.endswith('gz'):
                        # Assume gzipped bin (requires manually setting filename)
                        with gzip.open(fpath, 'rb') as f_in:
                            with open(fpath_extracted / '{}.bin'.format(path), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(fpath)
                    else:
                        shutil.move(fpath, fpath_extracted / fname)
        else:
            cprint('No URL specified, delegating embedding provision to external package ...', 'yellow')

        # TODO: Test this properly
        for name in ctx.path2embeddings[folder]:
            base_path = embeddings_path / ctx.embedding_registry['proper'][name]['path']
            try:
                bin_file = Path(ctx.embedding_registry['proper'][name]['binary_file'])
            except TypeError:
                bin_file = None
            emb_file = Path(ctx.embedding_registry['proper'][name]['embedding_file'])
            try:
                bin_path = base_path / bin_file
            except TypeError:
                bin_path = None
            emb_path = base_path / emb_file
            emb_dim = ctx.embedding_registry['proper'][name]['dimensions']
            
            # Convert from binary to text
            try:
                if ctx.embedding_registry['proper'][name]['binary']:
                    if ctx.embedding_registry['proper'][name]['binary_format'] == 'word2vec':
                        cprint('Converting binary to txt format ...', 'yellow')
                        word2vec_bin_to_txt(base_path, bin_file, emb_file)
                        os.remove(bin_path)

                    elif ctx.embedding_registry['proper'][name]['binary_format'] == 'elmo':
                        elmo_to_text(resources_path / 'standard_vocab.txt',
                                    emb_path,
                                    layer='nocontext')

                    elif ctx.embedding_registry['proper'][name]['binary_format'] == 'bert':
                        bert_to_text(resources_path / 'standard_vocab.txt',
                                    base_path,
                                    emb_path,
                                    emb_dim,
                                    NUM_BERT_WORKERS)
                    else:
                        cprint("Unknown binary format, aborting ...", "red")
                        return
            except FileNotFoundError:
                cprint("Warning: No binary file found, assume binarization already performed ...", "yellow")

            # Chunk embeddings
            if ctx.embedding_registry['proper'][name]['chunked']:
                cprint('Chunking {} ...'.format(name), 'yellow')
                chunk(base_path,
                      base_path,
                      emb_file,
                      ctx.embedding_registry['proper'][name]['chunked_file'],
                      number_of_chunks=ctx.embedding_registry['proper'][name]['chunk_number'],
                      truncate_first_line=ctx.embedding_registry['proper'][name]["truncate_first_line"])
                ctx.embedding_registry['proper'][name]["truncate_first_line"] = False

            cprint('Finished installing embedding "{}"'.format(name), 'green')

            if associate_rand_emb:                                                
                generate_random(name)
        
        if name.startswith('random'):
            ctx.embedding_registry['random_static'][name]['installed'] = True
        else:
            for name in ctx.path2embeddings[folder]:
                ctx.embedding_registry['proper'][name]['installed'] = True
        ctx.save_configuration()


@command
def list_embeddings():
    """
    List available and installed default embeddings as well as installed custom and random embeddings.
―
    """
    ctx = context.get_context()
    if ctx.debug:
        emb_types = ['proper', 'random_static', 'random_multiseed']
    else:
        emb_types = ['proper', 'random_multiseed']

    for emb_type in emb_types:
        cprint(emb_type.title(), attrs=['bold'])
        for key, value in ctx.embedding_registry[emb_type].items():
            cprint(key, 'cyan', end=' '*(25-len(key)))
            if value['installed']:
                cprint('installed', 'green', attrs=['bold'])
            else:
                cprint('not installed', 'red', attrs=['bold'])


@command
def list_cognitive_sources():
    """
    List CogniVal cognitive sources (must be installed)
―
    """
    ctx = context.get_context()
    resources_path = ctx.resources_path
    cog_config = _open_cog_config(resources_path)

    if not cog_config['cognival_installed']:
        cprint('CogniVal cognitive sources not installed, aborting ...', 'red')
        return

    for modality in cog_config['sources']:
        cprint(modality.upper(), attrs=['bold'])
        cprint('Resource                 Features')
        for key, value in cog_config['sources'][modality].items():
            cprint(key, 'cyan', end=' '*(25-len(key)))
            if isinstance(value["features"], str):
                cprint(value["features"], 'green')
            else:
                cprint(value["features"][0], 'green')
                for feature in value["features"][1:]:
                    cprint(' '*25 + feature, 'green')
            print()
        print()


@command
def history():
    '''
    Show history of commands executed in the interactive shell in descending order.
―
    '''
    # Adapted from: https://chase-seibert.github.io/blog/2012/10/31/python-fork-exec-vim-raw-input.html
    pager_list = []
    with open(Path.home() / '.cognival_history') as f:
        for line in f:
            if line.startswith('#'):
                pager_list.append(colored(line, 'green').encode('utf-8'))
            else:
                pager_list.append(colored(line.lstrip('+'), 'white').encode('utf-8'))
    
    # Reverse order of time-stamp-command-whitespace line triplets
    pager_list = itertools.chain.from_iterable(list(chunks(pager_list, 3))[::-1])
    page_list(pager_list)


@command
def readme():
    '''
    Show CogniVal README.md.
―
    '''
    ctx = context.get_context()
    cognival_path = ctx.cognival_path
    with open(cognival_path / 'README.md') as f:
        highlighted_str = highlight(f.read(), MarkdownLexer(), TerminalFormatter())
    page_list([highlighted_str.encode('utf-8')])


@command
def welcome():
    '''
    Show welcome message.
―
    '''
    welcome_msg = messages.WELCOME_MESSAGE_STR
    page_list([welcome_msg.encode('utf-8')])


@command
def clear():
    '''
    Clears console.
―
    '''
    print("\033c")
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

import copy
import json
import os

from pathlib import Path

from natsort import natsorted
from nubia import context
import numpy as np
import pandas as pd

from termcolor import cprint

from handlers.file_handler import write_results, write_options, update_version
from handlers.data_handler import chunk
from handlers.binary_to_text_conversion import bert_to_text, elmo_to_text

from utils import generate_df_with_header, word2vec_bin_to_txt

# Silence TF 2.0 deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local imports

from .form_editor import ConfigEditor, config_editor
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

def _filter_config(configuration,
                  embeddings,
                  cognitive_sources,
                  cognitive_features,
                  random_baseline):
    ctx = context.get_context()
    embedding_registry = ctx.embedding_registry
    resources_path = ctx.resources_path

    config_dict = _open_config(configuration, resources_path)
    
    if not config_dict:
        cprint("Configuration does not exist, aborting ...", "red")
        return
    if not config_dict['cogDataConfig']:
        cprint("No cognitive sources specified in configuration. Have you populated the configuration via 'config experiment ...'? Aborting ...", "red")
        return

    cog_sources_conf = _open_cog_config(resources_path)
    
    installed_embeddings = []
    for emb_type, emb_type_dict in embedding_registry.items():
        for emb, embedding_params in emb_type_dict.items():
            if embedding_params['installed']:
                installed_embeddings.append(emb)
    
    if embeddings[0] == 'all':
        embeddings_list = list(config_dict['wordEmbConfig'].keys())
    else:
        embeddings_list = embeddings

    if cognitive_sources[0] == 'all':
        if cognitive_features:
            cprint('Error: When evaluating all cognitive sources, features may not be specified.', 'red')
            return
        cog_sources_list, cog_feat_list = zip(*[(k, v['features']) for k, v in config_dict['cogDataConfig'].items()])
    else:
        cog_sources_list = cognitive_sources
        if cognitive_features:
            cog_feat_list = [fl.split(';') for fl in cognitive_features]
        else:
            cog_feat_list = [config_dict['cogDataConfig'][csource]['features'] for csource in cognitive_sources]
    
    if not cog_feat_list:
        for cog_source in cog_sources_list:
            modality, csource = cog_source.split('_')
            cog_feat_list.append(cog_sources_conf['sources'][modality][csource]['features'] if not cog_sources_conf['sources'][modality][csource]['features'] == 'single' else ['ALL_DIM'])

    cog_source_to_feature = {i:j for i, j in zip(cog_sources_list, cog_feat_list)}
    
    if not _check_cog_installed(resources_path):
        cprint('CogniVal sources not installed! Aborted ...', 'red')
        return

    for csource in cog_sources_list:
        if csource not in config_dict['cogDataConfig']:
            cprint('Cognitive source {} not registered in configuration {}, aborting ...'.format(csource, configuration), 'red')
            return
    
    emb_to_random_dict = {}
    # Check if embedding installed and registered, get random embeddings
    
    not_registered_str = ''
    not_installed_str = ''
    no_rand_emb_str = ''
    
    for emb in embeddings_list:    
        if emb not in config_dict['wordEmbConfig']:
            not_registered_str += '- {}\n'.format(emb)
        else:
            rand_emb = config_dict['wordEmbConfig'][emb]['random_embedding']

        if not _check_emb_installed(emb, embedding_registry):
            not_installed_str += '- {}\n'.format(emb)
        
        if random_baseline and not rand_emb:
            no_rand_emb_str += '- {}\n'.format(emb)
        else:
            emb_to_random_dict[emb] = rand_emb
    
    terminate = False

    if not_registered_str:
        cprint('The following embeddings are not registered in configuration "{}":'.format(configuration), 'red', attrs=['bold'])
        cprint(not_registered_str, 'red')
        terminate = True

    if not_installed_str:
        cprint('The following embeddings are not installed:', 'red', attrs=['bold'])
        cprint(not_installed_str, 'red')
        terminate = True

    if no_rand_emb_str:
        cprint('For the following embeddings, no random embeddings have been generated:', 'red', attrs=['bold'])
        cprint(no_rand_emb_str, 'red')
        terminate = True

    if terminate:
        return

    return config_dict, embeddings_list, emb_to_random_dict, cog_sources_list, cog_source_to_feature

def cumulate_random_emb_results(logging,
                                word_error,
                                history,
                                cum_rand_word_error_df,
                                cum_rand_logging,
                                cum_mse_prediction,
                                cum_mse_prediction_all_dim,
                                cum_average_mse_all_dim,
                                cum_average_mse,
                                cum_rand_counter):
    # Cumulate logging
    mse_prediction = np.array([x['MSE_PREDICTION'] for x in logging['folds']])
    
    # ALL_DIM for single feature sources
    try:
        mse_prediction_all_dim = np.array([x['MSE_PREDICTION_ALL_DIM'] for x in logging['folds']])
        average_mse_all_dim = np.array(logging['AVERAGE_MSE_ALL_DIM'])
    except KeyError:
        mse_prediction_all_dim = np.array([])
        average_mse_all_dim = np.array([])
    
    average_mse = logging['AVERAGE_MSE']

    if not cum_rand_counter:
        cum_rand_logging = copy.deepcopy(logging)
        cum_mse_prediction = mse_prediction
        cum_mse_prediction_all_dim = mse_prediction_all_dim
        cum_average_mse_all_dim = average_mse_all_dim
        cum_average_mse = average_mse
    else:
        cum_mse_prediction += mse_prediction
        cum_mse_prediction_all_dim += mse_prediction_all_dim
        cum_average_mse_all_dim += average_mse_all_dim
        cum_average_mse += average_mse

    # Cumulate word errors
    rand_word_error_df = pd.DataFrame(word_error)
    rand_word_error_df.columns = rand_word_error_df.iloc[0]
    rand_word_error_df.drop(rand_word_error_df.index[0], inplace=True)
    rand_word_error_df.set_index('word', drop=True, inplace=True)
    rand_word_error_df = rand_word_error_df.applymap((lambda x: float(x)))
    if not cum_rand_counter:
        cum_rand_word_error_df = rand_word_error_df
    else:
        cum_rand_word_error_df += rand_word_error_df
    cum_rand_counter += 1

    return cum_rand_word_error_df, \
           cum_rand_logging, \
           cum_mse_prediction, \
           cum_mse_prediction_all_dim, \
           cum_average_mse_all_dim, \
           cum_average_mse, \
           cum_rand_counter


def write_random_emb_results(rand_emb,
                             cum_rand_word_error_df,
                             cum_rand_logging,
                             cum_mse_prediction,
                             cum_mse_prediction_all_dim,
                             cum_average_mse_all_dim,
                             cum_average_mse,
                             cum_rand_counter,
                             config_dict):
    cum_rand_word_error_df = cum_rand_word_error_df / cum_rand_counter
    cum_rand_word_error = cum_rand_word_error_df.reset_index().to_numpy()
    cum_rand_word_error = np.vstack([np.array(['word', *list(cum_rand_word_error_df.columns)]), cum_rand_word_error])
    cum_mse_prediction /= cum_rand_counter
    cum_mse_prediction_all_dim /= cum_rand_counter
    cum_average_mse_all_dim /= cum_rand_counter
    cum_average_mse /= cum_rand_counter
    cum_mse_prediction_all_dim = list(cum_mse_prediction_all_dim)
    
    for idx, fold in enumerate(cum_rand_logging['folds']):
        fold['LOSS'] = []
        fold['VALIDATION_LOSS'] = []
        fold['MSE_PREDICTION'] = cum_mse_prediction[idx]
        if cum_mse_prediction_all_dim:
            fold['MSE_PREDICTION_ALL_DIM'] = list(cum_mse_prediction_all_dim[idx])

    if cum_average_mse_all_dim.any():
        cum_rand_logging['AVERAGE_MSE_ALL_DIM'] = list(cum_average_mse_all_dim)

    cum_rand_logging['AVERAGE_MSE'] = cum_average_mse
    cum_rand_logging['wordEmbedding'] = rand_emb
    cum_rand_logging['averagedRuns'] = cum_rand_counter
    write_results(config_dict, cum_rand_logging, cum_rand_word_error , [])


def process_and_write_results(proper_result,
                              random_results,
                              rand_emb,
                              config_dict,
                              options,
                              id_,
                              run_stats):
    cum_rand_word_error_df = None
    cum_rand_logging = None
    cum_mse_prediction = None
    cum_mse_prediction_all_dim = None
    cum_average_mse_all_dim = None
    cum_average_mse = None
    cum_rand_counter = 0

    # Proper embedding
    emb_label, logging, word_error, history = proper_result
    proper_avg_mse = logging["AVERAGE_MSE"]

    # Results proper embedding
    write_results(config_dict, logging, word_error, history)                        

    # Random embedding
    for emb_label, logging, word_error, history in random_results:
        if emb_label.startswith('random'):
            cum_rand_word_error_df, \
            cum_rand_logging, \
            cum_mse_prediction, \
            cum_mse_prediction_all_dim, \
            cum_average_mse_all_dim, \
            cum_average_mse, \
            cum_rand_counter = cumulate_random_emb_results(logging,
                                                            word_error,
                                                            history,
                                                            cum_rand_word_error_df,
                                                            cum_rand_logging,
                                                            cum_mse_prediction,
                                                            cum_mse_prediction_all_dim,
                                                            cum_average_mse_all_dim,
                                                            cum_average_mse,
                                                            cum_rand_counter)
            # Discard history
            history = []


            
    if cum_rand_counter:
        rand_avg_mse = cum_average_mse / cum_rand_counter
        write_random_emb_results(rand_emb,
                                cum_rand_word_error_df,
                                cum_rand_logging,
                                cum_mse_prediction,
                                cum_mse_prediction_all_dim,
                                cum_average_mse_all_dim,
                                cum_average_mse,
                                cum_rand_counter,
                                config_dict)
    
        # Collating options/results for aggregation
        proper_options, rand_options = copy.deepcopy(options), copy.deepcopy(options)
        rand_options['wordEmbedding'] = rand_options['random_embedding']
        del rand_options['random_embedding']

        proper_options["AVERAGE_MSE"] = proper_avg_mse
        rand_options["AVERAGE_MSE"] = rand_avg_mse
        run_stats['{}_{}_proper'.format(config_dict["version"], id_)] = proper_options
        run_stats['{}_{}_random'.format(config_dict["version"], id_)] = rand_options


def insert_config_dict(config_dict, reference_dict, mode, csource, target_emb, source_emb):
    if mode == 'reference':
        try:
            config_dict['cogDataConfig'][csource]["wordEmbSpecifics"][target_emb] = copy.deepcopy(reference_dict['cogDataConfig'][csource]['wordEmbSpecifics'][source_emb])
        except KeyError:
            config_dict['cogDataConfig'][csource]["wordEmbSpecifics"][target_emb] = copy.deepcopy(EMBEDDING_PARAMET_TEMPLATE)
    elif mode == 'empty':
        config_dict['cogDataConfig'][csource]["wordEmbSpecifics"][target_emb] = copy.deepcopy(EMBEDDING_PARAMET_TEMPLATE)


def resolve_cog_emb(modalities,
                    cognitive_sources,
                    embeddings,
                    config_dict,
                    cog_config_dict,
                    embedding_registry,
                    scope=None):
    all_cog = True if cognitive_sources and cognitive_sources[0] == 'all' else False
    all_emb = True if embeddings and embeddings[0] == 'all' else False

    if not modalities and all_cog:
        modalities = ['eye-tracking', 'fmri', 'eeg']
    
    if modalities:
        if scope == 'all':
            cognitive_sources = []
            for type_, type_dict in cog_config_dict['sources'].items():
                if type_ in modalities:
                    for source, source_dict in type_dict.items():
                        if source_dict['hypothesis_per_participant']:
                            for idx in range(len(source_dict['files'])):
                                cognitive_sources.append('{}_{}-{}'.format(type_, source, idx))
                        else:
                            cognitive_sources.append('{}_{}'.format(type_, source))
        elif scope == 'config':
            cognitive_sources = list(config_dict["cogDataConfig"].keys())
    
    if all_emb:
        if scope == 'all':
            embeddings = [k for k, v in embedding_registry['proper'].items() if v['installed']]
        elif scope == 'config':
            embeddings = list(config_dict["wordEmbConfig"])

    if embeddings:
        for x in embeddings:
            if not x in embedding_registry['proper'] or not embedding_registry['proper'][x]['installed']:
                cprint('Embedding {} unknown or not installed, aborting ...'.format(x), 'red')
                raise AbortException

    cog_source_index = set(cog_config_dict['index'])
    
    if cognitive_sources:
        for x in cognitive_sources:
            if x in cog_source_index:
                break
            else:
                cprint('Cognitive source {} unknown, aborting ...'.format(x), 'red')
                raise AbortException

    return cognitive_sources, embeddings


def _edit_config(config_dict, configuration):
    ctx = context.get_context()
    resources_path = ctx.resources_path

    config_patch = {}

    prefill_fields={'PATH': ctx.cognival_path,
                    'outputDir': configuration,
                    'version': 1,
                    'seed': 42,
                    'cpu_count': os.cpu_count()-1,
                    'folds': 5}

    conf_editor = ConfigEditor('main',
                                config_dict,
                                config_patch,
                                singleton_params='all',
                                skip_params=['cogDataConfig', 'wordEmbConfig', 'randEmbConfig', 'randEmbSetToParts'],
                                prefill_fields=prefill_fields)
    conf_editor()
    if config_patch:
        config_dict.update(config_patch)
        
        if not config_dict['outputDir'].startswith('results'):
            cprint('Prefixing outputDir with results ...', 'yellow')
            config_dict['outputDir'] = str(Path('results') / config_dict['outputDir'])

        _save_config(config_dict, configuration, resources_path)
    else:
        cprint("Aborting ...", "red")
        return


def update_emb_config(emb, csource, cdict, config_patch, rand_embeddings, main_conf_dict, embedding_registry):
    cdict.update(config_patch)
    if rand_embeddings or main_conf_dict['randEmbConfig']:
        rand_emb = embedding_registry['proper'][emb]['random_embedding']
        if rand_emb:
            for rand_emb_part in main_conf_dict['randEmbSetToParts']['{}_for_{}'.format(rand_emb, emb)]:
                main_conf_dict['cogDataConfig'][csource]["wordEmbSpecifics"][rand_emb_part].update(config_patch)


def generate_random_df(seed, vocabulary, embedding_dim, emb_file, path):
    np.random.seed(seed)
    rand_emb = np.random.uniform(low=-1.0, high=1.0, size=(len(vocabulary), embedding_dim))
    df = pd.DataFrame(rand_emb, columns=['x{}'.format(i+1) for i in range(embedding_dim)])
    df.insert(loc=0, column='word', value=vocabulary)
    df.to_csv(path / '{}.txt'.format(emb_file), sep=" ", encoding="utf-8", header=False, index=False)


def populate(configuration,
             config_dict,
             rand_embeddings,
             modalities=None,
             cognitive_sources=['all'],
             embeddings=['all'], mode="reference", quiet=False):
    '''
    Populates configuration with templates for some or all installed cognitive sources.
―
    '''
    # TODO: Finish; Show dialog for each source, buttons "Use defaults" (filling form), "Save & Next", "Abort"
    ctx = context.get_context()
    resources_path = ctx.resources_path
    embedding_registry = ctx.embedding_registry
    cog_config_dict = _open_cog_config(resources_path)

    reference_dict = None

    cognitive_sources, embeddings = resolve_cog_emb(modalities,
                                                    cognitive_sources,
                                                    embeddings,
                                                    config_dict,
                                                    cog_config_dict,
                                                    embedding_registry,
                                                    scope="all")

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
                if rand_emb and rand_embeddings:
                    emb_part_list = ['{}_for_{}'.format(rand_emb_part, emb) for rand_emb_part in embedding_registry['random_multiseed'][rand_emb]['embedding_parts']]
                    for rand_emb_part in emb_part_list:
                        insert_config_dict(config_dict, reference_dict, mode, csource, rand_emb_part, emb)

    # Add embedding configurations dicts
    for emb in embeddings:
        emb_key = 'wordEmbConfig'
        emb_dict = copy.deepcopy(embedding_registry['proper'][emb])
        
        config_dict[emb_key][emb] = copy.deepcopy({k:v for k, v in emb_dict.items() if k in WORD_EMB_CONFIG_FIELDS})
        config_dict[emb_key][emb]['path'] = str(Path('embeddings') / embedding_registry['proper'][emb]['path'] / embedding_registry['proper'][emb]['embedding_file'])
        if rand_embeddings and emb_dict['random_embedding']:
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
        else:
            config_dict[emb_key][emb]['random_embedding'] = None
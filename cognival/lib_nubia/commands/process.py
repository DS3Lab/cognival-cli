#!/usr/bin/env python3

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

from handlers.file_handler import write_results

# Silence TF 2.0 deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local imports

from .form_editor import ConfigEditor
from .utils import (_check_emb_installed,
                   AbortException,
                   NothingToDoException)

from .templates import (WORD_EMB_CONFIG_FIELDS,
                        COGNITIVE_CONFIG_TEMPLATE,
                        EMBEDDING_PARAMET_TEMPLATE)


def filter_config(embedding_registry,
                  cog_sources_conf,
                  configuration,
                  config_dict,
                  embeddings,
                  modalities,
                  cognitive_sources,
                  cognitive_features,
                  random_baseline):
    if not config_dict:
        cprint("Configuration does not exist, aborting ...", "red")
        return
    if not config_dict['cogDataConfig']:
        cprint("No cognitive sources specified in configuration. Have you populated the configuration via 'config experiment ...'? Aborting ...", "red")
        return

    emb_type = config_dict['type']

    installed_embeddings = []
    for emb_category, emb_category_dict in embedding_registry.items():
        for emb, embedding_params in emb_category_dict.items():
            try:
                if embedding_params['installed']:
                    installed_embeddings.append(emb)
            except KeyError:
                if embedding_params[emb_type]['installed']:
                    installed_embeddings.append(emb)
    
    # Collect embeddings
    if embeddings[0] == 'all':
        embeddings_list = list(config_dict['wordEmbConfig'].keys())
    else:
        embeddings_list = embeddings

    # Collect cognitive sources
    if modalities:
        if modalities[0] == 'all':
            modalities = ['eeg', 'eye-tracking', 'fmri']
        cog_source_feat_tuples = [(k, v['features']) for k, v in config_dict['cogDataConfig'].items() if v['modality'] in modalities]
        if cog_source_feat_tuples:
            cog_sources_list, cog_feat_list = zip(*cog_source_feat_tuples)
        else:
            cprint('None of the specified modalities found in configuration, aborting ...: {} '.format(', '.join(modalities)), 'red')
            return
    elif cognitive_sources[0] == 'all':
        if cognitive_features:
            cprint('Error: When evaluating all cognitive sources, features may not be specified.', 'red')
            return
        cog_source_feat_tuples = [(k, v['features']) for k, v in config_dict['cogDataConfig'].items()]
        if cog_source_feat_tuples:
            cog_sources_list, cog_feat_list = zip(*cog_source_feat_tuples)
        else:
            cprint('Configuration appears to be empty! Aborting ...', 'red')
            return
    else:
        cog_sources_list = cognitive_sources
        if cognitive_features:
            cog_feat_list = [fl.split(';') for fl in cognitive_features]
        else:
            cog_feat_list = [config_dict['cogDataConfig'][csource]['features'] for csource in cognitive_sources]
    
    if not cog_feat_list:
        for cog_source in cog_sources_list:
            modality, csource = cog_source.split('_')
            cog_feat_list.append(cog_sources_conf['sources'][emb_type][modality][csource]['features'] \
                                    if not cog_sources_conf['sources'][emb_type][modality][csource]['features'] == 'single' \
                                    else ['ALL_DIM'])

    cog_source_to_feature = {i:j for i, j in zip(cog_sources_list, cog_feat_list)}
    
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
        
        if random_baseline:
            if not rand_emb:
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
        cprint('Warning: For the following embeddings, no random embeddings have been associated:', 'yellow')
        cprint(no_rand_emb_str, 'yellow')

    if terminate:
        return
    return config_dict, embeddings_list, emb_to_random_dict, cog_sources_list, cog_source_to_feature


def cumulate_random_emb_results(logging,
                                word_error,
                                history,
                                cum_rand_word_error_df,
                                cum_mse_prediction,
                                cum_mse_prediction_all_dim,
                                cum_average_mse,
                                cum_average_mse_all_dim,
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
        # Init variables
        cum_mse_prediction = mse_prediction
        cum_mse_prediction_all_dim = mse_prediction_all_dim
        cum_average_mse = average_mse
        cum_average_mse_all_dim = average_mse_all_dim
    else:
        # Accumulate
        cum_mse_prediction += mse_prediction
        cum_mse_prediction_all_dim += mse_prediction_all_dim
        cum_average_mse += average_mse
        cum_average_mse_all_dim += average_mse_all_dim

    # Cumulate word errors

    rand_word_error_df = pd.DataFrame(word_error[1:], columns=word_error[0])
    rand_word_error_df.set_index(logging['type'], drop=True, inplace=True)
    rand_word_error_df = rand_word_error_df.applymap((lambda x: float(x)))
    if not cum_rand_counter:
        cum_rand_word_error_df = rand_word_error_df
    else:
        cum_rand_word_error_df += rand_word_error_df
    
    # Increment counter
    cum_rand_counter += 1
    return cum_rand_word_error_df, \
           cum_mse_prediction, \
           cum_mse_prediction_all_dim, \
           cum_average_mse, \
           cum_average_mse_all_dim, \
           cum_rand_counter


def write_random_emb_results(rand_emb,
                             cum_rand_word_error_df,
                             cum_rand_logging,
                             cum_mse_prediction,
                             cum_mse_prediction_all_dim,
                             cum_average_mse,
                             cum_average_mse_all_dim,
                             cum_rand_counter,
                             config_dict):
    # Get average measures by dividing with counter
    cum_rand_word_error_df = cum_rand_word_error_df / cum_rand_counter
    cum_rand_word_error = cum_rand_word_error_df.reset_index().to_numpy()
    cum_rand_word_error = np.vstack([np.array([config_dict['type'], *list(cum_rand_word_error_df.columns)]), cum_rand_word_error])
    cum_mse_prediction /= cum_rand_counter
    cum_mse_prediction_all_dim /= cum_rand_counter
    cum_mse_prediction_all_dim = list(cum_mse_prediction_all_dim)
    cum_average_mse /= cum_rand_counter
    cum_average_mse_all_dim /= cum_rand_counter

    # Do not report per-fold loss, only mse_prediction
    for idx, fold in enumerate(cum_rand_logging['folds']):
        fold['LOSS'] = []
        fold['VALIDATION_LOSS'] = []
        fold['MSE_PREDICTION'] = cum_mse_prediction[idx]
        if cum_mse_prediction_all_dim:
            fold['MSE_PREDICTION_ALL_DIM'] = list(cum_mse_prediction_all_dim[idx])

    if cum_average_mse_all_dim.any():
        cum_rand_logging['AVERAGE_MSE_ALL_DIM'] = list(cum_average_mse_all_dim)

    # Export cumlative random embedding logging information
    cum_rand_logging['AVERAGE_MSE'] = cum_average_mse
    cum_rand_logging['wordEmbedding'] = rand_emb
    cum_rand_logging['averagedRuns'] = cum_rand_counter
    write_results(config_dict, cum_rand_logging, cum_rand_word_error, [])
    return {'cum_rand_logging': cum_rand_logging, 'cum_rand_word_error': cum_rand_word_error}


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
    cum_average_mse = None
    cum_average_mse_all_dim = None
    cum_rand_counter = 0

    # Obtain embedding results
    emb_label, logging, word_error, history = proper_result
    proper_avg_mse = logging["AVERAGE_MSE"]

    # Export embedding resutls
    write_results(config_dict, logging, word_error, history)                        

    # Cumulate random embedding results (if applicable)
    for emb_label, logging, word_error, history in random_results:
        if not cum_rand_logging:
            cum_rand_logging = copy.deepcopy(logging)

        if emb_label.startswith('random'):
            cum_rand_word_error_df, \
            cum_mse_prediction, \
            cum_mse_prediction_all_dim, \
            cum_average_mse, \
            cum_average_mse_all_dim, \
            cum_rand_counter = cumulate_random_emb_results(logging,
                                                            word_error,
                                                            history,
                                                            cum_rand_word_error_df,
                                                            cum_mse_prediction,
                                                            cum_mse_prediction_all_dim,
                                                            cum_average_mse,
                                                            cum_average_mse_all_dim,
                                                            cum_rand_counter)
            # Discard history
            history = []
        else:
            raise RuntimeError

    # If random baselines, export (average) random embedding and store run_stats/options for results aggregation
    if cum_rand_counter:
        write_random_emb_results(rand_emb,
                                cum_rand_word_error_df,
                                cum_rand_logging,
                                cum_mse_prediction,
                                cum_mse_prediction_all_dim,
                                cum_average_mse,
                                cum_average_mse_all_dim,
                                cum_rand_counter,
                                config_dict)
    
        # Collating options/results for aggregation
        rand_avg_mse = cum_average_mse / cum_rand_counter
        proper_options, rand_options = copy.deepcopy(options), copy.deepcopy(options)
        rand_options['wordEmbedding'] = rand_options['random_embedding']
        del rand_options['random_embedding']

        proper_options["AVERAGE_MSE"] = proper_avg_mse
        rand_options["AVERAGE_MSE"] = rand_avg_mse
        run_stats['{}_{}_proper'.format(config_dict["run_id"], id_)] = proper_options
        run_stats['{}_{}_random'.format(config_dict["run_id"], id_)] = rand_options

        # For testing purposes
        return {'run_stats':run_stats, 'proper_options':proper_options, 'rand_options':rand_options}
    

def insert_config_dict(config_dict, reference_dict, mode, csource, target_emb, source_emb):
    we_specs = config_dict['cogDataConfig'][csource]["wordEmbSpecifics"]
    try:
        we_specs_reference = reference_dict['cogDataConfig'][csource]['wordEmbSpecifics']
    except KeyError:
        we_specs_reference = copy.deepcopy(COGNITIVE_CONFIG_TEMPLATE)
    if mode == 'reference':
        try:
            we_specs[target_emb] = copy.deepcopy(we_specs_reference[source_emb])
        except KeyError:
            we_specs[target_emb] = copy.deepcopy(EMBEDDING_PARAMET_TEMPLATE)
    elif mode == 'empty':
        we_specs[target_emb] = copy.deepcopy(EMBEDDING_PARAMET_TEMPLATE)
    
    return we_specs


def resolve_cog_emb(modalities,
                    cognitive_sources,
                    embeddings,
                    config_dict,
                    cog_config_dict,
                    embedding_registry,
                    scope=None):
    all_cog = True if cognitive_sources and cognitive_sources[0] == 'all' else False
    all_emb = True if embeddings and embeddings[0] == 'all' else False

    if (not modalities and all_cog) or (modalities and modalities[0] == 'all'):
        modalities = ['eye-tracking', 'fmri', 'eeg']

    emb_type = config_dict['type']

    # Resolve modalities or parameter 'all to cognitive sources
    if modalities:
        if scope == 'all':
            cognitive_sources = []
            for type_, type_dict in cog_config_dict['sources'][emb_type].items():
                if type_ in modalities:
                    for source, source_dict in type_dict.items():
                        if source_dict['multi_file']:
                            for idx in range(len(source_dict['files'])):
                                cognitive_sources.append('{}_{}-{}'.format(type_, source, idx))
                        else:
                            cognitive_sources.append('{}_{}'.format(type_, source))
        elif scope == 'config':
            cognitive_sources = [k for k, v in config_dict["cogDataConfig"].items() if v["modality"] in modalities]

    # If parameter 'all' passed, return all embeddings (overall or in config)
    if all_emb:
        if scope == 'all':
            # Allow only word embeddings if configuration type is 'word', else all embeddings
            embeddings = [k for k, v in embedding_registry['proper'].items() if \
                            v['installed'] and (v['type'] == 'word' or emb_type == 'sentence')]
        elif scope == 'config':
            embeddings = list(config_dict["wordEmbConfig"])

    # Checks
    if embeddings:
        for emb in embeddings:
            if not emb in embedding_registry['proper'] or \
               not embedding_registry['proper'][emb]['installed']:
                cprint('Embedding {} unknown or not installed, aborting ...'.format(emb), 'red')
                raise AbortException

            if emb in embedding_registry['proper'] and emb_type == 'word' \
               and not embedding_registry['proper'][emb]['type'] == 'sentence':
                cprint('Embedding {} is a sentence embedding, but the configuration has been parametrized for word embeddings!'.format(emb), 'red')
                raise AbortException

    cog_source_index = set(cog_config_dict['index'])
    cognitive_sources_resolved = []

    # Checks and resolution of multi-file sources
    if cognitive_sources:
        for source in cognitive_sources:
            if '{}_{}'.format(emb_type, source) in cog_source_index:
                cognitive_sources_resolved.append(source)
            else:
                if source in cog_config_dict['source_to_file_index'][emb_type]:
                    for subj_source in cog_config_dict['source_to_file_index'][emb_type][source]:
                        cognitive_sources_resolved.append(subj_source)
                else:
                    cprint('Cognitive source {} unknown, aborting ...'.format(source), 'red')
                    raise AbortException
    cognitive_sources = cognitive_sources_resolved
    if not cognitive_sources:
        raise NothingToDoException
    return cognitive_sources, embeddings


def _edit_config(resources_path,
                 cognival_path,
                 config_dict,
                 configuration,
                 create=False):
    config_patch = {}

    # Prefill fields (general configuration properties only)
    prefill_fields={'PATH': cognival_path,
                    'outputDir': configuration,
                    'run_id': 1,
                    'seed': 42,
                    'n_proc': os.cpu_count()-1,
                    'folds': 5}

    # Instantiate and launch configuration editor
    conf_editor = ConfigEditor('main',
                                config_dict,
                                config_patch,
                                singleton_params='all',
                                skip_params=['cogDataConfig',
                                             'wordEmbConfig',
                                             'randEmbConfig',
                                             'randEmbSetToParts'],
                                prefill_fields=prefill_fields,
                                create=create)
    conf_editor()

    # Apply config patch if returned by editor
    if config_patch:
        config_dict.update(config_patch)
        
        # Ensure that results go into the correct directory
        if not config_dict['outputDir'].startswith('results'):
            cprint('Prefixing outputDir with results ...', 'yellow')
            config_dict['outputDir'] = str(Path('results') / config_dict['outputDir'])

        return config_dict
    else:
        cprint("Aborting ...", "red")
        return


def update_emb_config(emb, csource, cdict, config_patch, rand_embeddings, main_conf_dict, embedding_registry):
    # Update source-embedding combination, propagate to associated random baselines if applicable
    main_conf_dict['cogDataConfig'][csource]["wordEmbSpecifics"][emb].update(config_patch)
    if rand_embeddings or main_conf_dict['randEmbConfig']:
        registry_rand_emb = embedding_registry['proper'][emb]['random_embedding']
        config_rand_emb = main_conf_dict['wordEmbConfig'][emb]['random_embedding']
        
        if config_rand_emb:
            config_rand_emb = config_rand_emb.split('_for_')[0]
        
        if config_rand_emb: 
            if not registry_rand_emb == config_rand_emb:
                cprint('Error: The current configuration has random baseline {} associated with embeddings {}. However, baseline {} is currently associated with embeddings {} in the embeddings registry. Please change the association using "import random-embeddings" if you wish to edit this configuration or edit the configuration manually.'. format(config_rand_emb, emb, registry_rand_emb, emb), 'red')
                raise AbortException
            # Propagate patch to all random baseline parts if associated with embeddings
            if config_rand_emb:
                for rand_emb_part in main_conf_dict['randEmbSetToParts']['{}_for_{}'.format(config_rand_emb, emb)]:
                    main_conf_dict['cogDataConfig'][csource]["wordEmbSpecifics"][rand_emb_part].update(config_patch)
    
    return main_conf_dict


def generate_random_df(emb_type, seed, vocabulary, embedding_dim):
    # Generate random embeddings for given vocabulary and dimensionality
    np.random.seed(seed)
    rand_emb = np.random.uniform(low=-1.0, high=1.0, size=(len(vocabulary), embedding_dim))
    df = pd.DataFrame(rand_emb, columns=['x{}'.format(i+1) for i in range(embedding_dim)])
    df.insert(loc=0, column=emb_type, value=vocabulary)
    return df


def populate(resources_path,
             embedding_registry,
             cog_config_dict,
             configuration,
             config_dict,
             rand_embeddings,
             modalities=None,
             cognitive_sources=['all'],
             embeddings=['all'],
             mode="reference",
             quiet=False):
    #Populates configuration with templates for some or all installed cognitive sources.

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
   
    if not cog_config_dict['cognival_installed']:
        cprint("Error: CogniVal cognitive vectors unavailable!", "red")
        raise AbortException
    else:
        emb_type = config_dict['type']
        cog_source_to_file_index = cog_config_dict["source_to_file_index"]
        cog_file_to_source_index = cog_config_dict["file_to_source_index"]
        cog_config_dict = cog_config_dict["sources"][emb_type]

    if mode == 'reference':
        reference_path = resources_path / 'reference_config.json'
        with open(reference_path) as f:
            reference_dict = json.load(f)[emb_type]
 
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

                        # Resolve cog. source if part of multi-hypothesis source
                        csource_resolved = cog_file_to_source_index[emb_type].get(csource, csource)
                        modality, csource_suff = csource.split('_', maxsplit=1)
                        modality, csource_resolved_suff = csource_resolved.split('_', maxsplit=1)
                        csource_dict = cog_config_dict[modality][csource_resolved_suff]

                        # Single file cognitive source
                        if not csource_dict['multi_file']:
                            keys = [key]
                            csource_resolved_suffs = [csource_resolved_suff]
                            features = csource_dict['features'] if not csource_dict['features'] == 'single' else ['ALL_DIM']
                        # Multi file cognitive source
                        else:
                            keys = cog_source_to_file_index[emb_type][csource_resolved] 
                            csource_resolved_suffs = [csource_file.split('_', maxsplit=1)[1] for csource_file in keys]
                            
                            if csource_dict['multi_hypothesis'] == 'feature':
                                # Features list must match source to file index
                                features = [csource_dict['hypothesis_to_feature'][csource_suff]]
                            else:
                                features = csource_dict['features'] if not csource_dict['features'] == 'single' else ['ALL_DIM']

                        if not csource_dict['multi_file']:
                            dataset = str(Path('cognitive_sources') / emb_type / modality / csource_dict['file'])
                        else:
                            dataset = str(Path('cognitive_sources') / emb_type / modality / csource_resolved_suff / csource_dict['hypothesis_to_file'][csource_suff])

                        cdc_dict = {
                                    "parent": "{}_{}".format(modality, csource_resolved_suff),
                                    "dataset": dataset,
                                    "multi_hypothesis": csource_dict['multi_hypothesis'],
                                    "multi_file": csource_dict['multi_file'],
                                    "stratified_sampling": csource_dict['stratified_sampling'],
                                    "balance": csource_dict['balance'],
                                    "modality": modality,
                                    "features": features, 
                                    "type": 'multivariate_output' if csource_dict['dimensionality'] > 1 else 'single_output'
                                    }

                        config_dict['cogDataConfig'][key] = copy.deepcopy(cdc_dict)
                        config_dict['cogDataConfig'][key]['wordEmbSpecifics'] = {}
                    except KeyError:
                        raise
                        cprint('Source {} unknown, aborting ...'.format(csource), 'red')
                        return

            elif mode == 'empty':
                config_dict['cogDataConfig'][key] = copy.deepcopy(COGNITIVE_CONFIG_TEMPLATE)
                
    # Add embedding experiments dicts
    # Dynamic; check if installed
    for csource in cognitive_sources:
        for emb in embeddings:
            if not emb in config_dict['cogDataConfig'][csource]['wordEmbSpecifics']:
                if not embedding_registry['proper'][emb]['installed']:
                    warnings.warn('Skipping {} ... (not installed)'.format(emb), UserWarning)
                    continue
                
                insert_config_dict(config_dict, reference_dict, mode, csource, emb, emb)
                
            if rand_embeddings or config_dict['randEmbConfig']:
                rand_emb = embedding_registry['proper'][emb]['random_embedding']
                if rand_emb and rand_embeddings:
                    emb_part_list = ['{}_for_{}'.format(rand_emb_part, emb) for rand_emb_part \
                                        in embedding_registry['random_multiseed'][rand_emb][emb_type]['embedding_parts']]
                    for rand_emb_part in emb_part_list:
                        insert_config_dict(config_dict, reference_dict, mode, csource, rand_emb_part, emb)

    # Add embedding configurations dicts
    for emb in embeddings:
        emb_key = 'wordEmbConfig'
        emb_dict = copy.deepcopy(embedding_registry['proper'][emb])
        
        config_dict[emb_key][emb] = copy.deepcopy({k:v for k, v in emb_dict.items() if k in WORD_EMB_CONFIG_FIELDS})
        config_dict[emb_key][emb]['path'] = \
                str(Path('embeddings') / embedding_registry['proper'][emb]['path'] / config_dict['type'] / embedding_registry['proper'][emb]['embedding_file'])
        if rand_embeddings and emb_dict['random_embedding']:
            config_dict[emb_key][emb]['random_embedding'] = '{}_for_{}'.format(config_dict[emb_key][emb]['random_embedding'], emb)
            rand_emb = embedding_registry['proper'][emb]['random_embedding']
            if rand_emb:
                emb_dict = copy.deepcopy(embedding_registry['random_multiseed'][rand_emb][emb_type])
                emb_part_list = natsorted(list(embedding_registry['random_multiseed'][rand_emb][emb_type]['embedding_parts']))
                config_dict['randEmbSetToParts']['{}_for_{}'.format(rand_emb, emb)] = ['{}_for_{}'.format(rand_emb_part, emb) for rand_emb_part in emb_part_list]
                for rand_emb_part in emb_part_list:
                    # Add random embedding parameters
                    config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)] = \
                            copy.deepcopy({k:v for k, v in emb_dict.items() if k in WORD_EMB_CONFIG_FIELDS})

                    # Resolve path
                    config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)]['path'] = \
                            str(Path('embeddings') / config_dict['randEmbConfig']['{}_for_{}'.format(rand_emb_part, emb)]['path'] \
                                    / config_dict['type'] / '{}.txt'.format(rand_emb_part))
            else:
                cprint('Embedding {} has no associated random baseline. Generate with `import random-baseliens {}. Aborting ...'.format(emb), 'red')
                raise AbortException
        else:
            config_dict[emb_key][emb]['random_embedding'] = None
    
    return config_dict


def remove_dangling_emb_random(emb, main_conf_dict):
    # Delete embedding config and associated random embedding configs if not longer used by any cognitive source
    if not any(emb in cd_dict["wordEmbSpecifics"] for cd_dict in main_conf_dict['cogDataConfig'].values()):
        cprint("Embedding {} no longer used by any cognitive source, removing (and associated random baselines if applicable and unused)".format(emb), 'yellow')
        assoc_rand_emb = main_conf_dict["wordEmbConfig"][emb]["random_embedding"]
        if assoc_rand_emb:
            for rand_emb_part in main_conf_dict["randEmbSetToParts"][assoc_rand_emb]:
                if not any(rand_emb_part in cd_dict["wordEmbSpecifics"] for cd_dict in main_conf_dict['cogDataConfig'].values()):
                    del main_conf_dict["randEmbConfig"][rand_emb_part]
            del main_conf_dict["randEmbSetToParts"][assoc_rand_emb]
        del main_conf_dict["wordEmbConfig"][emb]
    return main_conf_dict

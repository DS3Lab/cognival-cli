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

import collections
import copy
import csv
import itertools
import json
import gzip
import os
import sys
import traceback
import re
import shutil
import subprocess
import warnings
import zipfile

from pathlib import Path
from textwrap import fill

import gdown
from joblib import Parallel, delayed
from natsort import natsorted
from nubia import command, argument, context
import numpy as np
import pandas as pd
import tabulate
import tableformatter as tform

from prompt_toolkit.shortcuts import input_dialog, button_dialog, yes_no_dialog, message_dialog, radiolist_dialog, ProgressBar
from pygments import highlight
from pygments.lexers import MarkdownLexer
from pygments.formatters import TerminalFormatter
from termcolor import cprint, colored

from cog_evaluate_parallel import run_parallel
from handlers.file_handler import write_options
from handlers.data_handler import chunk
from handlers.binary_to_text_conversion import bert_to_text, elmo_to_text

from utils import word2vec_bin_to_txt

from lib_nubia.prompt_toolkit_table import *
from lib_nubia.commands import messages, commands

# Silence TF 2.0 deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local imports

from .form_editor import ConfigEditor, config_editor
from .process import (filter_config,
                      process_and_write_results,
                      resolve_cog_emb,
                      _edit_config,
                      update_emb_config,
                      generate_random_df,
                      populate)
                      
from .utils import (tupleit,
                   _open_config,
                   _open_cog_config,
                   _backup_config,
                   _save_cog_config,
                   _save_config,
                   DisplayablePath,
                   download_file,
                   AbortException,
                   NothingToDoException,
                   chunked_list_concat_str,
                   field_concat,
                   chunks,
                   page_list,
                   configure_tf_devices)

from .report import generate_report

from .templates import MAIN_CONFIG_TEMPLATE

from .strings import EXAMPLE_COMMANDS

# Source: https://stackoverflow.com/a/22376126
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

NUM_BERT_WORKERS = 1
COGNIVAL_SOURCES_URL = 'https://drive.google.com/uc?id=1ouonaByYn2cnDAWihnQ3cGmMT6bJ4NaP'

### Wrapped commands

@command
@argument("processes", type=int, description="No. of processes")
@argument("n_gpus", type=int, description="No. of processes")
@argument("embeddings", type=list, description="List of embeddings")
@argument('modalities', type=list, description="Modalities of cognitive sources")
@argument("cognitive_sources", type=list, description="List of cognitive sources")
@argument("cognitive_features", type=list, description="List of cognitive features")
@argument("baselines", type=bool, description="Compute random baseline(s) corresponding to specified embedding")
@argument("cache_random", type=bool, description="If True, random embeddings with identical parametrization are cached. Defaults to False.")
@argument("network", type=str, description="Use a multi-layer perceptron with Dropout and BatchNormalization ('mlp', default) or \
                                            a 1D CNN ('cnn') with MaxPooling after each layer and BatchNormalization (Keras default parameters otherwise). \
                                            If CNN, at least two hidden layers must be specified and the last hidden layer is considered a Dense layer.")
@argument("legacy", type=bool, description="If True, use legacy mode when network is MLP (no Dropout, no BatchRegularization). Defaults to False") 
def run(embeddings=['all'],
        modalities=None,
        cognitive_sources=['all'],
        cognitive_features=None,
        processes=None,
        n_gpus=None,
        baselines=True,
        cache_random=False,
        network='mlp',
        legacy=False):
    '''
    Run parallelized evaluation of single, selected or all combinations of embeddings and cognitive sources.
    '''
    ctx = context.get_context()
    resources_path = ctx.resources_path
    configuration = ctx.open_config
    embedding_registry = ctx.embedding_registry
    max_gpus = ctx.max_gpus
    visible_gpu_ids = ctx.visible_gpus

    if not configuration:
        cprint('No configuration open, aborting ...', 'red')
        return

    config_dict = _open_config(configuration, resources_path)

    config_dict = commands.run(configuration,
                               config_dict,
                               resources_path,
                               embedding_registry,
                               embeddings,
                               modalities,
                               cognitive_sources,
                               cognitive_features,
                               processes,
                               n_gpus,
                               max_gpus,
                               visible_gpu_ids,
                               baselines,
                               cache_random,
                               network,
                               legacy)
    if config_dict:
        _save_config(config_dict, configuration, resources_path)


@command
class List:
    """List properties of configurations, embeddings and cognitive sources.
    [Sub-commands] 
    - configs: List available configurations (except reference configuration, which is read-only)
    - embeddings: List available and imported default embeddings, generated random baselines and imported custom embeddings
    - cognitive-sources: List imported cognitive sources.
―
    """

    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command
    def configs(self):
        '''
        List available configurations with general parameters.
    ―
        '''
        ctx = context.get_context()
        resources_path = ctx.resources_path
        
        formatted_list = commands.list_configs(resources_path)

        page_list([x.encode('utf-8') for x in formatted_list])

    @command
    def embeddings(self):
        """
        List available and imported default embeddings as well as imported custom and random baselines.
    ―
        """
        ctx = context.get_context()
        debug = ctx.debug
        embedding_registry = ctx.embedding_registry
        formatted_list = commands.list_embeddings(debug, embedding_registry)

        page_list([x.encode('utf-8') for x in formatted_list])

    @command
    def cognitive_sources(self):
        """
        List CogniVal cognitive sources (must be imported)
    ―
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        cog_config = _open_cog_config(resources_path)

        if not cog_config['cognival_installed']:
            cprint('CogniVal cognitive sources not imported, aborting ...', 'red')
            return
        
        formatted_list = commands.list_cognitive_sources(cog_config)

        page_list([x.encode('utf-8') for x in formatted_list])


@command
class Config:
    """Generate or edit configuration files for experimental combinations (cog. data - embedding type)
    [Sub-commands]
    - open: Opens existing or creates new configuration edits its general properties.
    - properties: Edit general CogniVal properties (user directory, etc.).
    - show: Show details of a configuration and experiments.
    - experiment: Edit configuration of single, multiple or all combinations of embeddings
                  and cognitive sources of specified configuration. Populates default values from reference configuration.
    - delete: Remove cognitive sources or experiments (cog.source - embedding combinations) from specified configuration
              or delete entire configuration.
―
    """

    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command
    @argument('configuration', type=str, description='Name of configuration', positional=True)
    @argument('edit', type=bool, description='Open editor for general configuration properties.')
    @argument('overwrite', type=bool, description='Overwrite configuration (if already existing).')
    def open(self, configuration, edit=False, overwrite=False):
        '''
        Opens configuration or creates empty configuration file from template.
―
        '''
        ctx = context.get_context()
        cognival_path = ctx.cognival_path
        resources_path = ctx.resources_path
        
        try:
            configuration, _ = commands.config_open(configuration, cognival_path, resources_path, edit, overwrite)
        except TypeError:
            return

        if configuration:
            cprint('Configuration {} is now active.'.format(configuration), 'yellow')
            ctx.open_config = configuration
    
    @command
    @argument("details", type=bool, description="Whether to show details for all cognitive sources. Ignored when cognitive_source is specified.")
    @argument("cognitive_source", type=str, description="Cognitive source for which details should be shown")
    @argument("hide_baselines", type=bool, description="Hide random baselines from word embedding specifics")
    def show(self, details=False, cognitive_source=None, hide_baselines=True):
        '''
        Display an overview for the given configuration.
    ―
        '''
        ctx = context.get_context()
        resources_path = ctx.resources_path
        configuration = ctx.open_config
        if not configuration:
            cprint('No configuration open, aborting ...', 'red')
            return

        config_dict = _open_config(configuration, resources_path, quiet=True, protect_reference=False)
        if not config_dict:
            return

        table_strs = commands.config_show(configuration, config_dict, details, cognitive_source, hide_baselines)

        page_list([x.encode('utf-8') for x in table_strs])
    
    @command
    @argument('modalities', type=list, description="Modalities of cognitive sources sources to include.")
    @argument('cognitive_sources', type=list, description="Either list of cognitive sources or ['all'] (default).")
    @argument('embeddings', type=list, description="Either list of embeddings or ['all'] (default)")
    @argument('baselines', type=bool, description='Include random baselines. Note that if random baselines were included previously, changes are applied to them in any case.')
    @argument('single_edit', type=bool, description='Whether to edit embedding specifics one by one or all at once.')
    @argument('edit_cog_source_params', type=bool, description='Whether to edit parameters of the specified cognitive sources.')
    @argument('scope', type=str, description="Specifies the scope for meta-parameters and -arguments ('all', modalities). "
                                             "Either 'all' (all installed embeddings/cog. sources) or 'config' (configuration only). "
                                             "In the latter case, no automatic insertion of missing sources occurs. If set to None (default), "
                                             "the scope is 'all' if the configuration is empty and 'config' after adding the first source-embedding pair(s).")
    def experiment(self,
                   baselines=True,
                   modalities=None,
                   cognitive_sources=['all'],
                   embeddings=['all'],
                   single_edit=False,
                   edit_cog_source_params=False,
                   scope=None):
        '''
        Edit configuration of single, multiple or all combinations of embeddings and cognitive sources.
        '''
        ctx = context.get_context()
        configuration = ctx.open_config
        if not configuration:
            cprint('No configuration open, aborting ...', 'red')
            return

        embedding_registry = ctx.embedding_registry
        resources_path = ctx.resources_path

        main_conf_dict = _open_config(configuration, resources_path)
        cog_data_config_dict = _open_cog_config(resources_path)
        try:
            commands.config_experiment(configuration,
                                   main_conf_dict,
                                   cog_data_config_dict,
                                   embedding_registry,
                                   resources_path,
                                   baselines,
                                   modalities,
                                   cognitive_sources,
                                   embeddings,
                                   single_edit,
                                   edit_cog_source_params,
                                   scope)

        except NothingToDoException:
            cprint("Nothing to do. If the configuration is populated, pass scope=all to add embeddings and cognitive-sources in bulk.", "yellow")


    @command
    @argument('modalities', type=list, description="Modalities of cognitive sources to delete.")
    @argument('cognitive_sources', type=list, description="Either list of cognitive sources or None (for all)")
    @argument('embeddings', type=list, description="Either list of embeddings or None (for all)")
    def delete(self, modalities=None, cognitive_sources=None, embeddings=None):        
        '''
        Remove cognitive sources or experiments (cog.source - embedding combinations) from specified configuration or
        delete entire configuration.
        '''
        ctx = context.get_context()
        configuration = ctx.open_config
        if not configuration:
            cprint('No configuration open, aborting ...', 'red')
            return
        embedding_registry = ctx.embedding_registry
        resources_path = ctx.resources_path

        main_conf_dict = _open_config(configuration, resources_path)
        cog_config_dict = _open_cog_config(resources_path)

        main_conf_dict = commands.config_delete(configuration,
                                                main_conf_dict,
                                                cog_config_dict,
                                                embedding_registry,
                                                modalities,
                                                cognitive_sources,
                                                embeddings)

        _backup_config(configuration, resources_path)
        
        if main_conf_dict:
            _save_config(main_conf_dict, configuration, resources_path)


@command
@argument('run_id', type=int, description='Run ID to be aggregated. Defaults to 0, treated as last run (run_id - 1).')
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('alpha', type=str, description='Alpha value')
@argument('test', type=str, description='Significance test')
def significance(run_id=0,
                 modalities=['eye-tracking', 'eeg', 'fmri'],
                 alpha=0.01,
                 test='Wilcoxon',
                 quiet=False):
    '''
    Test significance of results in the given modality and produced based on the specified configuration.
―
    '''
    ctx = context.get_context()
    configuration = ctx.open_config
    if not configuration:
        cprint('No configuration open, aborting ...', 'red')
        return
    resources_path = ctx.resources_path

    config_dict = _open_config(configuration, resources_path, quiet=quiet)
    if not config_dict:
        return

    list(commands.significance(configuration,
                          config_dict,
                          run_id,
                          modalities,
                          alpha,
                          test,
                          quiet))


@command
@argument('run_id', type=int, description='Run ID to be aggregated. Defaults to 0, treated as last run (run_id - 1).')
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('test', type=str, description='Significance test')
def aggregate(run_id=0,
              modalities=['eye-tracking', 'eeg', 'fmri'],
              test="Wilcoxon",
              quiet=False):    
    '''
    Test significance of results in the given modality and produced based on the specified configuration.
―
    '''
    ctx = context.get_context()
    configuration = ctx.open_config
    if not configuration:
        cprint('No configuration open, aborting ...', 'red')
        return
    resources_path = ctx.resources_path
    config_dict = _open_config(configuration, resources_path, quiet=quiet)

    if not config_dict:
        return

    list(commands.aggregate(configuration,
                       config_dict,
                       run_id,
                       modalities,
                       test,
                       quiet))


@command
class Update:
    """Update CogniVal resources (vocabulary, sentences and embeddings)
    [Sub-commands]
    - vocabulary: Update the vocabulary based on all imported cognitive sources.
    - sentences: Update the CogniVal sentence listbased on all imported cognitive sources.
    - listings: Update both the CogniVal vocabulary and sentence list based on all imported cognitive_sources.
    - embeddings: Update embeddings that are generated in CogniVal (BERT, ELMo), average sentence embeddings and random embeddings.
―
    """

    def __init__(self) -> None:
        pass

    """This is the super command help"""

    @command
    def vocabulary(self):
        """
        Update the vocabulary based on all imported cognitive sources.
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        cog_sources_path = ctx.cog_sources_path
        vocab_path = resources_path / 'standard_vocab.txt'

        with open(vocab_path) as f:
            old_vocab = [x.rstrip('\n') for x in f]

        new_vocab_list = commands.update_vocabulary(resources_path,
                                                    cog_sources_path,
                                                    old_vocab)

        if new_vocab_list:
            cprint('Writing vocabulary to {} ...'.format(str(vocab_path)), 'green')
            with open(vocab_path, 'w') as f:
                for word in new_vocab_list:
                    if word:
                        f.write('{}\n'.format(word))

    @command
    def sentences(self):
        """
        Update the CogniVal sentence listbased on all imported cognitive sources.
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        cog_sources_path = ctx.cog_sources_path
        sentences_path = resources_path / 'standard_sentences.txt'
        sent_vocab_path = resources_path / 'standard_sent_vocab.txt'    

        with open(sentences_path) as f:
            old_sentences = [x.rstrip('\n') for x in f]

        new_sentences, new_vocab_list = commands.update_sentences(resources_path,
                                                                  cog_sources_path,
                                                                  old_sentences)

        if new_sentences:
            cprint('Writing sentences to {} ...'.format(str(sentences_path)), 'green')
            with open(sentences_path, 'w') as f:
                for sent in new_sentences:
                    if sent:
                        f.write('{}\n'.format(sent))

            cprint('Writing sentence vocabulary to {} ...'.format(str(sent_vocab_path)), 'green')
            with open(sent_vocab_path, 'w') as f:
                for word in new_vocab_list:
                    if word:
                        f.write('{}\n'.format(word))
     
    @command
    def listings(self):
        """
        Update both the CogniVal vocabulary and sentence list based on all imported cognitive_sources.
        """
        self.vocabulary()
        self.sentences()


    @command
    @argument('embeddings', type=list, description="Either list of embeddings or None (all embeddings, default).")
    @argument('which', type=str, description="Whether to update 'word', 'sentence' or 'both' (default) embeddings.")
    def embeddings(self, embeddings=None, which='both'):
        """
        Update embeddings that are generated in CogniVal (BERT, ELMo), average sentence embeddings and random embeddings.
        """
        ctx = context.get_context()
        embeddings_path = ctx.embeddings_path
        resources_path = ctx.resources_path
        embedding_registry = ctx.embedding_registry

        embedding_registry = commands.update_embeddings(resources_path, embeddings_path, embedding_registry, embeddings=embeddings, which=which)

        if embedding_registry:
            ctx.save_configuration()


@command
class Import:
    """Import CogniVal cognitive vectors, default embeddings, custom embeddings and
    generate random baselines.
    [Sub-commands]
    - cognitive-sources: Import the entire batch of preprocessed CogniVal and other cognitive sources.
    - embeddings: Download and import a default embedding (by name) or custom embedding (from URL)
    - random-baselines: Generate and import random baselines for specified embeddings.
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
        cog_config = _open_cog_config(resources_path)

        cog_config = commands.import_cognitive_sources(cognival_path,
                                                       resources_path,
                                                       cog_config,
                                                       source)

        if cog_config:
            _save_cog_config(cog_config, resources_path)

    @command()
    @argument('x', type=str, description='Force removal and download', positional=True) #choices=list(CTX.embedding_registry['proper']))
    @argument('force', type=bool, description='Force removal and download')
    def embeddings(self, x, force=False, which='both', log_only_success=False, are_set=False, associate_rand_emb=None):
        """
        Download and import a default embedding (by name) or custom embedding (from URL)
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        embeddings_path = ctx.embeddings_path
        embedding_registry = ctx.embedding_registry
        path2embeddings = ctx.path2embeddings
        debug = ctx.debug

        embedding_registry = commands.import_embeddings(x,
                                         which,
                                         embedding_registry,
                                         path2embeddings,
                                         resources_path,
                                         embeddings_path,
                                         force,
                                         log_only_success,
                                         are_set,
                                         associate_rand_emb,
                                         debug)

        if embedding_registry:
            ctx.save_configuration()


    @command()
    @argument('embeddings',
            type=str,
            description='Name of embeddings that have been registered (not necessarily imported).',
            positional=True)
    @argument('num_baselines',
            type=int,
            description='Number of random baselines to be generated (and across which performance will later be averaged).')
    @argument('seed_func',
            type=str,
            description='Seed generation function. Currently only "exp_e_floored" (np.floor((k+i)**np.e))) supported.')
    @argument('force',
            type=str,
            description='Force regeneration and association of random baselines with specified embeddings.')
    def random_baselines(self, embeddings, num_baselines=10, seed_func='exp_e_floored', force=False):
        """
        Generate and import random baselines for specified embeddings.
    ―
        """
        ctx = context.get_context()
        resources_path = ctx.resources_path
        embeddings_path = ctx.embeddings_path
        embedding_registry = ctx.embedding_registry

        embedding_registry = commands.import_random_baselines(embedding_registry,
                                                              resources_path,
                                                              embeddings_path,
                                                              embeddings,
                                                              num_baselines,
                                                              seed_func,
                                                              force)

        if embedding_registry:
            ctx.save_configuration()


### Direct commands

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
@argument('write_enabled',
          type=str,
          description='Enable manual editing of user files (Caution!)')
def browse(write_enabled=False):
    '''
    Browses the user directory and view files using vim, per default in read-only mode. (requires that vim is installed).
―
    '''
    ctx = context.get_context()
    cognival_path = ctx.cognival_path
    if write_enabled:
        command = ['vim', cognival_path]
    else:
        command = ['vim', '-RM', cognival_path]
    subprocess.run(command)


@command
def properties():
    '''
    Edit general CogniVal properties (user directory, etc.).
    '''
    # Creating config.json (initial run)
    installation_path = Path(os.path.dirname(__file__)) / '..' / '..'
    config_path = installation_path / 'config.json'
    with open(config_path , 'r') as f:
        config_dict = json.load(f)

    config_patch = {}

    conf_editor = ConfigEditor('properties',
                                config_dict,
                                config_patch,
                                singleton_params='all')
    conf_editor()
    if config_patch:
        config_dict.update(config_patch)
        with open(config_path , 'w') as f:
            json.dump(config_dict, f)
        cprint('Saved CogniVal properties, please restart the interactive shell. Quitting ...', 'green')
        sys.exit(1)


@command
@argument('run_id', type=int, description='Run ID for which to generate a report. Defaults to 0, treated as last run (run_id - 1).')
@argument('modalities', type=str, description='Modalities for which significance is to be termined (default: all applicable)')
@argument('alpha', type=str, description='Alpha value')
@argument('test', type=str, description='Significance test')
@argument('precision', type=int, description='Number of decimal points in report (except for bonferroni alpha)')
@argument('average_multi_hypothesis', type=bool, description='Average multi-hypothesis (multi-feature or multi-subject) results.')
@argument('history_plots', type=bool, description='Whether to include training history plots (note: significantly increases report size when number of experiments is large.')
@argument('features', type=bool, description='Whether to include the feature column in detail tables.')
@argument('heatmaps', type=bool, description='Whether to include word/sentence to embedding per-source error heatmaps')
@argument('heatmaps_sample_n', type=int, description='If not None (default), randomly sample n words/sentences. Show all otherwise.')
@argument('heatmaps_discard_na', type=bool, description='If True (default: False), only keep rows with values for all embeddings. Occurs *before* sampling.')
@argument('export_err_tables', type=bool, description='If True and heatmaps == True, export error DataFrames (basis of heatmap tables) as gzipped parquet to report directory. Requires pyarrow or fastparquet. Sampling and discarding of NaNs does not affect export.')
@argument('html', type=bool, description='Generate html report.')
@argument('open_html', type=bool, description='Open generated html report.')
@argument('pdf', type=bool, description='Generate pdf report.')
@argument('open_pdf', type=bool, description='Open generated pdf report.')
def report(run_id=0,
           modalities=['eye-tracking', 'eeg', 'fmri'],
           alpha=0.01,
           test="Wilcoxon",
           precision=3,
           average_multi_hypothesis=True,
           history_plots=False,
           features=True,
           heatmaps=False,
           heatmaps_sample_n=None,
           heatmaps_discard_na=False,
           export_err_tables=False,
           html=True,
           open_html=False,
           pdf=False,
           open_pdf=False):
    '''
    Compute significance tests, aggregate results and generate report
―
    '''
    ctx = context.get_context()
    configuration = ctx.open_config
    if not configuration:
        cprint('No configuration open, aborting ...', 'red')
        return
    resources_path = ctx.resources_path

    cprint('Computing significance stats ...', 'yellow')
    significance(run_id, modalities, alpha, test, quiet=True)
    cprint('Aggregating ...', 'yellow')
    aggregate(run_id, modalities, test, quiet=True)
    generate_report(configuration,
                    run_id,
                    resources_path,
                    precision,
                    average_multi_hypothesis,
                    history_plots,
                    features,
                    heatmaps,
                    heatmaps_sample_n,
                    heatmaps_discard_na,
                    export_err_tables,
                    html,
                    pdf,
                    open_html,
                    open_pdf)


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


@command
@argument('command',
        type=str,
        description='Command for which examples are to be shown. Examples for all commands shown if None.')
def example_calls(command=None):
    '''
    Lists example calls for a single or all commands.
―
    '''
    table_rows = []
    
    def get_table_rows(cmd, parametrization):
        if isinstance(parametrization, list):
            for example in parametrization:
                if example['example']:
                    example['command'] = cmd
                    example['example'] = fill(example['example'], 80).replace('[nl]', '\n\n')
                    example['subcommand'] = ''
                    example['description'] = fill(example['description'], 20)
                    table_rows.append(example)
        elif isinstance(parametrization, dict):
            for subcommand, subparametrization in parametrization.items():
                for example in subparametrization:
                    if example['example']:
                        example['command'] = cmd
                        example['example'] = fill(example['example'], 80).replace('[nl]', '\n\n')
                        example['subcommand'] = subcommand
                        example['description'] = fill(example['description'], 20)
                        table_rows.append(example)
        
    if not command:
        for command, parametrization in EXAMPLE_COMMANDS.items():
            get_table_rows(command, parametrization)
    else:
        get_table_rows(command, EXAMPLE_COMMANDS[command])
        
    df_cli = pd.DataFrame(table_rows)
    df_cli = df_cli[['command', 'subcommand', 'example', 'description']]
    df_cli.columns = [colored(col.title(), attrs=['bold']) for col in df_cli.columns]
    table_str = tabulate.tabulate(df_cli, headers="keys", tablefmt="fancy_grid", showindex=False)
    page_list([table_str.encode('utf-8')])

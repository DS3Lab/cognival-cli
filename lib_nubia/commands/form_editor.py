#!/usr/bin/env python3

# Derived from: TODO
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import asyncio
import collections
import csv
import copy
import itertools
import json
import gzip
import os
import sys
import requests
import signal
import socket
import shutil
import time
import typing
import zipfile

from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE

import gdown
from joblib import Parallel, delayed
from natsort import natsorted
from nubia import command, argument, context
import numpy as np
import pandas as pd

try:
    from allennlp.commands.elmo import ElmoEmbedder
except ImportError:
    cprint('Warning: Package allennlp not found. ELMo conversion unavailable.', 'yellow')

try:
    from bert_serving.client import BertClient
except ImportError:
    cprint('Warning: Package bert_serving.client not found. BERT conversion unavailable.', 'yellow')

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog, button_dialog, radiolist_dialog, ProgressBar
from prompt_toolkit.application.current import get_app
from termcolor import cprint

from cog_evaluate import run as run_serial
from cog_evaluate_parallel import run_parallel as run_parallel
from handlers.file_handler import write_results, update_version
from handlers.data_handler import chunk

from utils import generate_df_with_header, word2vec_bin_to_txt

#sys.path.insert(0, 'significance_testing/')
from significance_testing.statisticalTesting import extract_results as st_extract_results
from significance_testing.aggregated_eeg_results import extract_results as agg_eeg_extr_results
from significance_testing.aggregated_fmri_results import extract_results as agg_eeg_extr_results
from significance_testing.aggregated_gaze_results import extract_results_gaze as agg_gaze_extr_results
from significance_testing.testing_helpers import bonferroni_correction, test_significance

from lib_nubia.prompt_toolkit_table import *

# TODO: Make parametrizable
BINARY_CONVERSION_LIMIT = 1000
NUM_BERT_WORKERS = 1

_2D_FIELDS = set(['layers'])

EDITOR_TITLES = {"main": "General Configuration",
                 "cognitive": "Cognitive Source Configuration",
                 "embedding_exp": "Embedding Experiment Configuration",
                 "embedding_conf": "Embedding Parameter Configuration"}

EDITOR_DESCRIPTIONS = {"main": {"PATH": "Main working directory for all experiments. Set to application directory if empty",
                                "cpu_count": "Number of CPU cores used for execution",
                                "folds": "Number of folds evaluated in n-Fold cross-validation (CV)",
                                "outputDir": "Output directory",
                                "seed": "Random seed for train-test sampling",
                                "version": "Configuration version. Normally set to 1 when creating a new configuration file"
                                },
                       "cognitive": {"dataset": "Name of the cognitive data set. See resources/cognitive_sources.json for a list of all available source",
                                     "modality": "Cognitive source modality (eeg, eye-tracking or fmri)",
                                     "features": "Comma-separated list of features to be evaluated. Must be set to ALL_DIM for all sources with only one feature, represented by all dimensions.",
                                     "type": "'single_output' for most multi-feature resources (typically eye-tracking) and 'multivariate_output' for most single-feature resources (typicall eeg, fmri)"
                                    },
                       "embedding_exp": {"activations": "Activation function(s) used in the neural regression model. Comma-separated list for multiple values.",
                                         "batch_size": "Batch size used during training. Comma-separated list for multiple values.",
                                         "cv_split": "Number of cross-validation (CV) splits",
                                         "epochs": "Number of training epochs. Comma-separated list for multiple values.",
                                         "layers": "List of lists of layer specifications. Each row corresponds to possible layer sizes for a layer (comma-separated). Layers are separated by newlines.",
                                         "validation_split": "Ratio training data used as validation data during training"
                                        },
                        "embedding_conf":{"chunk_number": "Number of embedding chunks. Ignored if chunked == 0.",
                                         "chunked": "Whether the embedding is chunked (1) or not (0).",
                                         "chunked_file": "Prefix/root of chunk files. Ignored if chunked == 0.",
                                         "ending": "File suffix of chunk files. Ignored if chunked == 0.",
                                         "path": "Path of embedding (chunks)",
                                         "truncate_first_line": "Whether to remove the first line upon loading"
                                         }
                        }
                        # TODO: Why is truncate_first_line included?

class ConfigEditor():
    def __init__(self, conf_type, config_dict, config_dict_updated, singleton_params=None, skip_params=None, cognitive_sources=None, embeddings=None):
        self.buffers = {}
        self.config_dict_updated = config_dict_updated
        self.singleton_params = singleton_params if singleton_params else []
        self.skip_params = skip_params if skip_params else []
        self.table_fields = []
        
        # Add header information
        self.table_fields.append([Merge(Label("{} (Navigate with <Tab>/<Shift>-<Tab>)".format(EDITOR_TITLES[conf_type]), style="fg:ansigreen bold"), 2)])
        if cognitive_sources:
            self.table_fields.append([Merge(Label('Cognitive sources: {}'.format(cognitive_sources), style="fg:ansiyellow bold"), 2)])
        if embeddings:
            self.table_fields.append([Merge(Label('Embeddings: {}'.format(embeddings), style="fg:ansiyellow bold"), 2)])

        self.table_buttons = [[Button('Save', handler=self.save), Button('Abort', handler=self.abort)]]

        self.kb = KeyBindings()

        @self.kb.add('c-c')
        def _(event):
            " Abort when Control-C has been pressed. "
            event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

        @self.kb.add("tab")
        def _(event):
            event.app.layout.focus_next()

        @self.kb.add("s-tab")
        def _(event):
            event.app.layout.focus_previous()

        # Add row to editor table/form with label, buffer and description for each configuration item
        for k, v in config_dict.items():
            if k in self.skip_params:
                continue
            if v is None:
                v = ""
                style = "fg:ansiwhite"
            elif isinstance(v, (list, tuple)):
                if isinstance(v[0], (list, tuple)):
                    #TODO: Test this
                    v = "\n".join([", ".join([str(y) for y in x]) for x in v])
                else:
                    v = ", ".join([str(x) for x in v])
                style = "fg:ansigreen"
            elif v == '<multiple values>':
                style = "fg:ansimagenta italic"
            else:
                v = str(v)
                style = "fg:ansiwhite"

            buffer = TextArea(v, style=style)
            self.buffers[k] = buffer
            row = [Label(k, style="fg:ansicyan bold"), Label(EDITOR_DESCRIPTIONS[conf_type][k], style="italic")]
            self.table_fields.append(row)
            self.table_fields.append(Merge(buffer, 2))
        
        self.layout = Layout(
                        HSplit([
                            Table(
                                table=self.table_fields,
                                column_widths=[D(20, 20), D(80, 120)],
                                borders=RoundedBorder),
                            HorizontalLine(),
                            Table(
                                table=self.table_buttons,
                                column_widths=[D(20, 20), D(80, 120)],
                                borders=RoundedBorder)
                            ]
                        ),
                    )

    def __call__(self):
        return Application(self.layout, key_bindings=self.kb, full_screen=True).run()
    
    def abort(self):
        get_app().exit(result=True)

    def _cast_single(self, v):
        try:
            return int(v.text)
        except ValueError:
            try:
                return float(v.text)
            except ValueError:
                return v.text

    def _cast_list(self, v_list):
        if len(v_list) > 1:
            return [int(x) if x.isdigit() else x for x in v_list]
        else:
            try:
                return [int(v_list[0])]
            except ValueError:
                try:
                    return [float(v_list[0])]
                except ValueError:
                    return v_list

    def save(self):
        for k, v in self.buffers.items():
            v.text = v.text.strip()

            if v.text == '<multiple values>':
                continue

            if k in self.singleton_params or self.singleton_params == 'all':
                if ',' in v.text:
                    get_app().exit(result="Field '{}' does not support lists of values. "
                                          "Press Enter to reopen editor.".format(k))
                    return

                values = self._cast_single(v)
                
            else:
                if '\n' in v.text:
                    if k not in _2D_FIELDS:
                        get_app().exit(result="Field '{}' does not support multiple levels of nesting. "
                                              "Press Enter to reopen editor.".format(k))
                        return
                        
                    values_split = v.text.split('\n')
                    values_list = [v.replace(' ', '').split(',') for v in values_split]
                    values_list_cast = []
                    for v_list in values_list:
                        v_list = self._cast_list(v_list)
                        values_list_cast.append(v_list)
                    # TODO: Test this
                    values = values_list_cast
                else:
                    values_list = v.text.replace(' ', '').split(',')    
                    values = self._cast_list(values_list)

            self.config_dict_updated[k] = values

        get_app().exit(result=True)

def config_editor(conf_type,
                  config_dict,
                  embeddings,
                  cognitive_sources,
                  singleton_params=None,
                  skip_params=None):
    embeddings = ", ".join(embeddings)
    cognitive_sources = ", ".join(cognitive_sources)

    config_dict_updated = {}

    result = None
    
    while True:
        conf_editor = ConfigEditor(conf_type,
                                config_dict,
                                config_dict_updated,
                                singleton_params=singleton_params,
                                cognitive_sources=cognitive_sources,
                                embeddings=embeddings,
                                skip_params=skip_params)
        result = conf_editor()
        if result is True:
            break
        else:
            cprint('Error: {}'.format(result), 'red')
            prompt()
            result = None

    return config_dict_updated
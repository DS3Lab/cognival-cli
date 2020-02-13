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

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog, button_dialog, radiolist_dialog, ProgressBar
from termcolor import cprint

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

def bert_to_text(vocabulary_file, model_dir, output_path, dimensions, num_worker):
    # Source: https://github.com/hanxiao/bert-as-service
    # Start Bert Model: bert-serving-start -model_dir /tmp/uncased_L-12_H-768_A-12/ -num_worker=4
    # model: 'base' (768 dimensions) or 'large' (1024 dimensions)
   
    # Create tmp directory if non-existent
    try:
        os.makedirs('tmp')
    except FileExistsError:
        pass

    print('Starting BERT service server ...')
    proc = Popen(['bert-serving-start',
                  '-model_dir={}'.format(os.path.abspath(model_dir)),
                  '-num_worker={}'.format(num_worker)],
                  stdout=PIPE,
                  stderr=PIPE,
                  preexec_fn=os.setsid,
                  cwd='tmp')

    with open(vocabulary_file, 'r') as f:
        words = f.readlines()

    # Create directory
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'w') as embedding_file:
        # Load pre-trained model (weights)
        with BertClient(ignore_all_checks=True) as bc:

            count_not_found = 0
            words_stripped = [word.strip() for word in words]
            print('Obtaining per-word embedding ...')
            output_embedding = bc.encode(words_stripped)
            for idx, word in enumerate(words_stripped):
                embedding = ' '.join(map(str, output_embedding[idx]))

                print(word, embedding, file=embedding_file)

            print(count_not_found, ' words not found.')

    # Terminate BERT server
    print("Terminating BERT server ...")
    os.killpg(proc.pid, signal.SIGTERM)

    # Wait until processes terminated
    while proc.poll() is None:
        time.sleep(0.05)

    print("Waiting for file lock release ...")
    time.sleep(5)

    print("Removing temporary directory")
    shutil.rmtree('tmp')


def elmo_to_text(vocabulary_file, output_path, layer='nocontext'):
    """
    :param vocabulary_file: Vocabulary file. Note that usually no vocabulary file is provided with ELMo embeddings.
    :param output_path: Output file path
    :param layer: Either 'full' which equals to full Elmo after second biLSTM layer or
                  'nocontext' (context-insensitive)
    
    (Reused from original CogniVal paper)
    """
    if layer == 'full':
        layer_idx = 2
    elif layer == 'nocontext':
        layer_idx = 0
    else:
        raise ValueError('"layer" must be either "full" or "nocontext"')
    
    elmo = ElmoEmbedder()

    with open(vocabulary_file, 'r') as f:
        words = f.readlines()

    # Create directory
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'w') as embedding_file:
        with ProgressBar() as pb:
            for word in pb(words):
                word = word.strip()
                # print(word)
                vectors = elmo.embed_sentence(word)

                # context insensitive - first layer
                embedding = ' '.join(map(str, vectors[layer_idx][0]))
                print(word, embedding, file=embedding_file)

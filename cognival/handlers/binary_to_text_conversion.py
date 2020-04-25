#!/usr/bin/env python3

# Derived from: TODO
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import signal
import shutil
import time

from subprocess import Popen, PIPE
from termcolor import cprint

try:
    from allennlp.commands.elmo import ElmoEmbedder
except ImportError:
    cprint('Warning: Package allennlp not found. ELMo conversion unavailable.', 'yellow')

try:
    from bert_serving.client import BertClient
except ImportError:
    cprint('Warning: Package bert_serving.client not found. BERT conversion unavailable.', 'yellow')

from prompt_toolkit.shortcuts import ProgressBar, message_dialog
from termcolor import cprint

from lib_nubia.prompt_toolkit_table import *

def bert_to_text(vocabulary_file, base_dir, model_dir, output_path, num_worker):
    # Source: https://github.com/hanxiao/bert-as-service
    # Start Bert Model: bert-serving-start -model_dir /tmp/uncased_L-12_H-768_A-12/ -num_worker=4
    # model: 'base' (768 dimensions) or 'large' (1024 dimensions)

    message_dialog(title='BERT binary conversion to word embeddings',
                   text='BERT conversion requires that the server component of bert-as-service is running. Please execute\n'
                        'the following command in a separate terminal and wait until the server reports "all set, ready to serve request!".\n'
                        'Note that bert-as-service creates temporary files in the working directory the command is executed in, which\n'
                        'must be removed manually after termination. The number of workers may be increased depending on available resources \n'
                        'and ulimit restrictions.\n\n'
                        'bert-serving-start -model_dir={} -num_worker=1'.format(str(base_dir))).run()


    with open(vocabulary_file, 'r') as f:
        words = f.readlines()

    # Create directory if non-existent
    os.makedirs(output_path.parent, exist_ok=True)

    # Remove file if existent
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass

    with open(output_path, 'w') as embedding_file:
        # Load pre-trained model (weights)
        with BertClient() as bc:

            count_not_found = 0
            print('Obtaining per-word embedding ...')
            with ProgressBar() as pb:
                for word in pb(words):
                    word = word.strip()
                    output_embedding = bc.encode([word])
                    embedding = ' '.join(map(str, output_embedding[0]))
                    print(word, embedding, file=embedding_file)

            print(count_not_found, ' words not found.')
    
    message_dialog(title='BERT binary conversion to word embeddings',
                   text='BERT conversion complete. bert-as-service may now be safely terminated.').run()


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

    # Create directory if non-existent
    os.makedirs(output_path.parent, exist_ok=True)

    # Remove file if existent
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass

    with open(output_path, 'w') as embedding_file:
        with ProgressBar() as pb:
            for word in pb(words):
                word = word.strip()
                vectors = elmo.embed_sentence(word)

                # context insensitive - first layer
                embedding = ' '.join(map(str, vectors[layer_idx][0]))
                print(word, embedding, file=embedding_file)

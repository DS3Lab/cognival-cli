#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import collections
import json
import os
import sys
from pathlib import Path
from termcolor import cprint

from natsort import natsorted
from nubia import context
from nubia import exceptions
from nubia import eventbus

class NubiaCognivalContext(context.Context):
    def __init__(self, *args, **kwargs):
        self.gpu_ids = None
        self._gpu_ids_all = None
        self.state = {}
        self.messages = kwargs.get('messages', None)
        self.cognival_path = kwargs.get('cognival_path', None)
        
        if self.cognival_path:
            self.cog_sources_path = self.cognival_path / 'cognitive_sources'
            self.embeddings_path = self.cognival_path / 'embeddings'
            self.configurations_path = self.cognival_path / 'configurations'
            self.results_path = self.cognival_path / 'results'
        else:
            self.cognival_path, self.embeddings_path, \
                self.configurations_path, self.results_path = [None] * 4

        self.embedding_registry = None
        self.open_config = None
        self.path2embeddings = collections.defaultdict(list)
        self._load_configuration()
        super().__init__()

    def _load_configuration(self):
        if self.configurations_path:
            try:
                with open(self.configurations_path / 'embedding_registry.json') as f:
                    self.embedding_registry = json.load(f)
            
                for emb_category, emb_category_dict in self.embedding_registry.items():
                    if emb_category == 'random_multiseed':
                        for emb_type, emb_type_dict in emb_category_dict.items():
                            for embeddings, embedding_params in emb_type_dict.items():
                                self.path2embeddings[Path(embedding_params['path']) / emb_type].append(embeddings)
                    else:
                        for embeddings, embedding_params in emb_category_dict.items():
                            self.path2embeddings[embedding_params['path']].append(embeddings)
                for path, emb_list in self.path2embeddings.items():
                    self.path2embeddings[path] = natsorted(emb_list)
            except FileNotFoundError:
                cprint('Cannot open embedding registry, please correct user directory path (command "properties")', 'red')
        else:
            cprint('Error: Could not load resources path, aborting ...', 'red')
            sys.exit(1)

    def _set_gpu_ids(self, args):
        self.visible_gpus = args.visible_gpus.replace(' ', '')
        self.max_gpus = args.max_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if self.visible_gpus:
            gpu_ids_str = self.visible_gpus
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str  # for several GPUs
        self.visible_gpus = list(map(int, self.visible_gpus.split(',')))

    def on_connected(self, *args, **kwargs):
        pass

    def on_cli(self, cmd, args):
        # dispatch the on connected message
        self.debug = args.debug
        self.verbose = args.verbose
        self._set_gpu_ids(args)

        self.registry.dispatch_message(eventbus.Message.CONNECTED)

    def on_interactive(self, args):
        self.debug = args.debug
        self.verbose = args.verbose
        self.no_welcome = args.no_welcome
        self._set_gpu_ids(args)

        if not self.no_welcome:
            cprint(self.messages.LOGO_STR, "magenta")
            cprint(self.messages.WELCOME_MESSAGE_STR)
        
        ret = self._registry.find_command("connect").run_cli(args)
        if ret:
            raise exceptions.CommandError("Failed starting interactive mode")
        # dispatch the on connected message
        self.registry.dispatch_message(eventbus.Message.CONNECTED)


    def save_configuration(self):
        if self.embedding_registry:
            with open(self.configurations_path / 'embedding_registry.json', 'w') as f:
                json.dump(self.embedding_registry, f, indent=4)
        else:
            raise RuntimeError("No configuration loaded, cannot save!")

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import collections
import json
from termcolor import cprint

from natsort import natsorted
from nubia import context
from nubia import exceptions
from nubia import eventbus

class NubiaCognivalContext(context.Context):
    def __init__(self, *args, **kwargs):
        self.state = {}
        self.embedding_registry_path = kwargs.get('embedding_registry_path', None)
        self.embedding_registry = None
        self.path2embeddings = collections.defaultdict(list)
        self._load_configuration()
        super().__init__()

    def _load_configuration(self):
        if self.embedding_registry_path:
            with open(self.embedding_registry_path) as f:
                self.embedding_registry = json.load(f)
            for embedding_type_dict in self.embedding_registry.values():
                for embeddings, embedding_params in embedding_type_dict.items():
                    self.path2embeddings[embedding_params['path']].append(embeddings)
            for path, emb_list in self.path2embeddings.items():
                self.path2embeddings[path] = natsorted(emb_list)
        else:
            cprint('Warning: Could not load mapping from standard embeddings to URLs. Download functionality not available.', 'red')
        
    def on_connected(self, *args, **kwargs):
        pass

    def on_cli(self, cmd, args):
        # dispatch the on connected message
        self.verbose = args.verbose
        self.registry.dispatch_message(eventbus.Message.CONNECTED)

    def on_interactive(self, args):
        self.verbose = args.verbose
        self.debug = args.debug
        ret = self._registry.find_command("connect").run_cli(args)
        if ret:
            raise exceptions.CommandError("Failed starting interactive mode")
        # dispatch the on connected message
        self.registry.dispatch_message(eventbus.Message.CONNECTED)

    def save_configuration(self):
        if self.embedding_registry:
            with open(self.embedding_registry_path, 'w') as f:
                json.dump(self.embedding_registry, f, indent=4)
        else:
            raise RuntimeError("No configuration loaded, cannot save!")
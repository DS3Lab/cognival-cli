#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from termcolor import cprint

from nubia import context
from nubia import exceptions
from nubia import eventbus

class NubiaCognivalContext(context.Context):
    def __init__(self, *args, **kwargs):
        self.state = {}
        self.embedding2url_path = kwargs.get('embedding2url_path', None)
        self.embedding2url = None
        self._load_configuration()
        super().__init__()

    def _load_configuration(self):
        if self.embedding2url_path:
            with open(self.embedding2url_path) as f:
                self.embedding2url = json.load(f)
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
        ret = self._registry.find_command("connect").run_cli(args)
        if ret:
            raise exceptions.CommandError("Failed starting interactive mode")
        # dispatch the on connected message
        self.registry.dispatch_message(eventbus.Message.CONNECTED)

    def save_configuration(self):
        if self.embedding2url:
            with open(self.embedding2url_path, 'w') as f:
                json.dump(self.embedding2url, f, indent=4)
        else:
            raise RuntimeError("No configuration loaded, cannot save!")
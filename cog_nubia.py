#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import traceback
import warnings
import sys

import numpy as np

from nubia import Nubia, Options
from lib_nubia.nubia_plugin import NubiaCognivalPlugin
from lib_nubia import commands

warnings.filterwarnings("error", category=RuntimeWarning)

# Source: https://stackoverflow.com/a/22376126
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

if __name__ == "__main__":
    plugin = NubiaCognivalPlugin(embedding_registry_path='resources/embedding_registry.json')
    print("Launching interactive shell, please wait ...")
    shell = Nubia(
        name="cog_nubia",
        command_pkgs=commands,
        plugin=plugin,
        options=Options(persistent_history=True),
    )
    # Clear screen
    print("\033c")
    sys.exit(shell.run())
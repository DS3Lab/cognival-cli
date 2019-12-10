#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from nubia import Nubia, Options
from lib_nubia.nubia_plugin import NubiaCognivalPlugin
from lib_nubia import commands

if __name__ == "__main__":
    plugin = NubiaCognivalPlugin(embedding2url_path='resources/embedding2url.json')
    shell = Nubia(
        name="cog_nubia",
        command_pkgs=commands,
        plugin=plugin,
        options=Options(persistent_history=True),
    )
    sys.exit(shell.run())
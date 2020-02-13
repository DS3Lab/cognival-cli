#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
import pprint
import os
import json
import traceback
import warnings
import sys

import numpy as np
from pathlib import Path

from nubia import Nubia, Options

# Disable HDF5 file locking for ELMo/allennlp
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from lib_nubia.nubia_plugin import NubiaCognivalPlugin
from lib_nubia import commands
from lib_nubia.commands import messages

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras
from termcolor import cprint

import tensorflow as tf
import GPUtil

#warnings.filterwarnings("error", category=RuntimeWarning)

# Source: https://stackoverflow.com/a/22376126
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

#warnings.showwarning = warn_with_traceback
 
def main():
    # Creating config.json (initial run)
    installation_path = Path(os.path.dirname(__file__))
    config_path = installation_path / 'config.json'
    
    if not os.path.exists(config_path):
        cprint('No configuration file found, creating ...', 'green')
        usr_home = Path.home()
        cognival_usr_dir = usr_home / '.cognival'
        config_dict = {'cognival_path': str(cognival_usr_dir)}
        with open(config_path , 'w') as f:
            json.dump(config_dict, f)
    else:
        with open(config_path , 'r') as f:
            config_dict = json.load(f)

    cognival_path = config_dict['cognival_path']

    plugin = NubiaCognivalPlugin(cognival_path=cognival_path,
                                 messages=messages)
    cprint("Launching CogniVal, please wait ...", "magenta")
    shell = Nubia(
        name="cognival",
        command_pkgs=commands,
        plugin=plugin,
        options=Options(persistent_history=True),
    )
    # Clear screen
    print("\033c")
    sys.exit(shell.run())

if __name__ == "__main__":
    main()

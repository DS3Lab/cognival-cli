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

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras
import pprint
import os
from termcolor import cprint

import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.compat.v1.keras.backend import set_session

import GPUtil

#warnings.filterwarnings("error", category=RuntimeWarning)

# Source: https://stackoverflow.com/a/22376126
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

#warnings.showwarning = warn_with_traceback

NO_NVIDIA_GPUS = 'Note: No NVIDIA graphics cards found, leaving tensorflow at default settings for CPU-only computation.'

def get_available_processing_units(unit):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == unit]


def configure_tf_devices(gpu_ids=None, num_utilize_cpus=None, num_utilize_gpus=None):
    gpu_ids = ', '.join([str(x) for x in gpu_ids])
    
    # rattle shared GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids  # for several GPUs

    try:
        deviceIDs = GPUtil.getAvailable(order='load',
                                        limit=10,
                                        maxLoad=0.5,
                                        maxMemory=0.5,
                                        includeNan=False,
                                        excludeID=[],
                                        excludeUUID=[])
    except ValueError:
        cprint(NO_NVIDIA_GPUS, 'yellow')
        return

    if not deviceIDs:
        cprint(NO_NVIDIA_GPUS, 'yellow')
        return        

    shared_gpus = set(deviceIDs).intersection(set([0,1]))

    if shared_gpus:
        deviceID = shared_gpus.pop()
    else:
        deviceID = deviceIDs[0]

    if not deviceIDs:
        cprint("Error: Currently, no GPU is eligible (available memory and load at <=50%)", "red")
        GPUtil.showUtilization()
    else:
        cprint("Elected GPU #{} for computation.".format(deviceID), "green")

    num_available_gpus = min(len(get_available_processing_units('GPU')), num_utilize_cpus)
    num_available_cpus = min(len(get_available_processing_units('CPU')), num_utilize_gpus)

    print("Num GPUs allocated: ", num_available_gpus)
    print("Num CPUs allocated: ", num_available_cpus)

    config = tf.compat.v1.ConfigProto(device_count={'GPU': num_available_gpus , 'CPU': num_available_cpus}) 
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    config.gpu_options.visible_device_list = str(deviceID)
    sess = tf.compat.v1.Session(config=config) 
    set_session(sess)
    cprint("Configuration:", 'magenta')
    pprint.pprint(sess._config)
    


if __name__ == "__main__":
    configure_tf_devices(gpu_ids=list(range(7)),
                         num_utilize_cpus=1,
                         num_utilize_gpus=1)

    plugin = NubiaCognivalPlugin(embedding_registry_path='resources/embedding_registry.json')
    cprint("Launching interactive shell, please wait ...", "magenta")
    shell = Nubia(
        name="cog_nubia",
        command_pkgs=commands,
        plugin=plugin,
        options=Options(persistent_history=True),
    )
    # Clear screen
    print("\033c")
    sys.exit(shell.run())
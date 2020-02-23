#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from pygments.token import Token

from nubia import context
from nubia import statusbar


class NubiaCognivalStatusBar(statusbar.StatusBar):
    def __init__(self, context):
        self._last_status = None

    def get_rprompt_tokens(self):
        if self._last_status:
            return [(Token.RPrompt, "Error: {}".format(self._last_status))]
        return []

    def set_last_command_status(self, status):
        self._last_status = status

    def get_tokens(self):
        spacer = (Token.Spacer, "  ")
        barlist = [
            (Token.Toolbar, "CogniVal"),
            spacer,
            (Token.Toolbar, "Interactive Mode"),
        ]

        config = context.get_context().open_config
        if config:
            open_config = (Token.Info, config)
            barlist.extend([spacer,
                            (Token.Toolbar, "Configuration:"),
                            spacer,
                            open_config])

        return barlist

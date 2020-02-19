#!/usr/bin/env python3

# Derived from: TODO
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json

from termcolor import cprint

from prompt_toolkit import prompt
from prompt_toolkit.application.current import get_app

from lib_nubia.prompt_toolkit_table import *
from lib_nubia.commands.strings import *

# TODO: Make parametrizable
BINARY_CONVERSION_LIMIT = 1000
NUM_BERT_WORKERS = 1

_2D_FIELDS = set(['layers'])

# Adapted from: https://stackoverflow.com/a/42033176
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False

class ConfigEditor():
    def __init__(self,
                 conf_type,
                 config_dict,
                 config_dict_updated,
                 singleton_params=None,
                 skip_params=None,
                 cognitive_sources=None,
                 embeddings=None,
                 prefill_fields=None):
        self.buffers = {}
        self.config_dict_updated = config_dict_updated
        self.singleton_params = singleton_params if singleton_params else []
        self.skip_params = skip_params if skip_params else []
        self.table_fields = []
        if not prefill_fields:
            self.prefill_fields = {}
        else:
            self.prefill_fields = prefill_fields
        
        # Add header information
        self.table_fields.append([Merge(Label("{} (Navigate with <Tab>/<Shift>-<Tab>)".format(EDITOR_TITLES[conf_type]), style="fg:ansigreen bold"), 2)])
        if cognitive_sources:
            if len(cognitive_sources) > 1:
                cog_source_label = 'Cognitive sources: {}'
            else:
                cog_source_label = 'Cognitive source: {}'
            cognitive_sources = ", ".join(cognitive_sources)
            self.table_fields.append([Merge(Label(cog_source_label.format(cognitive_sources), style="fg:ansiyellow bold"), 2)])
        if embeddings:
            if len(embeddings) > 1:
                embedding_label = 'Embeddings: {}'
            else:
                embedding_label = 'Embedding: {}'
            embeddings = ", ".join(embeddings)
            self.table_fields.append([Merge(Label(embedding_label.format(embeddings), style="fg:ansiyellow bold"), 2)])

        self.table_buttons = [[Button('Save', handler=self.save), Button('Abort', handler=self.abort)]]

        self.kb = KeyBindings()

        @self.kb.add('c-c')
        def _(event):
            " Abort when Control-C has been pressed. "
            event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

        @self.kb.add("tab")
        def _(event):
            event.app.layout.focus_next()

        @self.kb.add("s-tab")
        def _(event):
            event.app.layout.focus_previous()

        # Add row to editor table/form with label, buffer and description for each configuration item
        for k, v in config_dict.items():
            if k in self.skip_params:
                continue
            if v is None:
                v = ""
                style = "fg:ansiwhite"
            elif isinstance(v, (list, tuple)):
                if v and isinstance(v[0], (list, tuple)):
                    v = "\n".join([", ".join([str(y) for y in x]) for x in v])
                else:
                    v = ", ".join([str(x) for x in v])
                style = "fg:ansigreen"
            elif v == '<multiple values>':
                style = "fg:ansimagenta italic"
            else:
                v = str(v)
                style = "fg:ansiwhite"

            # Prefill specified empty fields
            if not v and k in self.prefill_fields:
                v = str(self.prefill_fields[k])

            buffer = TextArea(v, style=style)
            self.buffers[k] = buffer
            row = [Label(k, style="fg:ansicyan bold"), Label(EDITOR_DESCRIPTIONS[conf_type][k], style="italic")]
            self.table_fields.append(row)
            self.table_fields.append(Merge(buffer, 2))
                        
        
        self.layout = Layout(
                        HSplit([
                            Table(
                                table=self.table_fields,
                                column_widths=[D(20, 20), D(80, 120)],
                                borders=RoundedBorder),
                            HorizontalLine(),
                            Table(
                                table=self.table_buttons,
                                column_widths=[D(20, 20), D(80, 120)],
                                borders=RoundedBorder)
                            ]
                        ),
                    )

    def __call__(self):
        return Application(self.layout, key_bindings=self.kb, full_screen=True).run()
    
    def abort(self):
        get_app().exit(result=None)

    def _cast_single(self, v):
        try:
            return int(v.text)
        except ValueError:
            try:
                return float(v.text)
            except ValueError:
                return v.text

    def _cast_list(self, v_list):
        if len(v_list) > 1:
            return [int(x) if x.isdigit() else x for x in v_list]
        else:
            try:
                return [int(v_list[0])]
            except ValueError:
                try:
                    return [float(v_list[0])]
                except ValueError:
                    return v_list

    def save(self):
        for k, v in self.buffers.items():
            v.text = v.text.strip()

            if v.text:
                if v.text == '<multiple values>':
                    continue

                if k in self.singleton_params or self.singleton_params == 'all':
                    if ',' in v.text:
                        get_app().exit(result="Field '{}' does not support lists of values. "
                                            "Press Enter to reopen editor.".format(k))
                        return

                    values = self._cast_single(v)
                    
                else:
                    if '\n' in v.text:
                        if k not in _2D_FIELDS:
                            get_app().exit(result="Field '{}' does not support multiple levels of nesting. "
                                                "Press Enter to reopen editor.".format(k))
                            return
                            
                        values_split = v.text.split('\n')
                        values_list = [v.replace(' ', '').split(',') for v in values_split]
                        values_list_cast = []
                        for v_list in values_list:
                            v_list = self._cast_list(v_list)
                            values_list_cast.append(v_list)
                        values = values_list_cast
                    else:
                        values_list = v.text.replace(' ', '').split(',')    
                        values = self._cast_list(values_list)
            else:
                if k in self.prefill_fields:
                    cprint('Repopulating empty field {}...'.format(k), 'magenta')
                    values = self.prefill_fields[k]
                    if not is_jsonable(values):
                        values = str(values)
                else:
                    values = None

            self.config_dict_updated[k] = values

        get_app().exit(result=True)


def config_editor(conf_type,
                  config_dict,
                  embeddings,
                  cognitive_sources,
                  singleton_params=None,
                  skip_params=None):
    config_dict_updated = {}

    result = None

    while True:
        conf_editor = ConfigEditor(conf_type,
                                   config_dict,
                                   config_dict_updated,
                                   singleton_params=singleton_params,
                                   cognitive_sources=cognitive_sources,
                                   embeddings=embeddings,
                                   skip_params=skip_params)
        result = conf_editor()

        if result is True:
            break
        elif result is None:
            return
        else:
            cprint('Error: {}'.format(result), 'red')
            prompt()
            result = None

    return config_dict_updated
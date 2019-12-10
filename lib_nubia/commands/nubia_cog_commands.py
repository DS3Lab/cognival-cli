#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import asyncio
import gzip
import os
import sys
import requests
import socket
import shutil
import typing
import zipfile

from pathlib import Path

import gdown
from nubia import command, argument, context
import numpy as np
import pandas as pd
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog, ProgressBar
from termcolor import cprint

CTX = context.get_context()

@command(aliases=["generate_random"])
@argument('embeddings',
          type=str,
          description='Name of embeddings that have been registered (not necessarily installed).',
          positional=True,
          choices=list(CTX.embedding2url['proper']))
def generate_random_embeddings(embeddings, no_embeddings=10, seed_func='exp_e_floored', use_custom_vocab=False):
    '''
    Generate random embeddings for embeddings that have been installed
    '''
    if embeddings.startswith('random'):
        cprint('✗ Reference embedding must be non-random! Aborting ...'. format(embeddings), 'red')
        return

    ctx = context.get_context()
    emb_properties = ctx.embedding2url['proper'].get(embeddings, None)
    if not emb_properties:
        cprint('✗ No specifications set for embeddings {}! Install custom embeddings or register them manually. Aborting ...'. format(embeddings), 'red')
        return

    if seed_func == 'exp_e_floored':
        seeds = [int(np.floor((k+1)**np.e)) for k in range(no_embeddings)]
    else:
        NotImplementedError('Only floor(x**e) (exp_e_floored) currently implemented')

    embedding_dim = emb_properties['dimensions']

    # Handle vocabulary (custom or bundled)
    if use_custom_vocab:
        if 'vocabulary_file' in emb_properties:
            with open(emb_properties['vocabulary_file']) as f:
                vocabulary = f.read().split('\n')
        else:
            cprint('When using a custom vocabulary, a corresponding newline-separated file must be specified in resources/embedding2url.json. Aborting ...', 'yellow')
            return
        cprint('Generating {}-dim. random embeddings using custom vocabulary ({} tokens)...'.format(embedding_dim, len(vocabulary)), 'yellow')
    else:
        with open('resources/standard_vocab.txt') as f:
            vocabulary = f.read().split('\n')
        cprint('Generating {}-dim. random embeddings using standard CogniVal vocabulary ({} tokens)...'.format(embedding_dim, len(vocabulary)), 'yellow')

    # Generate random embeddings
    emb_name = 'random-{}-{}'.format(embedding_dim, len(seeds))
    emb_files = ['{}_{}_{}.txt'.format(emb_name, idx+1, seed) for idx, seed in enumerate(seeds)]
    path = Path('embeddings/random_multiseed') / '{}_dim'.format(embedding_dim) / '{}_seeds'.format(len(seeds))
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    with ProgressBar() as pb:
        for seed, emb_file in pb(list(zip(seeds, emb_files))):
            np.random.seed(seed)
            rand_emb = np.random.uniform(low=-1.0, high=1.0, size=(len(vocabulary), embedding_dim))
            df = pd.DataFrame(rand_emb, columns=['x{}'.format(i+1) for i in range(embedding_dim)])
            df.insert(loc=0, column='word', value=vocabulary)
            df.to_csv(path / emb_file, sep=" ", encoding="utf-8")
    ctx.embedding2url['random_multiseed'][emb_name] = {'url': 'locally generated',
                                                       'dimensions': embedding_dim,
                                                       'path':str(path),
                                                       'embedding_files':emb_files,
                                                       'installed': True}
    cprint('✓ Generated random embeddings (Naming scheme: random-<dimensionality>-<no. seeds>-<#seed>-<seed_value>)', 'green')
    ctx.save_configuration()

@command
class Download:
    "This is a super command"

    def __init__(self, shared: int = 0) -> None:
        self._shared = shared

    @property
    def shared(self) -> int:
        return self._shared

    """This is the super command help"""

    @command(aliases=["cognitive"])
    @argument('force', type=bool, description='Force removal and download')
    @argument('url', type=str, description='Cognival vectors URL')
    def cognitive_sources(self, force=False, url='https://drive.google.com/uc?id=1pWwIiCdB2snIkgJbD1knPQ6akTPW_kx0'):
        """
        print a name
        """
        basepath = Path('cognitive_sources')
        if not os.path.exists(basepath) or force:
            try:
               shutil.rmtree(basepath)
            except FileNotFoundError:
               pass
            
            fullpath = basepath / 'cognival_vectors.zip'
            os.mkdir('cognitive_sources')
            cprint("Retrieving CogniVal cognitive sources ...", "yellow")
            gdown.cached_download(url, path=str(fullpath), quiet=False, postprocess=gdown.extractall)
            os.remove(basepath/ 'cognival_vectors.zip')
            shutil.rmtree(basepath / '__MACOSX')
            cprint("Completed installing CogniVal cognitive sources.", "green")
            paths = DisplayablePath.make_tree(basepath, max_len=10, max_depth=3)
            for path in paths:
                cprint(path.displayable(), 'cyan')

        else:
            cprint("CogniVal cognitive sources already present!", "green")

    @command()
    @argument('x', type=str, description='Force removal and download', positional=True)
    @argument('force', type=bool, description='Force removal and download')
    def embedding(self, x, force=False, log_only_success=False):
        """
        Lorem ipsum
        """
        ctx = context.get_context()
        if not os.path.exists('embeddings'):
            os.mkdir('embeddings')

        # Download all embeddings
        if x == 'all':
            for emb in ctx.embedding2url['proper']:
                self.embedding(emb)
            # Download random embeddings
            for rand_emb in ctx.embedding2url['random_static']:
                self.embedding(rand_emb, log_only_success=True)
            return

        # Download all static random embeddings
        elif x == 'all_random':
            for rand_emb in ctx.embedding2url['random_static']:
                self.embedding(rand_emb, log_only_success=True)
            return

        # Download a set of static random embeddings
        elif x.startswith('random'):
            name = x
            url = ctx.embedding2url['random_static'][x]['url']
            path = 'random_static'

        # Download a set of default embeddings
        elif x in ctx.embedding2url['proper']:
            name = x
            url = ctx.embedding2url['proper'][x]['url']
            path = ctx.embedding2url['proper'][x]['path']

        # Download custom embedding via URL
        elif x.startswith('http'):
            url = x
            
            name = input_dialog(title='Embedding registration',
                                text='You have provided a custom embedding URL ({}). Please make sure that\n'
                                     'all of the following criteria are met:\n\n'
                                     '- The URL is either a direct HTTP(S) link to the file or a Google Drive link. \n'
                                     '- The file is either a ZIP archive, gzipped file or usable as-is (uncompressed).\n\n'
                                     'Other modes of hosting and archival are currently NOT supported and will cause the installation to fail.\n'
                                     'In those instances, please manually download and extract the files in the "embeddings"'
                                     'directory and \nregister them in "resources/embedding2url.json"\n\n'
                                     'Please enter a short name for the embeddings:'.format(url)).run()
            if not name:
                cprint('Aborted.', 'red')
                return

            path = input_dialog(title='Embedding registration',
                                text='Optionally specify the directory name for the embeddings. The path may specify more details. \n'
                                     'If no string is specified, the filename (without extension) is used instead.'.format(url)).run()

            main_emb_file = input_dialog(title='Embedding registration',
                        text='Specify the main embedding file. This information is usually available from the supplier.\n'
                             'If not available, you can leave this information empty and manually edit resources/embeddings2url.json\n'
                             'after installation').run()

            if path == '':
                path = url.split('/')[-1].rsplit('.', maxsplit=1)[0]
            elif not path:
                cprint('Aborted.', 'red')
                return

            emb_dim = input_dialog(title='Embedding registration',
                        text='Please specify embedding dimensionality:').run()
            
            if not emb_dim:
                cprint('Aborted.', 'red')
                return

            associate_rand_emb = yes_no_dialog(title='Embedding registration',
                                   text='Do you wish to compare the embeddings with random embeddings of identical dimensionality? \n').run()
            
            vocab_file = None
            if associate_rand_emb:                                
                vocab_file = input_dialog(title='Embedding registration',
                text='Do you wish to create random embeddings using a custom vocabulary? If yes, \n'
                     'specify the path of the vocabulary file.\nIf left empty, the CogniVal standard vocabulary '
                     '(22,028 tokens) is used.').run()

                if emb_dim:
                    try:
                        emb_dim = int(emb_dim)
                    except ValueError:
                        cprint('Error: {} is not a valid embedding dimensionality.'.format(emb_dim), 'red')
                        return
                    
                    available_dims = set()
                    for emb, parameters in ctx.embedding2url['random_multiseed'].items():
                        available_dims.add(parameters['dimensions'])

                    generate_rand_emb = False
                    if emb_dim in available_dims:
                        cprint('Random embeddings of dimensionality {} already present.'.format(emb_dim), 'green')
                    else:
                        cprint('No pre-existing random embeddings of dimensionality {}, generating ...'.format(emb_dim), 'yellow')
                        generate_rand_emb = True

                else:
                    cprint('Aborted.', 'red')
                    return
                
            ctx.embedding2url['proper'][name] = {'url': url,
                                                 'dimensions': emb_dim,
                                                 'path': path,
                                                 'embedding_file': main_emb_file}
            if vocab_file:
                ctx.embedding2url['proper'][name]['vocabulary_file'] = vocab_file

            if generate_rand_emb:
                generate_random_embeddings(name, use_custom_vocab=False)
 
        # Show available default embeddings otherwise
        else:
            list_embeddings()
            # cprint('"{}" is not a default embedding or URL. Provide a download URL for custom embeddings.'.format(x), 'red')
            # cprint('Available embeddings:')
            # cprint('Name' + ' '*21 + 'URL')
            # for key, value in ctx.embedding2url['proper'].items():
            #     cprint(key, 'cyan', end=(' '*(25-len(key))))
            #     cprint(value['url'], 'green')
            # cprint('random_static', 'cyan', end=' '*19)
            # cprint(ctx.embedding2url['random_static']['url'], 'green')
            # return

        fname = url.split('/')[-1]
        fpath = Path('embeddings') / fname
        fpath_extracted = Path('embeddings') / path

        if not os.path.exists(fpath_extracted) or force:
            try:
               shutil.rmtree(fpath_extracted)
            except FileNotFoundError:
               pass

            cprint('Downloading and installing:', 'yellow', end =' ') 
            cprint('{}'.format(name), 'yellow', attrs=['bold'])
            # Google Drive downloads
            if 'drive.google.com' in url:
                gdown.download(url, 'embeddings/gdrive_embeddings.dat', quiet=False)
                try:
                    with zipfile.ZipFile('embeddings/gdrive_embeddings.dat', 'r') as zip_ref:
                        zip_ref.extractall(fpath_extracted)
                except zipfile.BadZipFile:
                    # Assume gzipped bin (requires manually creating path and setting filename)
                    os.mkdir(fpath_extracted)
                    with gzip.open('embeddings/gdrive_embeddings.dat', 'rb') as f_in:
                        with open(fpath_extracted / '{}.bin'.format(path), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                os.remove('embeddings/gdrive_embeddings.dat')
            # Normal HTTP downloads
            else:
                os.makedirs(fpath_extracted)
                download_file(url, fpath)
                if fname.endswith('zip'):
                    with zipfile.ZipFile(fpath, 'r') as zip_ref:
                        zip_ref.extractall(fpath_extracted)
                    os.remove(fpath)
                elif fname.endswith('gz'):
                    # Assume gzipped bin (requires manually setting filename)
                    with gzip.open('gdrive_embeddings.dat', 'rb') as f_in:
                        with open(fpath_extracted / '{}.bin'.format(path), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.move(fpath, fpath_extracted / fname)
            cprint('Finished installing embedding "{}"'.format(name), 'green')
        else:
            if not log_only_success:
                cprint('Embedding {} already installed. Use "force" to override'.format(name), 'yellow')
        
        if name.startswith('random'):
            ctx.embedding2url['random_static'][name]['installed'] = True
        else:
            ctx.embedding2url['proper'][name]['installed'] = True
        ctx.save_configuration()

    @command(aliases=["do"])
    def do_stuff(self, stuff: int):
        """
        doing stuff
        """
        cprint("stuff={}, shared={}".format(stuff, self.shared))

@command(aliases=["lookup"])
@argument("hosts", description="Hostnames to resolve", aliases=["i"])
@argument("bad_name", name="nice", description="testing")
def lookup_hosts(hosts: typing.List[str], bad_name: int):
    """
    This will lookup the hostnames and print the corresponding IP addresses
    """
    ctx = context.get_context()
    cprint("Input: {}".format(hosts), "yellow")
    cprint("Verbose? {}".format(ctx.verbose), "yellow")
    for host in hosts:
        ctx.state[host] = socket.gethostbyname(host)
        cprint("{} is {}".format(host, socket.gethostbyname(host)), "red")

    # optional, by default it's 0
    return 0

@command
def list_embeddings():
    """
    List all available and installed embeddings
    """
    ctx = context.get_context()
    for section in ['proper', 'random_static', 'random_multiseed']:
        cprint(section.title(), attrs=['bold'])
        for key, value in ctx.embedding2url[section].items():
            cprint(key, 'cyan', end=' '*(25-len(key)))
            if value['installed']:
                cprint('installed', 'green', attrs=['bold'])
            else:
                cprint('not installed', 'red', attrs=['bold'])

@command("good-name")
def bad_name():
    """
    This command has a bad function name, but we ask Nubia to register a nicer
    name instead
    """
    cprint("Good Name!", "green")


@command
@argument("number", type=int)
async def triple(number):
    "Calculates the triple of the input value"
    cprint("Input is {}".format(number))
    cprint("Type of input is {}".format(type(number)))
    cprint("{} * 3 = {}".format(number, number * 3))
    await asyncio.sleep(2)


@command("be-blocked")
def be_blocked():
    """
    This command is an example of command that blocked in configerator.
    """

    cprint("If you see me, something is wrong, Bzzz", "red")

@command
@argument("style", description="Pick a style", choices=["test", "toast", "toad"])
@argument("stuff", description="more colors", choices=["red", "green", "blue"])
@argument("code", description="Color code", choices=[12, 13, 14])
def pick(style: str, stuff: typing.List[str], code: int):
    """
    A style picking tool
    """
    cprint("Style is '{}' code is {}".format(style, code), "blue")


# instead of replacing _ we rely on camelcase to - super-command


@command
class SuperCommand:
    "This is a super command"

    def __init__(self, shared: int = 0) -> None:
        self._shared = shared

    @property
    def shared(self) -> int:
        return self._shared

    """This is the super command help"""

    @command
    @argument("firstname", positional=True)
    def print_name(self, firstname: str):
        """
        print a name
        """
        cprint("My name is: {}".format(firstname))

    @command(aliases=["do"])
    def do_stuff(self, stuff: int):
        """
        doing stuff
        """
        cprint("stuff={}, shared={}".format(stuff, self.shared))


#Source: https://stackoverflow.com/a/49912639
class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last, max_depth, max_len, depth):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        self.max_depth = max_depth
        self.max_len = max_len
        self.depth = depth
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None, max_depth=0, max_len=0, depth=0):
        if not root == "...":
            root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last, max_depth, max_len, depth)
        yield displayable_root

        if not root == '...':
            children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
            count = 1
            for idx, path in enumerate(children):
                is_last = count == len(children)
                if path.is_dir():
                    if not depth or depth < max_depth:
                        yield from cls.make_tree(path,
                                                parent=displayable_root,
                                                is_last=is_last,
                                                criteria=criteria,
                                                max_depth=max_depth,
                                                max_len=max_len,
                                                depth=depth+1)
                    else:
                        yield from cls.make_tree('...',
                                    parent=displayable_root,
                                    is_last=True,
                                    criteria=criteria,
                                    max_depth=max_depth,
                                    max_len=max_len,
                                    depth=depth+1)
                        break

                else:
                    if not max_len or idx+1 < max_len:
                        yield cls(path, displayable_root, is_last, max_depth, max_len, depth+1)
                    else:
                        yield from cls.make_tree('...',
                                    parent=displayable_root,
                                    is_last=True,
                                    criteria=criteria,
                                    max_depth=max_depth,
                                    max_len=max_len,
                                    depth=depth+1)
                        break
                count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))

#Source: https://sumit-ghosh.com/articles/python-download-progress-bar/
def download_file(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}] [{:.2f}/{:.2f}MB]'.format('█' * done, '.' * (50-done), downloaded/(1024*1024), total/(1024*1024)))
                sys.stdout.flush()
    sys.stdout.write('\n')
import json
import sys

import requests
from pathlib import Path
from termcolor import cprint

#http://stackoverflow.com/questions/1014352/how-do-i-convert-a-nested-tuple-of-tuples-and-lists-to-lists-of-lists-in-python
def tupleit(t):
    return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t

def _open_config(configuration, quiet=False):
    config_path = Path('resources') / '{}_config.json'.format(configuration)
    try:
        with open(config_path) as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        cprint('Error: Configuration file {}_config.json does not yet exist! Execute `edit-config create <filename> to create a new configuration.'.format(configuration), 'red')
        return
    if not quiet:
        cprint('Opened configuration file {} ...'.format(str(config_path)), 'green')
    return config_dict
    

def _open_cog_config():
    cog_sources_path = Path('resources') / 'cognitive_sources.json'
    with open(cog_sources_path) as f:
        cognitive_sources = json.load(f)
    return cognitive_sources


def _check_cog_installed():
    cog_config = _open_cog_config()
    return cog_config['installed']


def _check_emb_installed(embedding, embeddings_conf):
    try:
        return embeddings_conf['proper'][embedding]['installed']
    except KeyError:
        try:
            return embeddings_conf['random_static'][embedding]['installed']
        except KeyError:
            try:
                return embeddings_conf['random_multiseed'][embedding]['installed']
            except KeyError:
                raise RuntimeError('Embedding "{}" not known.'.format(embedding))


def _save_cog_config(config_dict):
    config_path = Path('resources') / 'cognitive_sources.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def _save_config(config_dict, configuration, quiet=False):
    config_path = Path('resources') / '{}_config.json'.format(configuration)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    if not quiet:
        cprint('Saved configuration file {} ...'.format(str(config_path)), 'green')


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
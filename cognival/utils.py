import errno
import os
import sys
import time
import warnings

from gensim.models.keyedvectors import KeyedVectors
import fasttext
from tqdm import tqdm

#
# Misc
#

#animation to know when script is running
def animated_loading(completed, total):
    chars = r"/-\|"
    for char in chars:
        sys.stdout.write('\r'+'loading...'+char)
        sys.stdout.write('\t'+str(completed)+"/"+str(total))
        time.sleep(.1)
        sys.stdout.flush()

#
# Embeddings
# 

def word2vec_bin_to_txt(binPath, binName, outputName, limit=None):
    with warnings.catch_warnings():
        # Silence smart_open deprecation warnings (erroneously raised as UserWarnings)
        warnings.simplefilter('ignore', UserWarning)
        model = KeyedVectors.load_word2vec_format(binPath.parent / binName, binary=True, limit=limit)
        os.makedirs(binPath, exist_ok=True)
        model.save_word2vec_format(binPath / outputName,binary=False)


def fasttext_bin_to_txt(binPath, binName, outputName, limit=None):
    f = fasttext.load_model(str(binPath.parent / binName))
    os.makedirs(binPath, exist_ok=True)
    # Copied from https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/bin_to_vec.py 
    words = f.get_words()
    with open(binPath / outputName, 'w') as f_out:
        for w in tqdm(words):
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                print(w + vstr, file=f_out)
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass 

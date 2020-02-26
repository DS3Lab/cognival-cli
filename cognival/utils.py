import sys
import time
import warnings

from gensim.models.keyedvectors import KeyedVectors

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
        model = KeyedVectors.load_word2vec_format(binPath / binName, binary=True, limit=limit)
        model.save_word2vec_format(binPath / outputName,binary=False)
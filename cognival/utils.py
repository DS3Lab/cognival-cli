import csv
import json
import sys
import time
import warnings

import pandas as pd
import numpy as np

from numpy import fromfile
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

def header_gen(fileName):
    df = pd.read_csv(fileName, sep=" ", engine='python', header=None, quoting=csv.QUOTE_NONE)
    dim = df.shape[1]
    header = "word"
    for i in range(1,dim):
        header = header+" x"+str(i)
    return header


def generate_df_with_header(fileName, skiprows=None):
    reader = pd.read_csv(fileName, sep=" ", engine='python', header=None, quoting=csv.QUOTE_NONE, chunksize=10000, skiprows=skiprows)
    df_chunk_1 = next(reader)
    dim = df_chunk_1.shape[1]
    header = "word"
    for i in range(1,dim):
        header = header+" x"+str(i)
    header = header.split()
    df_chunk_1.columns = header
    return df_chunk_1, reader


def word2vec_bin_to_df(filename,rec_dtype):
    return pd.DataFrame(fromfile(filename,rec_dtype))


def word2vec_bin_to_txt(binPath,binName,outputName, limit=None):
    with warnings.catch_warnings():
        # Silence smart_open deprecation warnings (erroneously raised as UserWarnings)
        warnings.simplefilter('ignore', UserWarning)
        model = KeyedVectors.load_word2vec_format(binPath / binName, binary=True, limit=limit)
        model.save_word2vec_format(binPath / outputName,binary=False)
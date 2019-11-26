import csv
import sys
import time

import pandas as pd
import numpy as np

from numpy import fromfile, dtype
from gensim.models.keyedvectors import KeyedVectors

#animation to know when script is running
def animatedLoading(completed, total):
	chars = r"/-\|"
	for char in chars:
		sys.stdout.write('\r'+'loading...'+char)
		sys.stdout.write('\t'+str(completed)+"/"+str(total))
		time.sleep(.1)
		sys.stdout.flush()


def headerGen(fileName):
	df = pd.read_csv(fileName, sep=" ", engine='python', quoting=csv.QUOTE_NONE)
	#print(df.shape)
	dim = df.shape[1]
	header = "word"
	for i in range(1,dim):
		header = header+" x"+str(i)
	return header


def word2vec_bin_to_df(filename,rec_dtype):
    return pd.DataFrame(fromfile(filename,rec_dtype))


def word2vec_bin_to_txt(binPath,binName,outputName):
    model = KeyedVectors.load_word2vec_format(binPath+binName,binary=True)
    model.save_word2vec_format(binPath+outputName,binary=False)


# def main(binPath,binName,outputName, dim):
#     header = [('word','str')] + [('x%s'%i,'float64') for i in range(1,dim+1)]
#     dt = dtype(header)
#     print(bin_to_df(binPath+binName,dt))
#     #bin_to_txt(binPath,binName,outputName)

# if __name__=="__main__":
#     main('/Users/delatan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/word2vec/',
#                   "GoogleNews-vectors-negative300.bin", 'word2vec.txt',300)

if __name__=="__main__":
	fileName = "../embeddings/glove-6B/glove.6B.50d_nohead.txt"
	print(headerGen(fileName))
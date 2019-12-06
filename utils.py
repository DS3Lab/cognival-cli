import csv
import json
import sys
import time

import pandas as pd
import numpy as np

from numpy import fromfile, dtype
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

#
# Formatting
#

def create_table(CONFIG):
    with open(CONFIG,'r') as fR:
        config = json.load(fR)
    header = [key for key in config['wordEmbConfig']]
    print(header)
    index1 = []
    index2 = []
    for cD in config["cogDataConfig"]:
    	for feature in config["cogDataConfig"][cD]["features"]:
    		index1.append(cD)
    		index2.append(feature)
    index = [index1,index2]	
    
    setup = {header[j]:[np.NaN for i in range(len(index2)) ] for j in range(len(header))} 
    df = pd.DataFrame(setup,index)
    print(df)	
    
    pass

def fill_table(PATH, table):

    pass

#def main():
#    #OPTIONS = "../test_final/options.json"
#    CONFIG = "../config/example_1.json"
#    createTable(CONFIG)
#    pass

#if __name__=="__main__":
#    main()

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
	print(header_gen(fileName))

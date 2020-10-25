

import sys
import re
import argparse
from pathlib import Path
import os
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  bert_path = os.path.join("/home/lvkleis/.cognival/embeddings/bert-large-cased-sentence/wwm_cased_L-24_H-1024_A-16/sentence", "bert-large-cased_as_sentembeddings_cognival.txt") 
  

  parser.add_argument('--s', '--source',
                        default=bert_path, help='path to file containing BERT sentence embeddings')
  args = parser.parse_args()
  source_path = args.s
  embeddings_file = open(bert_path, 'r')
  lines = embeddings_file.readlines()
  
  sentences = []  
  for line in lines:
    sentence = re.split("[-+]?\d*\.\d+|\d+", line, 2)[0][:-1]
    sentences.append(sentence + "\n")
  
  destination_filepath = 'sentence_vocabulary.txt'
  newfile = open(destination_filepath, 'w') 
  newfile.writelines(sentences) 
  newfile.close() 
  print(f'Done! Saved results to {destination_filepath}') 



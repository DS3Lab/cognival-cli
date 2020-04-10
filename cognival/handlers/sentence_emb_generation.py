import collections
import csv
import os

import gensim
import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm

def generate_avg_sent_embeddings(name,
                                 resources_path,
                                 emb_params,
                                 base_path,
                                 emb_file):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    emb_root = base_path.parent
    os.makedirs(emb_root / 'sentence', exist_ok=True)
    
    with open(resources_path / 'standard_sent_vocab.txt') as f:
        vocabulary_set = set([x.strip() for x in f])
    
    emb_dict = {}

    # Handle fasttext subwords 
    if 'fasttext' in name:
        embeddings = loast_facebook_model(base_path / emb_params["embedding_file"])
        for word in vocabulary_set:
            emb_dict[word] = embeddings.wv[word]
    else:
        df = pd.read_csv(base_path / emb_file,
                         sep=" ",
                         quoting=csv.QUOTE_NONE,
                         doublequote=True,
                         keep_default_na=False,
                         names=['word', *['x{}'.format(idx+1) for idx in range(emb_params['dimensions'])]])
                         
        df.set_index('word', inplace=True)
        assert df.index.is_unique
        counter = collections.Counter()
        
        # collect word embeddings
        for word in vocabulary_set:
            if word in df.index:
                emb_dict[word] = df.loc[word].to_numpy().ravel()
                counter['normal'] += 1
            elif word.lower() in df.index:
                emb_dict[word] = df.loc[word.lower()].to_numpy().ravel()
                counter['lower'] += 1
            else:
                pass
                counter['OOV'] += 1

        print(counter)
        del df

    sentences = []
    sent_embs = []
    with open(resources_path / 'standard_sentences.txt') as f:
        for sent in f:
            sentences.append(sent.strip())

    # Generate average word embeddings per sentence (not accounting for subwords)
    for sent in tqdm(sentences):
        sent_word_emb = []
        for token in nlp(sent):
            if token.text in emb_dict:
                sent_word_emb.append(emb_dict[token.text])
            elif token.text.lower() in emb_dict:
                sent_word_emb.append(emb_dict[token.text.lower()])
        if sent_word_emb:
            sent_embs.append(np.mean(sent_word_emb, axis=0))
        else:
            sent_embs.append(np.zeros(emb_params['dimensions']))
   
    df_sent_emb_rows = [{'sentence':sentence, **{'e{}'.format(idx+1):v for idx, v in enumerate(sent_emb)}} for sentence, sent_emb in zip(sentences, sent_embs)]
    df_sent_emb = pd.DataFrame(df_sent_emb_rows)
    df_sent_emb.set_index('sentence', inplace=True)
    df_sent_emb.to_csv(emb_root / 'sentence' / emb_file, sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC, index=True, header=False)

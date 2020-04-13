import collections
import csv
import os

import fasttext
import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm

def generate_bert_sentence_embs():
    import torch
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0][0]  # The last hidden-state is the first element of the output tuple
    cls_embeddings = last_hidden_states[0].detach().numpy()


def generate_elmo_sentence_embs():
    elmo = ElmoEmbedder()
    y = elmo.embed_sentence(words.split()) # pass in as list of tokens
    y_2 =  y.swapaxes(0, 1)
    z = y_2.reshape(5, 3072)
    z = z.mean(axis=0)


def generate_powermean_sentence_embs():
    import tensorflow as tf
    import tensorflow_hub as hub
    import time
    with tf.Graph().as_default():
        embed = hub.Module('https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1')
        emb_tensor = embed(["A long sentence .", "another sentence"])   
        sess = tf.train.SingularMonitoredSession()
        embeddings = emb_tensor.eval(session=sess)
        print(embeddings)


def generate_skipthought_sentence_embs():
    pass


def generate_quickthought_sentence_embs():
    pass


def generate_use_sentence_embs():
    pass


def generate_infersent_sentence_embs():
    pass


def generate_sent_embeddings(name):
    if 'bert' in name:
        pass
    elif name == 'elmo-sentence':
        pass
    elif name == 'powermean':
        pass
    elif name == 'skipthought':
        pass
    elif name == 'quickthought':
        pass
    elif name == 'use':
        pass
    elif name == 'infersent':
        pass
    else:
        return

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

    print("Getting word embeddings ...")
    # Handle fasttext subwords 
    if 'fasttext-cc-2018' in name:
        embeddings = fasttext.load_model(str(base_path / emb_params["binary_file"]))
        for word in tqdm(vocabulary_set):
            emb_dict[word] = embeddings[word]
    else:
        skiprows = 1 if emb_params['truncate_first_line'] else None
        df = pd.read_csv(base_path / emb_file,
                         sep=" ",
                         quoting=csv.QUOTE_NONE,
                         doublequote=True,
                         keep_default_na=False,
                         skiprows=skiprows,
                         names=['word', *['x{}'.format(idx+1) for idx in range(emb_params['dimensions'])]])
        df.set_index('word', inplace=True)
        assert df.index.is_unique
        counter = collections.Counter()
        
        # collect word embeddings
        for word in tqdm(vocabulary_set):
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

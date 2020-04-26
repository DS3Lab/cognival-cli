import collections
import csv
import gc
import os
import sys

import fasttext
import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm
from termcolor import cprint

from allennlp.commands.elmo import ElmoEmbedder
import torch
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel
from skipthoughts import BiSkip

import tensorflow as tf
import tensorflow_hub as hub
import time

tf.compat.v1.logging.set_verbosity(0)

from handlers.sentence_embeddings.infersent.models import InferSent
from handlers.sentence_embeddings.quickthought import configuration as quickthought_config
from handlers.sentence_embeddings.quickthought import encoder_manager as quickthought_enc_manager

from handlers.sentence_embeddings.skip_thoughts import configuration as skipthought_config
from handlers.sentence_embeddings.skip_thoughts import encoder_manager as skipthought_enc_manager

def get_resources(base_path, resources_path):
    nlp = get_nlp()
    emb_path = create_sent_path(base_path)
    sent_vocab = get_sent_vocab(resources_path)
    sentences = get_sentences(resources_path)
    
    return nlp, emb_path, sent_vocab, sentences


def get_nlp():
    return spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def create_sent_path(base_path):
    emb_root = base_path.parent
    emb_path = emb_root / 'sentence'
    os.makedirs(emb_path, exist_ok=True)
    return emb_path


def get_sent_vocab(resources_path):
    with open(resources_path / 'standard_sent_vocab.txt') as f:
        sentences_set = set([x.strip() for x in f])
    return sentences_set


def get_sentences(resources_path):
    with open(resources_path / 'standard_sentences.txt') as f:
        sentences = set([x.strip() for x in f])
    return sentences


def export_df(emb_path, emb_file, sentences, matrix, dimensions):
    df = pd.DataFrame(data=matrix,
                      columns=['x{}'.format(idx+1) for idx in range(dimensions)])

    df.insert(0, 'sentences', sentences)
    df.set_index('sentences', inplace=True)
    df.to_csv(emb_path / emb_file,
                       sep=" ",
                       encoding="utf-8",
                       quoting=csv.QUOTE_NONNUMERIC,
                       index=True,
                       header=False)


def generate_bert_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)

    print("Obtaining transformers model {}. Unless already downloaded, this may take several minutes ...".format(emb_params['internal_name']) )
    tokenizer = BertTokenizer.from_pretrained(emb_params['internal_name'])
    model = BertModel.from_pretrained(emb_params['internal_name'])
    
    embeddings = []
    # BERT uses specialized tokenizer, generating subwords
    print("Tokenizing and embedding ...")
    for sentence in tqdm(sentences):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0][0]  # The last hidden-state is the first element of the output tuple
        cls_embeddings = last_hidden_states[0].detach().numpy()
        embeddings.append(cls_embeddings)

    embeddings = np.vstack(embeddings)
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])


def generate_elmo_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
    print("Tokenizing ...")
    sentence_tokens = [[token.text for token in nlp(sentence)] for sentence in tqdm(sentences)]
    embeddings = []
    elmo = ElmoEmbedder()
    print("Embedding ...")
    for sent_tok in tqdm(sentence_tokens):
        elmo_output = elmo.embed_sentence(sent_tok) # pass in as list of tokens
        elmo_output = elmo_output.swapaxes(0, 1) # reorder for reshaping
        elmo_word_embs_flattened = elmo_output.reshape(len(sent_tok), 3072) # flatten layers
        elmo_sent_emb = elmo_word_embs_flattened.mean(axis=0)
        embeddings.append(elmo_sent_emb)

    embeddings = np.vstack(embeddings)
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])


def generate_powermean_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
    print("Tokenizing ...")
    # powermean requires tokenized sentence strings
    sentences_tokenized = [" ".join([token.text for token in nlp(sentence)]) for sentence in tqdm(sentences)]

    print("Embedding (chunked) ...")
    embeddings = []
    for sent_chunk in tqdm(np.array_split(sentences_tokenized, 32)):
        with tf.Graph().as_default():
            embed = hub.Module(emb_params['url'])
            emb_tensor = embed(list(sent_chunk))
            with tf.compat.v1.train.SingularMonitoredSession() as sess:
                embeddings.append(emb_tensor.eval(session=sess))
        tf.keras.backend.clear_session()
        
    embeddings = np.vstack(embeddings)
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])


def generate_skipthought_sentence_embs(variant, resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
        
    encoder = skipthought_enc_manager.EncoderManager()
    encoder.load_model(skipthought_config.model_config(),
                       vocabulary_file=str(base_path.parent / "skip_thoughts_uni_2017_02_02" / "vocab.txt"),
                       embedding_matrix_file=str(base_path.parent / "skip_thoughts_uni_2017_02_02" / "embeddings.npy"),
                       checkpoint_path=str(base_path.parent / "skip_thoughts_uni_2017_02_02" / "model.ckpt-501424"))
    if variant == 'bi':
        try:
            encoder.load_model(skipthought_config.model_config(bidirectional_encoder=True),
                               vocabulary_file=str(base_path.parent / "skip_thoughts_bi_2017_02_16" / "vocab.txt"),
                               embedding_matrix_file=str(base_path.parent / "skip_thoughts_bi_2017_02_16" / "embeddings.npy"),
                               checkpoint_path=str(base_path.parent / "skip_thoughts_bi_2017_02_16" / "model.ckpt-500008"))
        except FileNotFoundError:
            raise RuntimeError("Install skip-thoughts-uni first!")
    # Internally uses NLTK punct tokenizer for tokenization
    print("Tokenizing and embedding ...")
    encodings = encoder.encode(sentences)

    export_df(emb_path, emb_file, sentences, encodings, emb_params["dimensions"])


def generate_quickthought_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
    encoder = quickthought_enc_manager.EncoderManager()
    
    # TF flags required by QuickThought
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("results_path", str(base_path), "Model results path")
    tf.flags.DEFINE_string("Glove_path", str(base_path / 'dictionaries' / 'GloVe'), "GloVe dictionary")
    tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")

    model_config = { "encoder": "gru",
                     "encoder_dim": 1200,
                     "bidir": True,
                     "case_sensitive": True,
                     "checkpoint_path": base_path / "BS400-W300-S1200-Glove-BC-bidir/train",
                     "vocab_configs": [
                     {
                     "mode": "fixed",
                     "name": "word_embedding",
                     "cap": False,
                     "dim": 300,
                     "size": 2196018,
                     "vocab_file": "",
                     "embs_file": ""
                     }
                     ]
                   }  
    model_config = quickthought_config.model_config(model_config, mode="encode")
    encoder.load_model(model_config)

    # Quickthought internally uses NLTK tokenization
    print("Tokenizing and embedding ...")
    embeddings = encoder.encode(tqdm(sentences))
    encoder.close()
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])
    

def generate_use_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
    # tensorflow session
    session = tf.compat.v1.Session()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def make_embed_fn(module):
      with tf.Graph().as_default():
        sents = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sents)
        session = tf.compat.v1.train.MonitoredSession()
      return lambda x: session.run(embeddings, {sents: x})

    # Start TF session and load Google Universal Sentence Encoder
    encoder = make_embed_fn(emb_params['url'])
    print("Tokenizing and embedding ...")
    embeddings = encoder(list(sentences))
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])


def generate_infersent_sentence_embs(resources_path, emb_params, base_path, emb_file):
    nlp, emb_path, sent_vocab, sentences = get_resources(base_path, resources_path)
    params_model = {'bsize': 64,
                    'word_emb_dim': 300,
                    'enc_lstm_dim': 2048,
                    'pool_type': 'max',
                    'dpout_model': 0.0,
                    'version': 2} #InferSent version 2 with fasttext embeddings

    
    print("Embedding ...")
    # InferSent relies on NLTK punkt for tokenization (tokenize=False performs white-space splitting)
    model = InferSent(params_model)
    #model = model.cuda()
    model.load_state_dict(torch.load(base_path / 'infersent2.pkl'))
    model.set_w2v_path(base_path / 'crawl-300d-2M.vec')
    model.build_vocab(sentences, tokenize=True)
    embeddings = model.encode(tqdm(sentences), tokenize=True)
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])


def generate_sent_embeddings(name,
                             resources_path,
                             emb_params,
                             base_path,
                             emb_file):
    if 'bert' in name:
        generate_bert_sentence_embs(resources_path, emb_params, base_path, emb_file)
    elif name == 'elmo-sentence':
        generate_elmo_sentence_embs(resources_path, emb_params, base_path, emb_file)
    elif name == 'power-mean':
        generate_powermean_sentence_embs(resources_path, emb_params, base_path, emb_file)
    elif 'skip-thoughts' in name:
        if 'uni' in name:
            variant = 'uni'
        elif 'bi' in name:
            variant = 'bi'
        generate_skipthought_sentence_embs(variant, resources_path, emb_params, base_path, emb_file)
    elif name == 'quick-thoughts':
        generate_quickthought_sentence_embs(resources_path, emb_params, base_path, emb_file)
    elif name == 'use':
        generate_use_sentence_embs(resources_path, emb_params, base_path, emb_file)
    elif name == 'infersent':
        generate_infersent_sentence_embs(resources_path, emb_params, base_path, emb_file)
    else:
        return


def generate_avg_sent_embeddings(name,
                                 resources_path,
                                 emb_params,
                                 base_path,
                                 emb_file):
    nlp = get_nlp()
    emb_path = create_sent_path(base_path)
    sent_vocab = get_sent_vocab(resources_path)

    emb_dict = {}

    print("Getting word embeddings ...")
    # Handle fasttext subwords 
    if 'fasttext-cc-2018' in name:
        embeddings = fasttext.load_model(str(base_path.parent / emb_params["binary_file"]))
        for word in tqdm(sent_vocab):
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
        for word in tqdm(sent_vocab):
            if word in df.index:
                emb_dict[word] = df.loc[word].to_numpy().ravel()
                counter['normal'] += 1
            elif word.lower() in df.index:
                emb_dict[word] = df.loc[word.lower()].to_numpy().ravel()
                counter['lower'] += 1
            else:
                pass
                counter['OOV'] += 1
        
        counter_sum = sum(list(counter.values()))
        cprint("[Statistics] Normally resolved: {}/{:.2f}% | Lowercased resolved: {}/{:.2f}% | OOV: {}/{:.2f}%".format(counter['normal'],
                   (counter['normal']/counter_sum) * 100,
                   counter['lower'],
                   (counter['lower']/counter_sum) * 100,
                   counter['OOV'],
                   (counter['OOV']/counter_sum) * 100), color='magenta')
        del df

    sentences = get_sentences(resources_path)
    sent_embs = []

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
  

    embeddings = np.vstack(sent_embs)
    export_df(emb_path, emb_file, sentences, embeddings, emb_params["dimensions"])

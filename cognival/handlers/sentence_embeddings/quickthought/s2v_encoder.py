# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import nltk
import nltk.tokenize
import numpy as np
import tensorflow as tf

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s2v_model

FLAGS = tf.compat.v1.flags.FLAGS

UNK = "<unk>"

def _pad(seq, target_len):
  """Pads a sequence of word embeddings up to the target length.

  Args:
    seq: Sequence of word embeddings.
    target_len: Desired padded sequence length.

  Returns:
    embeddings: Input sequence padded with zero embeddings up to the target
      length.
    mask: A 0/1 vector with zeros corresponding to padded embeddings.

  Raises:
    ValueError: If len(seq) is not in the interval (0, target_len].
  """
  seq_len = len(seq)
  if seq_len <= 0 or seq_len > target_len:
    raise ValueError("Expected 0 < len(seq) <= %d, got %d" % (target_len,
                                                              seq_len))

  emb_dim = seq[0].shape[0]
  padded_seq = np.zeros(shape=(target_len, emb_dim), dtype=seq[0].dtype)
  mask = np.zeros(shape=(target_len,), dtype=np.int8)
  for i in range(seq_len):
    padded_seq[i] = seq[i]
    mask[i] = 1
  return padded_seq, mask


def _batch_and_pad(sequences):
  """Batches and pads sequences of word embeddings into a 2D array.

  Args:
    sequences: A list of batch_size sequences of word embeddings.

  Returns:
    embeddings: A numpy array with shape [batch_size, padded_length, emb_dim].
    mask: A numpy 0/1 array with shape [batch_size, padded_length] with zeros
      corresponding to padded elements.
  """
  batch_len = max([len(seq) for seq in sequences])
  batch = np.zeros((len(sequences), batch_len), dtype=np.int64)
  mask = np.zeros((len(sequences), batch_len), dtype=np.int8)
  for i in range(len(sequences)):
    seq = sequences[i]
    l = len(seq)
    batch[i][:l] = seq
    mask[i][:l] = 1
  return batch, mask


def _batch_and_pad_embs(sequences):
  """Batches and pads sequences of word embeddings into a 2D array.
  Args:
    sequences: A list of batch_size sequences of word embeddings.
  Returns:
    embeddings: A numpy array with shape [batch_size, padded_length, emb_dim].
    mask: A numpy 0/1 array with shape [batch_size, padded_length] with zeros
      corresponding to padded elements.
  """
  batch_embeddings = []
  batch_mask = []
  batch_len = max([len(seq) for seq in sequences])

  for seq in sequences:
    embeddings, mask = _pad(seq, batch_len)
    batch_embeddings.append(embeddings)
    batch_mask.append(mask)
  return np.array(batch_embeddings), np.array(batch_mask)

class s2v_encoder(object):

  def __init__(self, config):
    """Initializes the encoder.

    Args:
      embeddings: Dictionary of word to embedding vector (1D numpy array).
    """
    self._sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    self.config = config

  def _create_restore_fn(self, checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
      saver: Saver for restoring variables from the checkpoint file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.

    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """
    if tf.compat.v1.gfile.IsDirectory(checkpoint_path):
      latest_checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_path)
      if not latest_checkpoint:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      checkpoint_path = latest_checkpoint

    def _restore_fn(sess):
      tf.compat.v1.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.compat.v1.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, model_config): #, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
      model_config: Object containing configuration for building the model.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    tf.compat.v1.logging.info("Building model.")
    model = s2v_model.s2v(model_config, mode="encode")
    model.build_enc()
    self._embeddings = model.word_embeddings
    saver = tf.compat.v1.train.Saver()
    checkpoint_path = model_config.checkpoint_path

    return self._create_restore_fn(checkpoint_path, saver)

  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    """Builds the inference graph from serialized GraphDef and SaverDef protos.

    Args:
      graph_def_file: File containing a serialized GraphDef proto.
      saver_def_file: File containing a serialized SaverDef proto.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    # Load the Graph.
    tf.compat.v1.logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.compat.v1.import_graph_def(graph_def, name="")

    # Load the Saver.
    tf.compat.v1.logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.compat.v1.train.SaverDef()
    with tf.compat.v1.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.compat.v1.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def _tokenize(self, item):
    """Tokenizes an input string into a list of words."""
    tokenized = []
    for s in self._sentence_detector.tokenize(item):
      tokenized.extend(nltk.tokenize.word_tokenize(s))

    return tokenized

  def _word_to_embedding(self, w, word_to_embedding):
    """Returns the embedding of a word."""
    return word_to_embedding.get(w, word_to_embedding[UNK])

  def _preprocess(self, data, use_eos, word_to_embedding):
    """Preprocesses text for the encoder.

    Args:
      data: A list of input strings.
      use_eos: Whether to append the end-of-sentence word to each sentence.

    Returns:
      embeddings: A list of word embedding sequences corresponding to the input
        strings.
    """
    preprocessed_data = []
    for item in data:
      tokenized = self._tokenize(item)

      tokenized_word_embs = [self._word_to_embedding(w, word_to_embedding) for w in tokenized]

      preprocessed_data.append(tokenized_word_embs)

    return preprocessed_data

  def encode(self,
             sess,
             data,
             use_norm=True,
             verbose=True,
             batch_size=128,
             use_eos=False):
    """Encodes a sequence of sentences as skip-thought vectors.

    Args:
      sess: TensorFlow Session.
      data: A list of input strings.
      use_norm: Whether to normalize skip-thought vectors to unit L2 norm.
      verbose: Whether to log every batch.
      batch_size: Batch size for the encoder.
      use_eos: Whether to append the end-of-sentence word to each input
        sentence.

    Returns:
      thought_vectors: A list of numpy arrays corresponding to the skip-thought
        encodings of sentences in 'data'.
    """ 
    thought_vectors = []
    feed_data = []

    vocab_configs = self.config.vocab_configs
    assert(2 * len(vocab_configs) == len(self._embeddings))

    for i in range(len(vocab_configs)):
      v = vocab_configs[i]
      if v.mode == "expand":
        for j, inout in zip([0, 1], ["", "_out"]):
          data_proc = self._preprocess(data, use_eos, self._embeddings[2*i+j])
          feed_data.append((v.name + inout + ":0", data_proc, True))

      elif v.mode == "trained" or v.mode == "fixed":
        data_proc = self._preprocess(data, use_eos, self._embeddings[2*i])
        feed_data.append(("encode_ids:0", data_proc, False))

    batch_indices = np.arange(0, len(data), batch_size)
    for batch, start_index in enumerate(batch_indices):
      if verbose:
        tf.compat.v1.logging.info("Batch %d / %d.", batch, len(batch_indices))

      feed_dict = {}
      mask = None
      for feed_d in feed_data:
        data = feed_d[1]
        name = feed_d[0]
        if feed_d[-1]:
          embs, mask = _batch_and_pad_embs(
              data[start_index:start_index + batch_size])
          feed_dict[name] = embs
        else:
          batch_ids, mask = _batch_and_pad(
              data[start_index:start_index + batch_size])
          feed_dict[name] = batch_ids
      feed_dict["encode_mask:0"] = mask

      enc, enc_out = sess.run(
	  ["encoder/thought_vectors:0", "encoder_out/thought_vectors:0"],
          feed_dict=feed_dict)

      if use_norm:
        enc = [v / np.linalg.norm(v) for v in enc]
        enc_out = [v / np.linalg.norm(v) for v in enc_out]
        enc_cat = np.concatenate((enc, enc_out), axis=1)

        thought_vectors.extend(enc_cat)
      else:
        thought_vectors.extend(np.concatenate((enc, enc_out), axis=1))

    return thought_vectors

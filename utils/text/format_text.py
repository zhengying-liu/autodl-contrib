# Author: Zhengying LIU
# Creation date: 30 Sep 2018
# Description: format text datasets to TFRecords (SequenceExample proto)
#   for AutoDL challenge.
"""To generate an AutoDL dataset from text datasets, run a command line (in the
current directory) with something like:
`python format_text.py -input_dir='../../raw_datasets/text/' -output_dir='../../formatted_datasets/' -dataset_name='20newsgroup' -max_num_examples_train=None -max_num_examples_test=None`

"""

import tensorflow as tf
import numpy as np
import os
import sys

import urllib
import zipfile

# NLP
import nltk
from sklearn.datasets import fetch_20newsgroups

EMBEDDING_DIMENSION = 50
GLOVE_DIR = '/usr/local/share/glove'
GLOVE_WEIGHTS_FILE_PATH = os.path.join(GLOVE_DIR,
                                       f'glove.6B.{EMBEDDING_DIMENSION}d.txt')

tf.flags.DEFINE_string('input_dir', '../../raw_datasets/text/',
                       "Directory containing text datasets.")

tf.flags.DEFINE_string('dataset_name', '20newsgroup', "Basename of dataset.")

tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                       "Output data directory.")

tf.flags.DEFINE_string('max_num_examples_train', 'None',
                       "Number of examples in training set we want to format.")

tf.flags.DEFINE_string('max_num_examples_test', 'None',
                       "Number of examples in test set we want to format.")

tf.flags.DEFINE_string('num_shards_train', '1', # TODO: sharding feature is not implemented yet
                       "Number of shards for training set.")

tf.flags.DEFINE_string('num_shards_test', '1',
                       "Number of shards for training set.")

FLAGS = tf.flags.FLAGS

verbose = False

def get_text_label_pairs(dataset_name):
  """
  Return:
    pairs_train:  an iterable of (text, labels) pairs for training set, where
      `text` is a string and `labels` is a list of integers.
    pairs_test:   same as pairs_train, but for test set.
  """
  if dataset_name == '20newsgroup':
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',
                                     shuffle=True,
                                     random_state=42)
    text_train = twenty_train.data
    labels_train = [[label] for label in twenty_train.target] # should be a list
    text_test = twenty_test.data
    labels_test = [[label] for label in twenty_test.target]
    pairs_train = zip(text_train, labels_train)
    pairs_test = zip(text_test, labels_test)
    return pairs_train, pairs_test
  else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")


def download_GloVe_pretrained_weights():
  if not os.path.isdir(GLOVE_DIR):
    print(f"Creating directory {GLOVE_DIR}")
    os.mkdir(GLOVE_DIR)
  if not os.path.isfile(GLOVE_WEIGHTS_FILE_PATH):
    # Glove embedding weights can be downloaded from https://nlp.stanford.edu/projects/glove/
    glove_fallback_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    local_zip_file_path = os.path.join(GLOVE_DIR, os.path.basename(glove_fallback_url))
    if not os.path.isfile(local_zip_file_path):
      print(f'Retreiving glove weights from {glove_fallback_url}')
      urllib.request.urlretrieve(glove_fallback_url, local_zip_file_path)
      with zipfile.ZipFile(local_zip_file_path, 'r') as z:
        print(f'Extracting glove weights from {local_zip_file_path}')
        z.extractall(path=GLOVE_DIR)
  word2idx = {}
  idx2word = []
  weights = []
  print("Construct GloVe weight matrix...")
  with open(GLOVE_WEIGHTS_FILE_PATH, 'r') as f:
    for index, line in enumerate(f):
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      word2idx[word] = index
      idx2word.append(word)
      weights.append(coefs)
  weights = np.array(weights)
  return word2idx, idx2word, weights

def tokenize(document):
  words = nltk.word_tokenize(document.lower())
  return words

def word2vec(word):
  """
  Global variables:
    word2idx, weights
  """
  if word in word2idx:
    idx = word2idx[word]
    return weights[idx]
  else: # if word is unknown, return 0 vector
    return np.zeros(EMBEDDING_DIMENSION)

def get_unknown_words(document):
  """Helper function to list all unknown words to see if the word dictionary is
  general enough.

  Global variables:
    word2idx
  """
  words = tokenize(document)
  unknown_words = [w for w in words if w not in word2idx]
  unknown_rate = len(unknown_words) / len(words)
  return unknown_rate, unknown_words

def doc2vecs(document, window_size=3, strides=2):
  """
  Global variables:
    tokenize, word2vec
  """
  words = tokenize(document)
  num_words = len(words)
  if window_size > num_words:
    print("WARNING: window size should be smaller than document length!")
  if strides > window_size:
    print("WARNING: strides larger than window size! This may cause information loss.")
  vecs = []
  for i in range(0, num_words - window_size + 1, strides):
    vec = np.zeros(EMBEDDING_DIMENSION)
    for j in range(i, i + window_size):
      vec += word2vec(words[j])
    vecs.append(vec)
  vecs = np.array(vecs)
  return vecs

def get_features_labels_pairs(dataset_name):
  pass


if __name__ == '__main__':
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  output_dir = FLAGS.output_dir
  try:
    max_num_examples_train = int(FLAGS.max_num_examples_train)
  except:
    print("Couldn't parse max_num_examples_train...setting to None.")
    max_num_examples_train = None
  try:
    max_num_examples_test = int(FLAGS.max_num_examples_test)
  except:
    print("Couldn't parse max_num_examples_test...setting to None.")
    max_num_examples_test = None

  pairs_train, pairs_test = get_text_label_pairs(dataset_name)
  word2idx, idx2word, weights =\
    download_GloVe_pretrained_weights()
  for doc, label in pairs_train:
    print(doc)
    print(doc2vecs(doc).shape)
    print(doc2vecs(doc))
    break

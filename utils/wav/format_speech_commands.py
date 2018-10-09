# Author: Zhengying LIU
# Creation date: 9 Oct 2018
"""Generate AutoDL datasets (SequenceExample TFRecords) from Speech Commands
dataset: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
Run
`python format_speech_commands.py `
to generate TIMIT datasets in AutoDL format.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter
from format_timit import wav_to_data
from scipy.io import wavfile
from sklearn.utils import shuffle
np.random.seed(42)

def get_speech_commands_info_df(dataset_dir, tmp_dir='/tmp/', from_scratch=False, classes=None):
  """Format Speech Commands dataset to AutoDL format.
  """
  csv_filepath = os.path.join(tmp_dir, 'speech_commands_info.csv')
  if not from_scratch and os.path.isfile(csv_filepath):
    info_df = pd.read_csv(csv_filepath)
    print("Successfully loaded existing info table. Now life is easier.")
    return info_df
  else:
    print("Couldn't load existing info table. Now building from scatch...")
  path = os.path.abspath(dataset_dir)
  li = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if filename.endswith('.wav'):
        label = dirpath.split(os.sep)[-1]
        if label != '_background_noise_':
          ext_filename = os.path.join(label, filename)
          assert(os.path.isfile(os.path.join(dataset_dir, ext_filename)))
          li.append((ext_filename, label))
        else:
          print("Background noise file! Passing...")
  info_df = pd.DataFrame({'ext_filename':        [x[0] for x in li],
                     'label':               [x[1] for x in li]})
  info_df['label'] = info_df['label'].astype('category')
  test_df = pd.read_csv(os.path.join(dataset_dir, 'testing_list.txt'), header=None)
  valid_df = pd.read_csv(os.path.join(dataset_dir, 'validation_list.txt'), header=None)
  ext_filenames_test = set(test_df[0])
  ext_filenames_valid = set(valid_df[0])
  def get_subset(ext_filename):
    if ext_filename in ext_filenames_test:
      return 'test'
    elif ext_filename in ext_filenames_valid:
      return 'valid'
    else:
      return 'train'
  info_df['subset'] = info_df['ext_filename'].apply(get_subset).astype('category')
  info_df.to_csv(csv_filepath, index=False)
  return info_df

def get_processed_df(info_df, classes=None, proba_keep=1.0, shuffled=True):
  """
  Consider only those examples with label in `classes`

  Args:
    classes: an iterable of class names, should be a subset of
    ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow']
  """
  if classes:
    info_df = info_df.loc[info_df['label'].isin(classes)]
  if proba_keep < 1.0:
    info_df = info_df.loc[np.random.rand(len(info_df)) < proba_keep]
  if shuffled:
    info_df = shuffle(info_df)
  processed_df = info_df.copy()
  processed_df['label'] = processed_df['label'].astype('category')
  processed_df['label_num'] = processed_df['label'].cat.codes
  return processed_df

def get_features_labels_pairs_generator(processed_df, subset='train'):
  """Get generator of (features, labels) pairs to be used for
  dataset_formatter.UniMediaDatasetFormatter.
  """
  def index_row_to_features_labels_pair(index_row):
    index, row = index_row
    ext_filename = row['ext_filename']
    wav_filepath = os.path.join(FLAGS.dataset_dir, ext_filename)
    features = wav_to_data(wav_filepath)
    features = [[x] for x in features]
    labels = row['label_num']
    labels = [labels]
    return features, labels
  subset_s = processed_df['subset']
  if subset=='train':
    processed_df = processed_df.loc[(subset_s=='train') | (subset_s=='valid')]
  elif subset=='test':
    processed_df = processed_df.loc[(subset_s=='test')]
  else:
    raise ValueError("Wrong subset key! Should be 'train' or 'test'.")
  index_row_generator = processed_df.iterrows
  features_labels_generator = lambda: map(index_row_to_features_labels_pair, index_row_generator())
  return features_labels_generator

def main():
  dataset_dir = FLAGS.dataset_dir
  tmp_dir = FLAGS.tmp_dir
  output_dir = FLAGS.output_dir
  info_df = get_speech_commands_info_df(dataset_dir, from_scratch=False)
  classes = ['zero', 'one', 'two', 'three', 'four',
             'five', 'six', 'seven', 'eight', 'nine']
  proba_keep = 1
  shuffled = True
  processed_df = get_processed_df(info_df, classes=classes, proba_keep=proba_keep, shuffled=shuffled)
  features_labels_pairs_train =\
    get_features_labels_pairs_generator(processed_df, subset='train')
  features_labels_pairs_test =\
    get_features_labels_pairs_generator(processed_df, subset='test')

  row_count = 1
  col_count = 1
  output_dim = len(classes)
  dataset_name = 'Speech Commands'
  new_dataset_name = 'santaclaus'
  dataset_formatter =  UniMediaDatasetFormatter(dataset_name,
                                                output_dir,
                                                features_labels_pairs_train,
                                                features_labels_pairs_test,
                                                output_dim,
                                                col_count,
                                                row_count,
                                                sequence_size=None, # for strides=2
                                                num_examples_train=None,
                                                num_examples_test=None,
                                                is_sequence_col='false',
                                                is_sequence_row='false',
                                                has_locality_col='true',
                                                has_locality_row='true',
                                                format='DENSE',
                                                is_sequence='false',
                                                sequence_size_func=max,
                                                new_dataset_name=new_dataset_name)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

if __name__ == '__main__':
  tf.flags.DEFINE_string('dataset_dir', '../../raw_datasets/speech/speech_commands_v0.01/',
                         "Directory containing the whole Speech Commands dataset.")

  tf.flags.DEFINE_string("tmp_dir", "/tmp/", "Temporary directory.")

  tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/",
                         "Output data directory.")

  tf.flags.DEFINE_string('max_num_examples_train', '120', #TODO: to be changed.
                         "Number of examples in training set we want to format.")

  tf.flags.DEFINE_string('max_num_examples_test', '100', #TODO: to be changed.
                         "Number of examples in test set we want to format.")

  tf.flags.DEFINE_string('num_shards', '1', "Number of shards.")

  FLAGS = tf.flags.FLAGS
  main()

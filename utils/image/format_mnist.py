# Author: Zhengying LIU
# Creation date: 15 Oct 2018
"""Generate AutoDL datasets (SequenceExample TFRecords) from MNIST.

Run
`python format_mnist.py`
to generate MNIST dataset in AutoDL format.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter
from tensorflow.contrib.learn.python.learn.datasets import mnist

def get_features_labels_pairs_generator(subset='train'):
  """Get generator of (features, labels) pairs to be used for
  dataset_formatter.UniMediaDatasetFormatter.
  """
  datasets = mnist.read_data_sets(train_dir='/tmp/data/', validation_size=0)
  if subset == 'train':
    features = datasets.train.images
    labels = datasets.train.labels
  else:
    features = datasets.test.images
    labels = datasets.test.labels
  features = [[x] for x in features]
  labels = [[x] for x in labels] # each item in labels should be a list
  return lambda: (x for x in zip(features, labels))

def main():
  output_dir = FLAGS.output_dir
  classes = ['zero', 'one', 'two', 'three', 'four',
             'five', 'six', 'seven', 'eight', 'nine']

  row_count = 28
  col_count = 28
  output_dim = len(classes)
  dataset_name = 'MNIST'
  new_dataset_name = 'munster'
  classes_dict = {s:i for i, s in enumerate(classes)}
  features_labels_pairs_train =\
      get_features_labels_pairs_generator(subset='train')
  features_labels_pairs_test =\
      get_features_labels_pairs_generator(subset='test')
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
                                                new_dataset_name=new_dataset_name,
                                                classes_dict=classes_dict)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

if __name__ == '__main__':
  tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/",
                         "Output data directory.")
  FLAGS = tf.flags.FLAGS
  main()

# Author: Zhengying LIU
# Creation date: 21 Sep 2018
# Description: for formatted AutoDL datasets, inspect, retrieve information
#   and check its integrety

import tensorflow as tf
import os
import sys
# Directory containing code defining AutoDL dataset: dataset.py, data.proto,
# etc. You should change this line if the default directory doesn't exist,
# typically when you didn't clone the whole git repo.
definition_dir = '../../tfrecord_format/autodl_format_definition'
sys.path.append(definition_dir)
from dataset import AutoDLDataset

# Add flags for command line argument parsing
tf.flags.DEFINE_string('input_dir', '../../formatted_datasets/',
                       "Directory containing formatted AutoDL datasets.")

tf.flags.DEFINE_string('dataset_name', 'adult_600_100', "Basename of dataset.")

tf.flags.DEFINE_string('definition_dir',
                       '../../tfrecord_format/autodl_format_definition',
                       "Basename of dataset.")

FLAGS = tf.flags.FLAGS

def get_train_and_test_data(input_dir, dataset_name):
  train_path = os.path.join(input_dir, dataset_name, 'train')
  test_path = os.path.join(input_dir, dataset_name, 'test')
  D_train = AutoDLDataset(train_path)
  D_train.init()
  D_test = AutoDLDataset(test_path)
  D_test.init()
  return D_train, D_test

def test():
  D_train, D_test = get_train_and_test_data(input_dir, dataset_name)
  print(D_train.get_metadata())

if __name__ == "__main__":
  print("haha", os.listdir('.'))
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  test()

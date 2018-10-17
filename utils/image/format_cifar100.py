# Author: Zhengying LIU
# Creation date: 15 Oct 2018
"""Generate AutoDL datasets (SequenceExample TFRecords) from CIFAR-100 dataset.

Run
`python format_cifar100.py`
to generate CIFAR-100 dataset in AutoDL format.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter

TASK_TYPE = 'multiclass'
assert(TASK_TYPE in ['multilabel', 'multiclass'])

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_features_labels_pairs_generator(subset='train'):
  """Get generator of (features, labels) pairs to be used for
  dataset_formatter.UniMediaDatasetFormatter.
  """
  if subset == 'train':
    data_dict = train_dict
  else:
    data_dict = test_dict
  images = data_dict[b'data']
  coarse_labels = data_dict[b'coarse_labels']
  if TASK_TYPE == 'multilabel':
    translation = len(coarse_label_names) # 20
  else: # TASK_TYPE == 'multiclass'
    translation = 0
  fine_labels = [x + translation for x in data_dict[b'fine_labels']]
  num_examples = images.shape[0]
  rgb = images.reshape(num_examples, 3, 32, 32).transpose([0, 2, 3, 1])
  VECT = [0.299, 0.587, 0.114]
  gray = rgb.dot(VECT) # Convert to gray scale
  features = gray.reshape(num_examples, 32*32)

  features = [[x] for x in features]
  if TASK_TYPE == 'multilabel':
    labels = list(zip(coarse_labels, fine_labels))
  else: # TASK_TYPE == 'multiclass'
    labels = [[x] for x in fine_labels]

  return lambda: (x for x in zip(features, labels))


if __name__ == '__main__':
  tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/",
                         "Output data directory.")
  FLAGS = tf.flags.FLAGS
  output_dir = FLAGS.output_dir
  root_dir = '../../raw_datasets/image/cifar-100-python/'
  metadata_dict = unpickle(root_dir + 'meta')
  train_dict = unpickle(root_dir + 'train')
  test_dict = unpickle(root_dir + 'test')

  fine_label_names = metadata_dict[b'fine_label_names']
  fine_label_names = [x.decode('utf-8') for x in fine_label_names]
  coarse_label_names = metadata_dict[b'coarse_label_names']
  coarse_label_names = [x.decode('utf-8') for x in coarse_label_names]

  if TASK_TYPE == 'multilabel':
    classes_list = coarse_label_names + fine_label_names
    new_dataset_name = 'ciao'
  else: # TASK_TYPE == 'multiclass'
    classes_list = fine_label_names
    new_dataset_name = 'chao'

  row_count = 32
  col_count = 32
  output_dim = len(classes_list)
  dataset_name = 'CIFAR-100'

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
                                                classes_list=classes_list)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

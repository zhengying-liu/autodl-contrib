# Author: Zhengying LIU
# Creation date: 1 Mar 2018
"""Generate AutoDL datasets (SequenceExample TFRecords) from CIFAR-10 dataset.

Run
`python format_cifar10.py`
to generate CIFAR-10 dataset in AutoDL format.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import glob
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter
from format_utils import download_and_extract_archive, check_integrity

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
    data_dicts = train_dicts
  else:
    data_dicts = test_dicts
  batches = [data_dict[b'data'] for data_dict in data_dicts]
  images = np.concatenate(batches, axis=0)
  labels = np.concatenate([data_dict[b'labels'] for data_dict in data_dicts],
                          axis=0)

  num_examples = images.shape[0]
  rgb = images.reshape(num_examples, 3, 32, 32).transpose([0, 2, 3, 1])
  features = rgb.reshape(num_examples, 32*32*3)

  features = [[x] for x in features]
  labels = [[x] for x in labels]

  return lambda: (x for x in zip(features, labels))


if __name__ == '__main__':
  tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/",
                         "Output data directory.")
  FLAGS = tf.flags.FLAGS
  output_dir = FLAGS.output_dir
  root_dir = '../../raw_datasets/image/'

  base_folder = 'cifar-10-batches-py'
  url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  filename = "cifar-10-python.tar.gz"
  tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
  train_list = [
      ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
      ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
      ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
      ['data_batch_4', '634d18415352ddfa80567beed471001a'],
      ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
  ]

  test_list = [
      ['test_batch', '40351d587109b95175f43aff81a1287e'],
  ]
  meta = {
      'filename': 'batches.meta',
      'key': 'label_names',
      'md5': '5ff9c542aee3614f3951f8cda6e48888',
  }

  fpath = os.path.join(root_dir, base_folder, filename)

  def _check_integrity(root, base_folder, train_list, test_list):
    for fentry in (train_list + test_list):
      filename, md5 = fentry[0], fentry[1]
      fpath = os.path.join(root, base_folder, filename)
      if not check_integrity(fpath, md5):
        return False
    return True

  if _check_integrity(root_dir, base_folder, train_list, test_list):
    print('Files already downloaded and verified')
  else:
    download_and_extract_archive(url, root_dir, filename=filename, md5=tgz_md5)

  base_dir = os.path.join(root_dir, base_folder)

  metadata_file = os.path.join(base_dir, 'batches.meta')
  train_files_glob = os.path.join(base_dir, 'data_batch_*')
  test_file = os.path.join(base_dir, 'test_batch')

  metadata_dict = unpickle(metadata_file)
  train_dicts = [unpickle(train_file) for train_file in sorted(glob.glob(train_files_glob))]
  test_dicts = [unpickle(test_file)]

  label_names = metadata_dict[b'label_names']
  label_names = [x.decode('utf-8') for x in label_names]

  classes_list = label_names
  new_dataset_name = 'cifar10'

  sequence_size = 1
  row_count = 32
  col_count = 32
  num_channels = 3
  output_dim = len(classes_list)

  num_examples_train = 50000
  num_examples_test = 10000

  dataset_name = 'CIFAR-10'

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
                                                sequence_size=sequence_size,
                                                num_channels=num_channels,
                                                num_examples_train=num_examples_train,
                                                num_examples_test=num_examples_test,
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

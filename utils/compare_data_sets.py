# Author: Zhengying LIU
# Date: 4 Apr 2019
"""Verify if two datasets (one in File Format and another in TFRecord Format)
have the same examples, each with the same labels.

Usage:
  python compare_data_sets.py -file_dataset_dir=DATASET_DIR_1 -tfrecord_dataset_dir=DATASET_DIR_2
"""

import tensorflow as tf
from dataset_manager import compare_datasets

def main(*argv):
  """Do you really need a docstring?"""

  tf.flags.DEFINE_string('file_dataset_dir', '../formatted_datasets/Caucase_file_format',
                         "Path to dataset in File Format.")
  tf.flags.DEFINE_string('tfrecord_dataset_dir', '../formatted_datasets/Caucase',
                         "Path to dataset in TFRecord Format.")

  FLAGS = tf.flags.FLAGS
  del argv
  file_dataset_dir = FLAGS.file_dataset_dir
  tfrecord_dataset_dir = FLAGS.tfrecord_dataset_dir

  compare_datasets(file_dataset_dir, tfrecord_dataset_dir)

if __name__ == '__main__':
  main()

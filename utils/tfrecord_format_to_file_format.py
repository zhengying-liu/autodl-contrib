# Author: Zhengying Liu
# Date: 3 Apr 2019
"""Convert a dataset in TFRecord Format to File Format. New dataset will be
located in the same parent directory as old dataset.

Usage:
  python tfrecord_format_to_file_format.py -dataset_dir=../formatted_datasets/Hammer
"""
import tensorflow as tf
from dataset_manager import TFRecordFormatDataset

def main(*argv):
  """Do you really need a docstring?"""

  tf.flags.DEFINE_string('dataset_dir', '../formatted_datasets/Hammer',
                         "Path to dataset.")

  FLAGS = tf.flags.FLAGS
  del argv
  dataset_dir = FLAGS.dataset_dir

  tfrecord_format_dataset = TFRecordFormatDataset(dataset_dir)
  tfrecord_format_dataset.tfrecord_format_to_file_format()

if __name__ == '__main__':
  main()

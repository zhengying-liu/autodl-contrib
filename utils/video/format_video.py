# Author: Zhengying LIU
# Creation date: 1 Oct 2018
# Description: format video datasets to TFRecords (SequenceExample proto)
#   for AutoDL challenge.

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter

from pprint import pprint

tf.flags.DEFINE_string('input_dir', '../../raw_datasets/video/',
                       "Directory containing text datasets.")

tf.flags.DEFINE_string('dataset_name', 'kth', "Basename of dataset.")

tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                       "Output data directory.")

FLAGS = tf.flags.FLAGS

verbose = False

if __name__ == '__main__':
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  output_dir = FLAGS.output_dir
  print("haha")

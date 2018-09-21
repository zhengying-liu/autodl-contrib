# Author: Zhengying LIU
# Creation date: 21 Sep 2018
# Description: for formatted AutoDL datasets, inspect, retrieve information
#   and check its integrety

import tensorflow as tf

tf.flags.DEFINE_string('input_dir', '../../formatted_datasets/',
                       "Directory containing formatted AutoDL datasets.")

tf.flags.DEFINE_string("dataset_name", "adult", "Basename of dataset.")

def test():
  pass

if __name__ == "__main__":
  test()

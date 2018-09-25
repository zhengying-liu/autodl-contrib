# Author: Zhengying LIU
# Creation date: 20 Sep 2018
# Description: for a give AutoDL dataset, test it with baseline method in
#   the starting kit of the AutoDL competition bundle
"""Run `bash test_autodl_dataset.sh <path_to_autodl_dataset>` to test the AutoDL
dataset.
"""

import tensorflow as tf


tf.flags.DEFINE_string('dataset_dir', '../../formatted_datasets/adult_600_100/',
                       "Directory containing the formatted AutoDL dataset.")

tf.flags.DEFINE_string('starting_kit_dir', '../../../autodl/codalab_competition_bundle/AutoDL_starting_kit/',
                       "Directory containing ingestion program "
                       "`AutoDL_ingestion_program/`, "
                       "scoring program `AutoDL_scoring_program`, "
                       "and the starting kit code "
                       "`AutoDL_sample_code_submission/`.")

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  dataset_dir = FLAGS.dataset_dir
  starting_kit_dir = FLAGS.starting_kit_dir

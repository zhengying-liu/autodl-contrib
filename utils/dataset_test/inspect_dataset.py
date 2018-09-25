# Author: Zhengying LIU
# Creation date: 21 Sep 2018
# Description: for formatted AutoDL datasets, inspect, retrieve information
#   and check its integrety

import tensorflow as tf
import pandas as pd
import yaml
import os
import sys
# Directory containing code defining AutoDL dataset: dataset.py, data.proto,
# etc. You should change this line if the default directory doesn't exist on
# your disk, typically when you didn't clone the whole git repo.
definition_dir = '../../tfrecord_format/autodl_format_definition'
sys.path.append(definition_dir)
from dataset import AutoDLDataset

from pprint import pprint

# Add flags for command line argument parsing
tf.flags.DEFINE_string('input_dir', '../../formatted_datasets/',
                       "Directory containing formatted AutoDL datasets.")

tf.flags.DEFINE_string('dataset_name', 'adult_600_100', "Basename of dataset.")

tf.flags.DEFINE_string('definition_dir',
                       '../../tfrecord_format/autodl_format_definition',
                       "Basename of dataset.")

FLAGS = tf.flags.FLAGS

verbose = True

def get_train_and_test_data(input_dir, dataset_name, repeat=False):
  train_path = os.path.join(input_dir, dataset_name,
                            dataset_name + '.data', 'train')
  test_path = os.path.join(input_dir, dataset_name,
                           dataset_name + '.data', 'test')
  D_train = AutoDLDataset(train_path)
  D_train.init(repeat=repeat)
  D_test = AutoDLDataset(test_path)
  D_test.init(repeat=repeat)
  return D_train, D_test

def print_metadata():
  D_train, D_test = get_train_and_test_data(input_dir, dataset_name)
  train_metadata = D_train.get_metadata()
  test_metadata = D_test.get_metadata()
  print("Training set's metadata:\n", train_metadata.metadata_)
  print("Test set's metadata:\n", test_metadata.metadata_)
  return train_metadata, test_metadata

def _len_feature_list(tf_feature_list):
  """Give a tensorflow.core.example.feature_pb2.FeatureList, return number of
  feature, i.e. its length.
  """
  return len(tf_feature_list.feature)

def _get_first_feature(tf_feature_list):
  return tf_feature_list.feature[0]

def _len_feature(tf_feature):
  """Give a tensorflow.core.example.feature_pb2.FeatureList, return its length.
  """
  assert(tf_feature)
  attrs = ['bytes_list', 'float_list', 'int64_list']
  for attr in attrs:
    if hasattr(tf_feature, attr):
      feature_vec = getattr(tf_feature, attr).value
      res = len(feature_vec)
      if res > 0:
        return res
  return 0

def extract_info_from_sequence_example(path_to_tfrecord, from_scratch=False):
  """Extract basic information for a given SequenceExample TFRecord.

  Args:
    path_to_tfrecord: path to a SequenceExample file. This SequenceExample
      should contain fields defined in `_parse_function` in `dataset.py`
  Returns:
    dataset_info: a dict containing some basic info on the dataset
    examples_info: a pandas.DataFrame object containing columns: `num_timestamps`, ``
      if sparse matrix,
  """
  assert(os.path.isfile(path_to_tfrecord))

  # The csv file containing extraction result
  output_dir = os.path.dirname(path_to_tfrecord)
  yaml_name = '.do_not_modify.dataset_info.yaml'
  csv_name = '.do_not_modify.example_info.csv'
  yaml_filepath = os.path.join(output_dir, yaml_name)
  csv_filepath = os.path.join(output_dir, csv_name)

  if not from_scratch \
    and os.path.isfile(yaml_filepath) \
    and os.path.isfile(csv_filepath):
    with open(yaml_filepath, 'r') as f:
      dataset_info = yaml.load(f)
    examples_info = pd.read_csv(csv_filepath)
    if verbose:
      print("Successfully loaded existing dataset info and examples info.")
    return dataset_info, examples_info
  else: # from scratch
    if verbose:
      print("Extracting dataset info and examples info from scratch.")

    # Some basic information on the dataset
    matrix_bundle_fields = []
    classes = set()
    # For now we only have dataset having 1 single bundle (e.g. no video+audio)
    num_bundles = 1
    num_classes = 0
    num_examples = 0
    sequence_size_max = 0
    sequence_size_min = 0
    sequence_size_median = 0
    is_sparse = None # True or False
    # Domain in ['audio_text_or_time_series', 'image_or_vector', 'video']
    # inferred_dataset_domain = None

    # Some basic information on each example
    num_timestamps = []
    num_features = []
    num_labels = []

    # Begin extracting
    counter = 0
    for se in tf.python_io.tf_record_iterator(path_to_tfrecord):
      sequence_example = tf.train.SequenceExample.FromString(se)

      context_feature = sequence_example.context.feature
      feature_lists_container = sequence_example.feature_lists.feature_list
      # Update num_labels
      labels = list(context_feature['label_index'].int64_list.value)
      num_labels.append(len(labels))

      if not matrix_bundle_fields:
        matrix_bundle_fields += list(feature_lists_container)
      else: # Make sure that fields name are consistent (coherent)
        assert(all([x in matrix_bundle_fields for x in feature_lists_container]))

      # Update classes
      classes = classes.union(set(labels))

      dense_key = '0_dense_input'
      sparse_value = '0_sparse_value'
      if dense_key in feature_lists_container:
        if is_sparse:
          raise ValueError("Inconsistent sparsity at index {}!".format(counter))
        elif is_sparse is None:
          is_sparse = False
        key = dense_key
      elif sparse_value in feature_lists_container:
        if is_sparse is not None:
          if not is_sparse:
            raise ValueError("Inconsistent sparsity at index {}!"\
                              .format(counter))
        else:
          is_sparse = True
        key = sparse_value

      # Update num_timestamps
      feature_list = feature_lists_container[key]
      num_timestamps.append(_len_feature_list(feature_list))
      # Update num_features
      feature_vec = _get_first_feature(feature_list)
      num_features.append(_len_feature(feature_vec))

      counter += 1

    examples_info = pd.DataFrame({'num_timestamps': num_timestamps,
                                  'num_features': num_features,
                                  'num_labels': num_labels})

    sequence_sizes = examples_info['num_timestamps']

    dataset_info = {'matrix_bundle_fields': matrix_bundle_fields,
                    'classes': list(classes),
                    'num_bundles': num_bundles,
                    'num_classes': len(classes),
                    'num_examples': examples_info.shape[0],
                    'sequence_size_max': sequence_sizes.max(),
                    'sequence_size_min': sequence_sizes.min(),
                    'sequence_size_median': sequence_sizes.median(),
                    'is_sparse': is_sparse
                    }
    examples_info.to_csv(csv_filepath, index=False)
    with open(yaml_filepath, 'w') as f:
      yaml.dump(dataset_info, f)
    return dataset_info, examples_info

def test_extract_info_from_sequence_example():
  path_to_tfrecord =\
    os.path.join(input_dir, dataset_name, dataset_name+'.data','train',
            'sample-adult-train.tfrecord')
  from_scratch = True
  dataset_info, examples_info =\
    extract_info_from_sequence_example(path_to_tfrecord,
                                       from_scratch=from_scratch)
  print('dataset_info:')
  pprint(dataset_info)
  print('examples_info:')
  print(examples_info)


if __name__ == "__main__":
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  # D_train, D_test = get_train_and_test_data(input_dir, dataset_name,)
  # num_examples_train = get_num_examples(D_train)
  # num_examples_test = get_num_examples(D_test)
  # print(num_examples_train)
  test_extract_info_from_sequence_example()

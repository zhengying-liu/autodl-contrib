# Author: Zhengying LIU
# Creation date: 20 Sep 2018
# Description: format datasets in AutoML format to TFRecords for AutoDL
"""Generate AutoDL datasets from datasets in AutoML format.

Run a command line (in the current directory) with something like:
`python format_automl.py -input_dir='../../raw_datasets/automl/' -output_dir='../../formatted_datasets/' -dataset_name=adult -max_num_examples_train=600 -max_num_examples_test=100`

Please change `input_dir` to the right directory on your disk containing the AutoML datasets.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import os
import sys
from pprint import pprint
sys.path.append('./ingestion_program/')
from data_manager import DataManager
from shutil import copyfile

tf.flags.DEFINE_string('input_dir', '../../raw_datasets/automl/',
                       "Directory containing all AutoML datasets.")

tf.flags.DEFINE_string("dataset_name", "adult", "Basename of dataset.")

tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/", "Output data directory.")

tf.flags.DEFINE_string('max_num_examples_train', '600',
                       "Number of examples in training set we want to format.")

tf.flags.DEFINE_string('max_num_examples_test', '100',
                       "Number of examples in test set we want to format.")

tf.flags.DEFINE_string('num_shards_train', '1', # TODO: sharding feature is not implemented yet
                       "Number of shards for training set.")

tf.flags.DEFINE_string('num_shards_test', '1',
                       "Number of shards for training set.")

FLAGS = tf.flags.FLAGS

verbose = False

class AutoMLMetadata():
  def __init__(self, dataset_name=None,
               sample_count=None,
               output_dim=None,
               set_type='train',
               col_count=None,
               row_count=1,      # each matrix in matrix bundle is a vector, so row_count=1
               sequence_size=1): # not time series
    self.dataset_name = dataset_name
    self.sample_count = sample_count
    self.output_dim = output_dim
    self.set_type = set_type
    self.col_count = col_count
    self.row_count = row_count
    self.sequence_size = sequence_size
    assert(set_type in ['train', 'test'])
  def __str__(self):
    return "AutoMLMetadata: {}".format(self.__dict__)
  def __repr__(self):
    return "AutoMLMetadata: {}".format(self.__dict__)

def is_sparse(obj):
  return scipy.sparse.issparse(obj)

def binary_to_multilabel(binary_label):
  return np.stack([1 - binary_label, binary_label], axis=1)

def regression_to_multilabel(regression_label, get_threshold=np.median):
  threshold = get_threshold(regression_label)
  binary_label = (regression_label > threshold)
  return binary_to_multilabel(binary_label)

def _prepare_metadata_features_and_labels(D, set_type='train'):
  data_format = D.info['format']
  task = D.info['task']
  if set_type == 'train':
    # Fetch features
    X_train = D.data['X_train']
    X_valid = D.data['X_valid']
    Y_train = D.data['Y_train']
    Y_valid = D.data['Y_valid']
    if is_sparse(X_train):
      concat = scipy.sparse.vstack
    else:
      concat = np.concatenate
    features = concat([X_train, X_valid])
    # Fetch labels
    labels = np.concatenate([Y_train, Y_valid])
  elif set_type == 'test':
    features = D.data['X_test']
    labels = D.data['Y_test']
  else:
    raise ValueError("Wrong set type, should be `train` or `test`!")
  # when the task if binary.classification or regression, transform it to multilabel
  if task == 'regression':
    labels = regression_to_multilabel(labels)
  elif task == 'binary.classification':
    labels = binary_to_multilabel(labels)
  # Generate metadata
  metadata = AutoMLMetadata(dataset_name=D.info['name'],
                            sample_count=features.shape[0],
                            output_dim=labels.shape[1],
                            set_type=set_type,
                            col_count=features.shape[1])
  return metadata, features, labels

def csr_feature_vector_to_lists(sparse_feature_vector):
  sparse_col_index = sparse_feature_vector.indices
  sparse_value = sparse_feature_vector.data
  sparse_row_index = np.zeros(len(sparse_col_index),dtype=int) # only 1 row, so row_index always 0
  return sparse_col_index, sparse_row_index, sparse_value

def dense_to_sparse_label(dense_label):
  """
  Args:
    dense_label: a 1-D vector
  Returns:
    label_indexes: a list of integers containing label indexes
    label_scores: a list of floats containing label scores
  """
  label_indexes = []
  label_scores = []
  for index, value in enumerate(dense_label):
    if value:
      label_indexes.append(index)
      label_scores.append(value)
  return label_indexes, label_scores

def _int64_feature(value):
  # Here `value` is a list of integers
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  # Here `value` is a list of bytes
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  # Here `value` is a list of floats
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _feature_list(feature):
  # Here `feature` is a list of tf.train.Feature
  return tf.train.FeatureList(feature=feature)

def print_first_sequence_example(path_to_tfrecord):
  for bytes in tf.python_io.tf_record_iterator(path_to_tfrecord):
    sequence_example = tf.train.SequenceExample.FromString(bytes)
    print(sequence_example)
    break

def _write_metadata_textproto(counter, metadata, D_info, filepath):
  # Write metadata.textproto
  sample_count = counter
  sequence_size = metadata.sequence_size
  output_dim = metadata.output_dim
  col_count = metadata.col_count
  if D_info['format'] == 'dense':
    _format = 'DENSE'
  else:
    _format = 'SPARSE'
  metadata_filename = 'metadata.textproto'
  metadata_filepath = os.path.join(os.path.dirname(filepath), metadata_filename)
  metadata_textproto = """is_sequence: false
sample_count: <sample_count>
sequence_size: <sequence_size>
output_dim: <output_dim>
matrix_spec {
  col_count: <col_count>
  row_count: 1
  is_sequence_col: false
  is_sequence_row: false
  has_locality_col: true
  has_locality_row: true
  format: <format>
}
"""
  metadata_textproto = metadata_textproto.replace('<sample_count>', str(sample_count))
  metadata_textproto = metadata_textproto.replace('<sequence_size>', str(sequence_size))
  metadata_textproto = metadata_textproto.replace('<output_dim>', str(output_dim))
  metadata_textproto = metadata_textproto.replace('<col_count>', str(col_count))
  metadata_textproto = metadata_textproto.replace('<format>', str(_format))
  with open(metadata_filepath, 'w') as f:
    f.write(metadata_textproto)

def convert_vectors_to_sequence_example(filepath, metadata, features, labels, D_info,
                                        max_num_examples=None, num_shards=1):
  """
  Args:
    metadata: an AutoMLMetadata object
    features: feature matrix, can be dense or sparse
    labels: an iterable of label arrays (or a matrix)
  Returns:
    Save a TFRecord to `filepath` and create a `metadata.textproto`
    file in the same directory.
  """
  assert(isinstance(labels, np.ndarray))
  dataset_name = metadata.dataset_name
  set_type = metadata.set_type
  is_test_set = (set_type == 'test')
  has_sparse_features = is_sparse(features)

  if is_test_set: # Save a solution file
    id_translation = 0
    solution_name = dataset_name + '.solution'
    solution_dir = os.path.abspath(os.path.dirname(filepath))
    solution_path = os.path.join(solution_dir, solution_name)
    if verbose:
      print("========= Writing solutions to: ", solution_path)
    np.savetxt(solution_path, labels, fmt='%.1f')
  else:
    id_translation = D_info['test_num']

  counter = 0
  with tf.python_io.TFRecordWriter(filepath) as writer:
    for feature_row, label_row in zip(features, labels):
      if is_test_set:
        label_index = _int64_feature([])
        label_score = _float_feature([])
      else:
        label_indexes, label_scores = dense_to_sparse_label(label_row)
        label_index = _int64_feature(label_indexes)
        label_score = _float_feature(label_scores)
      context_dict = {
          'id': _int64_feature([counter + id_translation]),
          'label_index': label_index,
          'label_score': label_score
      }

      if has_sparse_features:
        sparse_col_index, sparse_row_index, sparse_value =\
            csr_feature_vector_to_lists(feature_row)
        feature_list_dict = {
          '0_sparse_col_index': _feature_list([_int64_feature(sparse_col_index)]),
          '0_sparse_row_index': _feature_list([_int64_feature(sparse_row_index)]),
          '0_sparse_value': _feature_list([_float_feature(sparse_value)])
        }
      else:
        feature_list_dict={
          '0_dense_input': _feature_list([_float_feature(feature_row)])
        }

      context = tf.train.Features(feature=context_dict)
      feature_lists = tf.train.FeatureLists(feature_list=feature_list_dict)
      sequence_example = tf.train.SequenceExample(
          context=context,
          feature_lists=feature_lists)
      writer.write(sequence_example.SerializeToString())
      counter += 1
      if max_num_examples and counter  >= max_num_examples:
        break
  # Write metadata.textproto
  _write_metadata_textproto(counter, metadata, D_info, filepath)

def test():
  input_dir = '../../datasets/automl/' # Change this to the directory containing AutoML datasets
  if not os.path.isdir(input_dir):
    raise ValueError("input_dir not found. You can change this value in your ")
  small_datasets = ['jasmine', 'dexter', 'adult', 'cadata', 'arturo']
  filepath = './sample-haha'
  dataset_name = np.random.choice(small_datasets)
  set_type = np.random.choice(['train','test'])
  # set_type = 'test'
  D = DataManager(dataset_name, input_dir, replace_missing=False, verbose=verbose)
  D_info = D.info
  print("dataset_name={}, set_type={}, sparse or dense: {}".format(dataset_name, set_type, D.info['format']))
  metadata, features, labels = _prepare_metadata_features_and_labels(D, set_type=set_type)
  convert_vectors_to_sequence_example(filepath, metadata, features, labels, D_info,
                                          max_num_examples=None, num_shards=1)
  print_first_sequence_example(filepath)
  pprint(D.info)
  print("Now you should see 2 or 3 new files in current directory. :)")

def press_a_button_and_give_me_an_AutoDL_dataset(input_dir,
                                                 dataset_name,
                                                 output_dir,
                                                 max_num_examples_train,
                                                 max_num_examples_test,
                                                 num_shards_train,
                                                 num_shards_test):
  """Well there is actually not a button and instead you need to run a command
  line.

  Args:
    dataset_name: string, should be like `ada` or `nova`.
      (pay ATTENTION to nova dataset...weird dataset name in D.info)
      (AND waldo dataset doesn't have solutions for valid and test)
  """
  D = DataManager(dataset_name, input_dir, replace_missing=False, verbose=verbose)
  new_dataset_name = dataset_name

  if max_num_examples_train:
    new_dataset_name += '_' + str(max_num_examples_train)
  if max_num_examples_test:
    new_dataset_name += '_' + str(max_num_examples_test)

  dataset_dir = os.path.join(output_dir, new_dataset_name)
  if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)

  dataset_data_dir = os.path.join(dataset_dir, new_dataset_name+'.data')
  if not os.path.isdir(dataset_data_dir):
    os.mkdir(dataset_data_dir)

  # Format test set
  set_type = 'test'
  test_dir = os.path.join(dataset_data_dir, set_type)
  if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
  filepath = os.path.join(dataset_data_dir, set_type, "sample-{}-{}.tfrecord".format(dataset_name, set_type))
  metadata, features, labels = _prepare_metadata_features_and_labels(D, set_type=set_type)
  convert_vectors_to_sequence_example(filepath, metadata, features, labels, D.info,
                                      max_num_examples=max_num_examples_test)
  # Format training set
  set_type = 'train'
  train_dir = os.path.join(dataset_data_dir, set_type)
  if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
  filepath = os.path.join(dataset_data_dir, set_type, "sample-{}-{}.tfrecord".format(dataset_name, set_type))
  metadata, features, labels = _prepare_metadata_features_and_labels(D, set_type=set_type)
  convert_vectors_to_sequence_example(filepath, metadata, features, labels, D.info,
                                      max_num_examples=max_num_examples_train)

  # Move solution file to grand-parent directory
  solution_filepath = os.path.join(dataset_data_dir, 'test',
                                   dataset_name + '.solution')
  new_solution_filepath = os.path.join(dataset_dir,
                                   new_dataset_name + '.solution')
  try:
    os.rename(solution_filepath, new_solution_filepath)
  except Exception as e:
    print('WARNING: Unable to move '+solution_filepath)
    #log = open('log.txt', 'a')
    #log.write('Solution file not move: '+dataset_name+'\n')
    #log.close()
  
  # Copy original info file to formatted dataset
  try:
      for info_file_type in ['_public', '_private']:
          info_filepath = os.path.join(input_dir, dataset_name, dataset_name + info_file_type + '.info')
          new_info_filepath = os.path.join(dataset_dir, new_dataset_name + info_file_type + '.info')
          copyfile(info_filepath, new_info_filepath)
  except Exception as e:
      print('Unable to copy info files')
  
  return dataset_dir, new_dataset_name
  

if __name__ == '__main__':
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  output_dir = FLAGS.output_dir
  try:
    max_num_examples_train = int(FLAGS.max_num_examples_train)
  except:
    print("Couldn't parse max_num_examples_train...setting to None.")
    max_num_examples_train = None
  try:
    max_num_examples_test = int(FLAGS.max_num_examples_test)
  except:
    print("Couldn't parse max_num_examples_test...setting to None.")
    max_num_examples_test = None
  try:
    print("Couldn't parse num_shards_train...setting to 1.")
    num_shards_train = int(FLAGS.num_shards_train)
  except:
    num_shards_train = 1
  try:
    print("Couldn't parse num_shards_test...setting to 1.")
    num_shards_test = int(FLAGS.num_shards_test)
  except:
    num_shards_test = 1

  dataset_dir, new_dataset_name = press_a_button_and_give_me_an_AutoDL_dataset(
                                     input_dir,
                                     dataset_name,
                                     output_dir,
                                     max_num_examples_train,
                                     max_num_examples_test,
                                     num_shards_train,
                                     num_shards_test)

  print("Congratulations! You pressed a button and you created an AutoDL " +
        "dataset `{}` ".format(new_dataset_name) +
        "with {} maximum training examples".format(max_num_examples_train) +
        "and {} maximum test examples".format(max_num_examples_test) +
        "in the directory `{}`.".format(dataset_dir)
        )

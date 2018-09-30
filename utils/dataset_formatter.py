# Author: Zhengying LIU
# Creation date: 30 Sep 2018
# Description: API for formatting AutoDL datasets

import tensorflow as tf
import os
import numpy as np

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

def label_sparse_to_dense(li_label_nums, output_dim):
  dense_label = np.zeros(output_dim)
  for label_num in li_label_nums:
    dense_label[label_num] = 1
  return dense_label

def feature_sparse_to_dense(features): # TODO
  return features

class UniMediaDatasetFormatter():
  def __init__(self,
               dataset_name,
               output_dir,
               features_labels_pairs_train,
               features_labels_pairs_test,
               output_dim,
               col_count,
               row_count,
               sequence_size=None,
               is_sequence_col='false',
               is_sequence_row='false',
               has_locality_col='true',
               has_locality_row='true',
               format='DENSE',
               is_sequence='false'):
    # Dataset basename, e.g. `adult`
    self.dataset_name = dataset_name
    # Output directory, absolute path
    self.output_dir = os.path.abspath(output_dir)
    # Iterables containing (features, labels) pairs, where `features` is a list
    # of vectors in float. `labels` is a list of integers.
    self.feature_labels_pairs_train = feature_labels_pairs_train
    self.feature_labels_pairs_test = feature_labels_pairs_test
    # Some metadata on the dataset
    self.output_dim = output_dim
    self.col_count = col_count
    self.row_count = row_count
    if isinstance(sequence_size, int):
      self.sequence_size = sequence_size
    else:
      self.sequence_size = self.get_sequence_size(func=max)
    self.is_sequence_col = is_sequence_col
    self.is_sequence_row = is_sequence_row
    self.has_locality_col = has_locality_col
    self.has_locality_row = has_locality_row
    self.format = format
    self.is_sequence = is_sequence

    # Some computed properties
    self.dataset_dir = self.get_dataset_dir()
    self.dataset_data_dir = self.get_dataset_data_dir()
    self.num_examples_train = self.get_num_examples(set='train')
    self.num_examples_test = self.get_num_examples(set='test')

  def get_dataset_dir(self):
    dataset_dir = os.path.join(self.output_dir, self.dataset_name)
    return dataset_dir

  def get_dataset_data_dir(self):
    dataset_data_dir = os.path.join(self.dataset_dir,
                                    self.dataset_name + '.data')
    return dataset_data_dir

  def get_num_examples(self, set='train'):
    if set == 'train':
      data = self.feature_labels_pairs_train
    elif set == 'test':
      data = self.feature_labels_pairs_test
    else:
      raise ValueError("Wrong key `set`! Should be 'train' or 'test'.")
    if hasattr(data, '__len__'):
      return len(data)
    else:
      return sum([1 for x in data]) # This step could be slow.

  def get_metadata_filename(self, set='train'):
    filename = 'metadata.textproto'
    path = os.path.join(self.dataset_data_dir, set, filename)
    return path

  def get_data_filename(self, set='train'):
    filename = 'sample-' + dataset_name + '.tfrecord'
    path = os.path.join(self.dataset_data_dir, set, filename)
    return path

  def get_solution_filename(self): # solution file only for solution
    output_dir = self.output_dir
    dataset_name = self.dataset_name
    path = os.path.join(output_dir, dataset_name, dataset_name + '.solution')
    return path

  def get_sequence_size(self, func=max):
    length_train = [len(x) for x, _ self.feature_labels_pairs_train]
    length_test = [len(x) for x, _ self.feature_labels_pairs_test]
    length_all = length_train + length_test
    return func(length_all)

  def get_metadata(self, set='train'):
    metadata = """is_sequence: <is_sequence>
sample_count: <sample_count>
sequence_size: <sequence_size>
output_dim: <output_dim>
matrix_spec {
  col_count: <col_count>
  row_count: <row_count>
  is_sequence_col: <is_sequence_col>
  is_sequence_row: <is_sequence_row>
  has_locality_col: <has_locality_col>
  has_locality_row: <has_locality_row>
  format: <format>
}
"""
    if set == 'train':
      sample_count = self.num_examples_train
    else:
      sample_count = self.num_examples_test
    metadata = metadata.replace('<sample_count>', str(sample_count))
    metadata = metadata.replace('<is_sequence>', str(self.is_sequence))
    metadata = metadata.replace('<sequence_size>', str(self.sequence_size))
    metadata = metadata.replace('<col_count>', str(self.col_count))
    metadata = metadata.replace('<row_count>', str(self.row_count))
    metadata = metadata.replace('<is_sequence_col>', str(self.is_sequence_col))
    metadata = metadata.replace('<is_sequence_row>', str(self.is_sequence_row))
    metadata = metadata.replace('<has_locality_col>',str(self.has_locality_col))
    metadata = metadata.replace('<has_locality_row>',str(self.has_locality_row))
    metadata = metadata.replace('<format>', str(self.format))
    return metadata

  def write_tfrecord_and_metadata(self, set='train'):
    # Make directories if necessary
    if not os.path.isdir(self.output_dir):
      os.mkdir(self.output_dir)
    if not os.path.isdir(self.dataset_dir):
      os.mkdir(self.dataset_dir)
    if not os.path.isdir(self.dataset_data_dir):
      os.mkdir(self.dataset_data_dir)
    set_dir = os.path.join(self.dataset_data_dir, set)
    if not os.path.isdir(set_dir):
      os.mkdir(set_dir)

    # Write metadata
    path_to_metadata = self.get_metadata_filename(set=set)
    metadata = self.get_metadata()
    with open(path_to_metadata, 'r') as f:
      f.write(metadata)

    # Write TFRecords
    path_to_tfrecord = self.get_data_filename(set=set)
    is_test_set = (set == 'test')
    if is_test_set:
      id_translation = 0
      data = self.feature_labels_pairs_test
      num_examples = self.num_examples_test
    else:
      id_translation = self.num_examples_test
      data = self.feature_labels_pairs_train
      num_examples = self.num_examples_train

    counter = 0
    labels_array = np.zeros((num_examples, self.output_dim))
    with tf.python_io.TFRecordWriter(path_to_tfrecord) as writer:
      for features, labels in data:
        if is_test_set:
          label_index = _int64_feature([])
          label_score = _float_feature([])
          labels_array[counter] = label_sparse_to_dense(labels, self.output_dim)
        else:
          label_index = _int64_feature(labels)
          label_score = _float_feature([1]*len(labels))
        context_dict = {
            'id': _int64_feature([counter + id_translation]),
            'label_index': label_index,
            'label_score': label_score
        }

        if self.format == 'SPARSE':
          sparse_col_index, sparse_row_index, sparse_value =\
              feature_sparse_to_dense(features) # TODO
          feature_list_dict = {
            '0_sparse_col_index': _feature_list([_int64_feature(sparse_col_index)]),
            '0_sparse_row_index': _feature_list([_int64_feature(sparse_row_index)]),
            '0_sparse_value': _feature_list([_float_feature(sparse_value)])
          }
        elif self.format == 'DENSE':
          feature_list = [_float_feature(x) for x in features]
          feature_list_dict={
            '0_dense_input': _feature_list(feature_list)
          }
        else:
          raise ValueError(f"Wrong format key: {self.format}")

        context = tf.train.Features(feature=context_dict)
        feature_lists = tf.train.FeatureLists(feature_list=feature_list_dict)
        sequence_example = tf.train.SequenceExample(
            context=context,
            feature_lists=feature_lists)
        writer.write(sequence_example.SerializeToString())
        counter += 1

    # Write solution file
    if is_test_set:
      path_to_solution = self.get_solution_filename()
      np.savetxt(path_to_solution, labels_array, fmt='%.0f')

  def press_a_button_and_give_me_an_AutoDL_dataset(self):
    self.write_tfrecord_and_metadata(set='test')
    self.write_tfrecord_and_metadata(set='train')

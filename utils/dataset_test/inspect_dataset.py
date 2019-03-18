# Author: Zhengying LIU
# Creation date: 21 Sep 2018
# Description: for formatted AutoDL datasets, inspect, retrieve information
#   and check its integrity
"""To check the integrity of a dataset, run following command line:
`python inspect_dataset.py -input_dir='../../formatted_datasets/' -dataset_name=adult_600_100`
where you need to change `input_dir` and `dataset_name`.

After checking the integrity, 2 hidden files are be generated for each tfrecord
file in `train/` and also in `test/`. These hidden files contain information
extracted from pure iteration of examples. You can check these files if you want
to know more information on the dataset.
"""

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

tf.flags.DEFINE_string('original_dir', '../raw_datasets/automl/',
                       "Directory containing original tabular datasets.")

tf.flags.DEFINE_string('dataset_name', 'adult_600_100', "Basename of dataset.")

tf.flags.DEFINE_string('definition_dir',
                       '../../tfrecord_format/autodl_format_definition',
                       "Definition of AutoDL dataset"
                       "(with `dataset.py`, `data.proto` etc).")

FLAGS = tf.flags.FLAGS

verbose = True

def get_train_and_test_data(input_dir, dataset_name, repeat=False):
  """
  Returns:
    D_train, D_test: 2 AutoDLDataset objects (defined in `dataset.py`)
  """
  train_path = os.path.join(input_dir, dataset_name,
                            dataset_name + '.data', 'train')
  test_path = os.path.join(input_dir, dataset_name,
                           dataset_name + '.data', 'test')
  D_train = AutoDLDataset(train_path)
  D_train.init(repeat=repeat)
  D_test = AutoDLDataset(test_path)
  D_test.init(repeat=repeat)
  return D_train, D_test

def get_tfrecord_paths(input_dir, dataset_name):
  """For now this only works for num_shards = 1!"""
  tfrecord_glob_pattern = "sample*"
  data_path = os.path.join(input_dir, dataset_name, dataset_name + '.data')
  train_pattern = os.path.join(data_path, 'train', tfrecord_glob_pattern)
  test_pattern = os.path.join(data_path, 'test', tfrecord_glob_pattern)
  train_file = tf.gfile.Glob(train_pattern)[0]
  test_file = tf.gfile.Glob(test_pattern)[0]
  return train_file, test_file

def get_metadata(input_dir, dataset_name):
  """
  Returns:
   train_metadata, test_metadata: 2 AutoDLMetadata objects (defined in
    `dataset.py`)
  """
  D_train, D_test = get_train_and_test_data(input_dir, dataset_name)
  train_metadata = D_train.get_metadata()
  test_metadata = D_test.get_metadata()
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
      print("Extracting dataset info and examples info from scratch",
            "(by iterating the sequence examples)...")

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
    sequence_size_max = int(sequence_sizes.max())
    sequence_size_min = int(sequence_sizes.min())
    sequence_size_median = sequence_sizes.median()

    dataset_info = {'matrix_bundle_fields': matrix_bundle_fields,
                    'classes': list(classes),
                    'num_bundles': num_bundles,
                    'num_classes': len(classes),
                    'num_examples': examples_info.shape[0],
                    'sequence_size_max': sequence_size_max,
                    'sequence_size_min': sequence_size_min,
                    'sequence_size_median': sequence_size_median,
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


def compare_rows(path_to_tfrecord, dataset_name, number_to_check=5):
    """ Check if the first rows of the dataset are the same in TFRecord format and in original tabular data.
        WARNING: The code below is clearly not elegant.

        :param path_to_tfrecord: Path to TFRecord format dataset
        :param dataset_name: Name of the dataset, required to read its original tabular form
        :param number_to_check: Number of rows to compare
        :rtype: bool
        :return: True if every rows are consistent, False otherwise
    """
    consistent = True
    original_dir = FLAGS.original_dir

    try:
        # Read original tabular data
        train_data = pd.read_csv(original_dir+dataset_name+'/'+dataset_name+'_train.data', header=None, sep='\s+')

        counter = 0
        for se in tf.python_io.tf_record_iterator(path_to_tfrecord):

            # Number of examples to compare
            if counter < number_to_check:
                # Values of row in original tabular data
                row_values = list(train_data.iloc[counter])

                # TFRecord SequenceExample
                sequence_example = tf.train.SequenceExample.FromString(se)

                # Parsing values
                sestr = str(sequence_example.feature_lists)
                sestr = sestr.split('\n')
                sestr = [item for item in sestr if ('value:' in item)]

                row_values_tf = []
                for e in sestr:
                    row_values_tf.append(float(e.split(': ')[1]))

                # Comparing rows, if one is different then it is not consistent
                if(round(row_values[counter], 5) != round(row_values_tf[counter], 5)):
                    consistent = False
                    for i in range(10):
                        print('ERROR: TFRecords data is different from original data: {} != {}'.format(row_values, row_values_tf))

            counter += 1

    except Exception as e:
        print('WARNING: Unable read original tabular data, it may be SPARSE data.')
        print(e)
        log = open('log.txt', 'a')
        log.write('No first rows check: '+dataset_name+'\n')
        log.close()
        #for se in tf.python_io.tf_record_iterator(path_to_tfrecord):
        #    sequence_example = tf.train.SequenceExample.FromString(se)
        #    sestr = str(sequence_example.feature_lists)
        #    f = open('debug_{}.txt'.format(dataset_name), 'w') # DEBUG
        #    f.write(sestr) # write in file and compare by hand ?
        #    f.close()
        #    break

    return consistent


def check_integrity(input_dir, dataset_name, check_first_rows=False):
  train_metadata, test_metadata = get_metadata(input_dir, dataset_name)
  train_file, test_file = get_tfrecord_paths(input_dir, dataset_name)
  dataset_info_train, examples_info_train =\
    extract_info_from_sequence_example(train_file, from_scratch=False)
  dataset_info_test, examples_info_test =\
    extract_info_from_sequence_example(test_file, from_scratch=False)
  print("INTEGRITY CHECK: comparing existing metadata and inferred metadata...")
  # show training set info
  print("INTEGRITY CHECK: existing metadata for TRAINING:")
  print(train_metadata.metadata_)
  print("INTEGRITY CHECK: inferred metadata for TRAINING:")
  pprint(dataset_info_train)
  # show test set info
  print("INTEGRITY CHECK: existing metadata for TEST:")
  print(test_metadata.metadata_)
  print("INTEGRITY CHECK: inferred metadata for TEST:")
  pprint(dataset_info_test)

  # Check if first rows are the same in TFRecord format
  # than in original tabular data
  consistent_first_rows = True
  if check_first_rows:
    print("INTEGRITY CHECK: comparing first rows of TFRecords and original tabular data")
    consistent_first_rows = compare_rows(train_file, dataset_name)

  # Check the consistency on number of examples
  num_examples_existing_train = train_metadata.size()
  num_examples_inferred_train =  dataset_info_train['num_examples']
  num_examples_existing_test = test_metadata.size()
  num_examples_inferred_test =  dataset_info_test['num_examples']
  consistent_num_examples_train =\
    (num_examples_existing_train == num_examples_inferred_train)
  print("INTEGRITY CHECK: number of training examples: {} and {}"\
        .format(num_examples_existing_train, num_examples_inferred_train))
  print("INTEGRITY CHECK: num_examples_train consistent: ",
        consistent_num_examples_train)
  if not consistent_num_examples_train:
    for i in range(10):
      print("WARNING: inconsistent number of examples for training set!!!")
  consistent_num_examples_test =\
    (num_examples_existing_test == num_examples_inferred_test)
  print("INTEGRITY CHECK: number of test examples: {} and {}"\
        .format(num_examples_existing_test, num_examples_inferred_test))
  print("INTEGRITY CHECK: num_examples_test consistent: ",
        consistent_num_examples_test)
  if not consistent_num_examples_test:
    for i in range(10):
      print("WARNING: inconsistent number of examples for test set!!!")

  # Check the consistency on number of classes
  num_classes_existing_train = train_metadata.get_output_size()
  num_classes_inferred_train =  dataset_info_train['num_classes']
  num_classes_existing_test = test_metadata.get_output_size()
  num_classes_inferred_test =  dataset_info_test['num_classes']
  print("INTEGRITY CHECK: number of classes: {}, {}, {} and {}."\
        .format(num_classes_existing_train,
                num_classes_inferred_train,
                num_classes_existing_test,
                num_classes_inferred_test),
        "(it's normal to have number of classes = 0 for test",
        "since we don't know the true labels)")
  consistent_num_classes =\
    (num_classes_existing_train==num_classes_inferred_train) and \
    (num_classes_inferred_train==num_classes_existing_test)
  print("INTEGRITY CHECK: consistent number of classes:",
        consistent_num_classes)

  consistent_dataset = consistent_num_examples_train and \
                       consistent_num_examples_test and \
                       consistent_num_classes and \
                       consistent_first_rows
  if consistent_dataset:
    print("\nCongratulations! Your dataset is CONSISTENT with {}"\
          .format(num_examples_existing_train),
          "training examples, {} test examples,"\
          .format(num_examples_existing_test),
          "and {} classes.".format(num_classes_existing_train))
  else:
    print("Holy shit! Your dataset is NOT consistent!")
    log = open('log.txt', 'a')
    log.write('NOT CONSISTENT: '+dataset_name+'\n')
    if not consistent_num_examples_train:
      print("Inconsistent number of training examples: {} and {}"\
            .format(num_examples_existing_train, num_examples_inferred_train))
    if not consistent_num_examples_test:
      print("Inconsistent number of test examples: {} and {}"\
            .format(num_examples_existing_test, num_examples_inferred_test))
    if not consistent_num_classes:
      print("Inconsistent number of classes.",
            "But this might be due to too few examples",
            "and the program cannot infer num_classes correctly")
    if not consistent_first_rows:
      print('Inconsistent first rows. Train data were corrupted by the formatting !')
      log.write('Integrity issue: rows comparison.\n')
    log.close()
  return consistent_dataset, dataset_name, num_examples_existing_train,\
         num_examples_existing_test, num_classes_existing_train

def print_first_tensor(autodl_dataset):
  """
  Args:
    autodl_dataset: an AutoDLDataset object (defined in `dataset.py`)
  """
  tf_dataset = autodl_dataset.get_dataset()
  iterator = tf_dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  print(next_element)
  with tf.Session() as sess:
    value = sess.run(next_element)
  print(value)
  print("The shape of the first sampel is: ", [x.shape for x in value])
  return value


if __name__ == "__main__":
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  check_integrity(input_dir, dataset_name)

  # D_train, D_test =\
  #   get_train_and_test_data(input_dir, dataset_name, repeat=False)
  # print_first_tensor(D_test)

  # subset = 'train'
  # path_to_tfrecord = f'../../formatted_datasets/{dataset_name}/{dataset_name}.data/{subset}/sample-{dataset_name}-{subset}.tfrecord'
  # print(path_to_tfrecord)
  # dataset_info, examples_info =\
  #   extract_info_from_sequence_example(path_to_tfrecord)
  # pprint(dataset_info)
  # print(examples_info)
  # print(int(examples_info['num_timestamps'].quantile(0.95)))

# Author: Zhengying LIU
# Creation date: 18 Sep 2018
# Description: generate AutoDL datasets (SequenceExample TFRecords)
#              from TIMIT dataset.
"""Run
`python format_timit.py -level=sentence -max_num_examples_train=120 -max_num_examples_test=20`
to generate TIMIT datasets in AutoDL format.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from scipy.io import wavfile

tf.flags.DEFINE_string('timit_dir', '../../raw_datasets/speech/timit/',
                       "Directory containing the whole TIMIT dataset.")

tf.flags.DEFINE_string("tmp_dir", "/tmp/", "Temporary directory.")

tf.flags.DEFINE_string("output_dir", "../../formatted_datasets/",
                       "Output data directory.")

tf.flags.DEFINE_string('level', 'phonetic',
                       "Level of labels, must be one of "
                       "`phonetic`, `word` or `sentence`.")

tf.flags.DEFINE_string('max_num_examples_train', '120', #TODO: to be changed.
                       "Number of examples in training set we want to format.")

tf.flags.DEFINE_string('max_num_examples_test', '100', #TODO: to be changed.
                       "Number of examples in test set we want to format.")

tf.flags.DEFINE_string('num_shards', '1', "Number of shards.")

FLAGS = tf.flags.FLAGS

def get_timit_info_df(timit_dir, tmp_dir, from_scratch=False):
  filepath = os.path.join(tmp_dir, 'timit_files_info.csv')
  if not from_scratch and os.path.isfile(filepath):
    timit_df = pd.read_csv(filepath)
    print("Successfully loaded existing info table. Now life is easier.")
    return timit_df
  else:
    print("Couldn't load existing info table. Now building from scatch...")
  path = os.path.join(timit_dir, 'TIMIT/')
  li = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if filename.endswith('.WAV'):
        dir_split = dirpath.split(os.sep)
        data_type, region, speaker =  dir_split[6:9]
        gender = speaker[0]
        basename = filename.split('.')[0]
        phonetic_label = basename + '.PHN'
        word_label = basename + '.WRD'
        sentence_label = basename + '.TXT'
        li.append((dirpath, filename, phonetic_label, word_label,
                   sentence_label, data_type, region, speaker, gender))
  timit_df = pd.DataFrame({'dirpath':        [x[0] for x in li],
                           'wavfile':        [x[1] for x in li],
                           'phonetic_label': [x[2] for x in li],
                           'word_label':     [x[3] for x in li],
                           'sentence_label': [x[4] for x in li],
                           'data_type':      [x[5] for x in li],
                           'region':         [x[6] for x in li],
                           'speaker':        [x[7] for x in li],
                           'gender':         [x[8] for x in li]})
  timit_df.to_csv(filepath, index=False)
  return timit_df

def parse_label_file(label_file):
  """Parse a label file in TIMIT dataset to a pandas.DataFrame object.

  Returns:
    a pandas.DataFrame object containing 3 columns: begin, end, label
  """
  with open(label_file, 'r') as f:
    lines = f.readlines()
  begins = [int(line.split(' ')[0]) for line in lines]
  ends = [int(line.split(' ')[1]) for line in lines]
  labels = [' '.join(line.split(' ')[2:])[:-1] for line in lines]
  df = pd.DataFrame({'begin': begins, 'end': ends, 'level_label': labels})
  df['level_label'] = df['level_label'].astype('category')
  df['label_file'] = label_file
  return df

def get_level_label_df(level, timit_df):
  li = []
  for index, row in timit_df.iterrows():
    label_file = os.path.join(row['dirpath'], row[level + '_label'])
    li.append(parse_label_file(label_file))
  level_label_df = pd.concat(li, ignore_index=True)
  return level_label_df

def total_num_class(categorical_labels_df):
  df = categorical_labels_df
  for col in df:
      df[col] = df[col].astype('category')
  nums_categories = [len(df[col].cat.categories) for col in df.columns]
  return sum(nums_categories)

def cat_to_num(categorical_labels_df):
  """Convert a pd.DataFrame object having only categorical columns to
  a pd.DataFrame object having only integer values.

  This can be considered as an extension of `index encoding` for several
  columns at the same time: firstly, each categorical column is  converted
  to integer value independently then translated by the sum of numbers of
  categories of all columns on the right.
  """
  df = categorical_labels_df
  for col in df:
      df[col] = df[col].astype('category')
  nums_categories = [len(df[col].cat.categories) for col in df.columns]
  translation = 0
  li =[]
  for idx, col in enumerate(df.columns):
    assert(str(df[col].dtype) == 'category')
    translated_codes = df[col].cat.codes + translation # to avoid index conflict
    translated_codes = translated_codes.rename('label' + str(idx))
    li.append(translated_codes)
    translation += nums_categories[idx]
  return pd.concat(li, axis=1)

def get_merged_df(level, timit_df, tmp_dir,
                  from_scratch=False):
  filepath = os.path.join(tmp_dir, 'timit_merged_{}_info.csv'.format(level))
  if not from_scratch and os.path.isfile(filepath):
    merged_df = pd.read_csv(filepath)
    print("Successfully loaded existing merged table. Now life is easier.")
    return merged_df
  else:
    print("Couldn't load existing merged table. Now building from scatch...")
  level_label_df = get_level_label_df(level, timit_df)
  timit_df['label_file'] = timit_df.apply(lambda row:
      os.path.join(row['dirpath'], row[level + '_label']), axis=1)
  merged_df = pd.merge(level_label_df, timit_df, on='label_file')
  # list of features concerned
  labels = ['gender', 'region', 'level_label', 'speaker']
  labels_df = merged_df[labels]
  for col in labels_df:
      labels_df[col] = labels_df[col].astype('category')
  labels_num_df = cat_to_num(labels_df)
  merged_df = pd.merge(merged_df, labels_num_df,
                       left_index=True, right_index=True)
  print("Saving merged table to ", filepath)
  merged_df.to_csv(filepath, index=False)
  return merged_df

def get_label_cols(label_lvl, numeric=False):
  """Convert a number `label_lvl` to a list of column names.

  Args:
    label_lvl: integer, should be between 1 and 4 (included).
  """
  if numeric:
    label_cols = ['label' + str(x) for x in range(label_lvl)]
  else:
    all_labels = ['gender', 'region', 'level_label', 'speaker']
    label_cols = all_labels[:label_lvl]
  return label_cols

def label_to_index(pd_series):
  label_to_index_map = dict()
  pd_series = pd_series.astype('category')
  for index, label in enumerate(pd_series.cat.categories):
    label_to_index_map[label] = index
  return label_to_index_map

def get_label_to_index_map(merged_df, label_cols):
  """
  Arg:
    label_cols: a list of column names
  Returns:
    a dict mapping label to a uniform index
  """
  map_dict = {col:label_to_index(merged_df[col]) for col in label_cols}
  nums_categories = [len(map_dict[col]) for col in label_cols]
  translations = [sum(nums_categories[:i]) for i in range(len(label_cols))]
  label_to_index_map = dict()
  for idx, col in enumerate(label_cols):
    for key in map_dict[col]:
      col_key = col + '_' + key
      label_to_index_map[col_key] = map_dict[col][key] + translations[idx]
  return label_to_index_map

def wav_to_data(wav_filepath):
  sample_rate, data = wavfile.read(wav_filepath)
  return data

def sphere_to_data(sphere_filepath):
  try:
    tmp_filepath = '/tmp/haha.wav'
    os.system('sox -t sph ' + sphere_filepath + ' ' + tmp_filepath)
    res = wav_to_data(tmp_filepath)
    return res
  except:
    raise ValueError("Converting SPHERE files requires SoX installed."
                     " For Max OS, run 'brew install sox'.")

def get_interval_data(wav_filepath, begin, end):
  data = sphere_to_data(wav_filepath)
  return data[begin: end]

def get_interval_data_from_row(row):
  wav_filepath = os.path.join(row['dirpath'], row['wavfile'])
  begin = row['begin']
  end = row['end']
  return get_interval_data(wav_filepath, begin, end)

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

def time_series_to_sequence_example_df(merged_df, labels_df, filepath,
                                       max_num_examples, output_dim,
                                       is_test_set, num_shards):
  """Convert a list of time series (Numpy array) to TFRecords
  following SequenceExample proto.

  Args:
    merged_df: a pd.DataFrame object containing columns: `dirpath`,
               `wavfile`, `begin`, `end`
    labels_df: a pd.DataFrame object only containing integer labels
    filepath: a string
  Returns:
    Save a TFRecord to filepath.
    num_examples: number of examples formatted.
  """
  num_examples = merged_df.shape[0]
  num_labels = labels_df.shape[0] # number of lines of labels
  if num_examples != num_labels:
    raise ValueError("Number of examples {:d} does not match number of labels {:d}."\
                     .format(num_examples, num_labels))

  feature_label_generator = zip(merged_df.iterrows(), labels_df.iterrows())
  print("Writing to: {}... Total number of examples: {:d}".format(filepath,
                                                                  num_examples))
  first_index = None
  label_arrays = []
  max_sequence_size = 0
  avg_sequence_size = 0
  counter = 0
  with tf.python_io.TFRecordWriter(filepath) as writer:
    for (index, feature_row), (_, label_row) in feature_label_generator:
      if not first_index:
        first_index = index
      if index % 100 == 0:
        print("Writing example of index: ", index)
      le = len(label_row) # number of labels in this line
      label_array = label_row.values
      if is_test_set:
        label_arrays.append(label_sparse_to_dense(label_array,
                                                  output_dim)[None, :])
        label_index = _int64_feature([])
        label_score = _float_feature([])
      else:
        label_index = _int64_feature(label_array)
        label_score = _float_feature([1]*le)
      feature_array = get_interval_data_from_row(feature_row)
      feature_list = [_float_feature([x]) for x in feature_array]
      # Update max_sequence_size and avg_sequence_size
      len_feature_array = len(feature_array)
      if len_feature_array > max_sequence_size:
        max_sequence_size = len(feature_array)
      avg_sequence_size += (len_feature_array - avg_sequence_size)/(counter+1)

      context = tf.train.Features(
            feature={
                'id': _int64_feature([index]), # use index as id
                'label_index': label_index,
                'label_score': label_score
            })
      feature_lists = tf.train.FeatureLists(
          feature_list={
          '0_dense_input': _feature_list(feature_list)
          })
      sequence_example = tf.train.SequenceExample(
          context=context,
          feature_lists=feature_lists)
      writer.write(sequence_example.SerializeToString())
      # to have exactly `max_num_examples` examples
      counter += 1
      if max_num_examples and counter >= max_num_examples:
        break
    if is_test_set:
      all_labels = np.concatenate(label_arrays)
      solution_dir = os.path.abspath(os.path.join(os.path.dirname(filepath),
                                  os.path.pardir, os.path.pardir))
      print("solution_dir", solution_dir)
      solution_name = solution_dir.split(os.path.sep)[-1] + '.solution'
      solution_path = os.path.join(solution_dir, solution_name)
      np.savetxt(solution_path, all_labels, fmt='%.0f')
    if max_num_examples:
      num_examples = min(num_examples, max_num_examples)
    return num_examples, max_sequence_size, avg_sequence_size

def timit_to_autodl(timit_dir, level, label_lvl, max_num_examples,
                    is_test_set, num_shards, tmp_dir, output_dir,
                    sequence_size):
  """Convert TIMIT dataset to AutoDL datasets (TFRecords following
  SequenceExample proto).

  Args:
    label_lvl: a number from 1 to 4, indicating labels: gender, region,
      level_label, speaker
    The others are clear.
  """
  if is_test_set:
    data_type = 'test'
  else:
    data_type = 'train'
  new_dataset_name = data_type
  timit_df = get_timit_info_df(timit_dir, tmp_dir)
  merged_df = get_merged_df(level, timit_df, tmp_dir)
  label_cols = get_label_cols(label_lvl)
  label_to_index_map = get_label_to_index_map(merged_df, label_cols)
  output_dim = len(label_to_index_map)

  if is_test_set:
    merged_df = merged_df[merged_df['data_type'].apply(lambda x: x.lower()) == 'test']
  else:
    merged_df = merged_df[merged_df['data_type'].apply(lambda x: x.lower()) == 'train']
  label_cols = get_label_cols(label_lvl, numeric=True)
  labels_df = merged_df[label_cols]
  filename = 'sample-' + new_dataset_name
  new_dataset_dir = os.path.join(output_dir, new_dataset_name)
  if not os.path.isdir(new_dataset_dir):
    os.mkdir(new_dataset_dir)
  filepath = os.path.join(new_dataset_dir, filename)
  num_examples, max_sequence_size, avg_sequence_size =\
      time_series_to_sequence_example_df(merged_df, labels_df, filepath,
                                         max_num_examples, output_dim,
                                         is_test_set, num_shards)
  if is_test_set:
    sequence_size = int(avg_sequence_size * 1.5)

  # Write metadata
  metadata_filename = 'metadata.textproto'
  metadata_filepath = os.path.join(new_dataset_dir, metadata_filename)
  metadata = """is_sequence: false
sample_count: <sample_count>
sequence_size: <sequence_size>
output_dim: <output_dim>
matrix_spec {
  col_count: 1
  row_count: 1
  is_sequence_col: false
  is_sequence_row: false
  has_locality_col: true
  has_locality_row: true
  format: DENSE
}
"""
  metadata = metadata.replace('<sample_count>', str(num_examples))
  metadata = metadata.replace('<sequence_size>', str(sequence_size))
  metadata = metadata.replace('<output_dim>', str(output_dim))
  with open(metadata_filepath, 'w') as f:
    f.write(metadata)
  return filepath, sequence_size

def print_first_sequence_example(path_to_tfrecord):
  for bytes in tf.python_io.tf_record_iterator(path_to_tfrecord):
    sequence_example = tf.train.SequenceExample.FromString(bytes)
    print(sequence_example)
    break

if __name__ == '__main__':
  timit_dir = FLAGS.timit_dir # WARNING: you should change this to your own directory containing TIMIT dataset.
  assert(os.path.isdir(timit_dir))
  output_dir = FLAGS.output_dir
  tmp_dir = FLAGS.tmp_dir
  level = FLAGS.level
  try:
    max_num_examples_train = int(FLAGS.max_num_examples_train)
  except:
    print("Error parsing max_num_examples_train...setting to None.")
    max_num_examples_train = None
  try:
    max_num_examples_test = int(FLAGS.max_num_examples_test)
  except:
    print("Error parsing max_num_examples_test...setting to None.")
    max_num_examples_test = None
  try:
    print("Error parsing num_shards...setting to 1.")
    num_shards = int(FLAGS.num_shards)
  except:
    num_shards = 1

  label_lvl = 3 # for labels: ['gender', 'region', 'level_label']

  new_dataset_name = 'timit-{}'.format(level)
  if max_num_examples_train:
    new_dataset_name += '-{}'.format(max_num_examples_train)
  if max_num_examples_test:
    new_dataset_name += '-{}'.format(max_num_examples_test)

  new_dataset_dir = os.path.join(output_dir, new_dataset_name)
  if not os.path.isdir(new_dataset_dir):
    os.mkdir(new_dataset_dir)

  new_dataset_data_dir = os.path.join(new_dataset_dir, new_dataset_name+'.data')
  if not os.path.isdir(new_dataset_data_dir):
    os.mkdir(new_dataset_data_dir)

  # Format test set
  # is_test_set = True
  # filepath, sequence_size = timit_to_autodl(timit_dir, level, label_lvl,
  #                             max_num_examples_test,
  #                             is_test_set, num_shards, tmp_dir,
  #                             new_dataset_data_dir, None)
  sequence_size = 1934
  # Format training set
  is_test_set = False
  filepath, _ = timit_to_autodl(timit_dir, level, label_lvl,
                               max_num_examples_train,
                               is_test_set, num_shards, tmp_dir,
                               new_dataset_data_dir, sequence_size) # Using the sequence_size computed in formatting test set

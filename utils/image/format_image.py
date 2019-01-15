# Author: Adrien and Zhengying
# Date: 12 Dec 2018

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
import sys
sys.path.append('../')
from shutil import copyfile
from dataset_formatter import UniMediaDatasetFormatter


tf.flags.DEFINE_string('input_dir', '../../raw_datasets/image/',
                       "Directory containing image datasets.")

tf.flags.DEFINE_string('dataset_name', 'Caltech256', "Basename of dataset.")

tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                       "Output data directory.")

tf.flags.DEFINE_string('new_dataset_name', 'new_dataset',
                       "Basename of formatted dataset.")

FLAGS = tf.flags.FLAGS

def get_labels_df(dataset_dir):
  if not os.path.isdir(dataset_dir):
    raise IOError("{} is not a directory!".format(dataset_dir))
  labels_csv_files = [file for file in glob.glob(os.path.join(dataset_dir, '*labels*.csv'))]
  if len(labels_csv_files) > 1:
    raise ValueError("Ambiguous label file! Several of them found: {}".format(labels_csv_files))
  elif len(labels_csv_files) < 1:
    raise ValueError("No label file found! The name of this file should follow the glob pattern `*labels*.csv` (e.g. monkeys_labels_file_format.csv).")
  else:
    labels_csv_file = labels_csv_files[0]
  labels_df = pd.read_csv(labels_csv_file)
  return labels_df

def get_merged_df(labels_df, train_size=0.8):
  """Do train/test split by generating random number in [0,1]."""
  np.random.seed(42)
  merged_df = labels_df.copy()
  def get_subset(u):
    if u < train_size:
      return 'train'
    else:
      return 'test'
  merged_df['subset'] = merged_df.apply(lambda x: get_subset(np.random.rand()), axis=1)
  return merged_df

def get_features(filename):
  filepath = os.path.join(dataset_dir, filename)
  with open(filepath, 'rb') as f:
    image_bytes = f.read()
  features = [[image_bytes]]
  return features

def get_labels(label_confidence_pairs):
  """Parse label confidence pairs into two lists of labels and confidence.

  Args:
    label_confidence_pairs: string, of form `2 0.0001 9 0.48776 0 1.0`."
  """
  li_string = label_confidence_pairs.split(' ')
  if len(li_string) % 2 != 0:
    raise ValueError("In the column LabelConfidencePairs, one can only have pairs of (integer, confidence)!")
  labels = [int(x) for i, x in enumerate(li_string) if i%2 == 0]
  confidences = [float(x) for i, x in enumerate(li_string) if i%2 == 1]
  return labels, confidences

def get_features_labels_pairs(merged_df, subset='train'):
  def func(x):
    index, row = x
    filename = row['FileName']
    label_confidence_pairs = row['LabelConfidencePairs']
    features = get_features(filename)
    labels = get_labels(label_confidence_pairs)
    return features, labels
  g = merged_df[merged_df['subset'] == subset].iterrows
  features_labels_pairs = lambda:map(func, g())
  return features_labels_pairs

def show_image_from_bytes(image_bytes):
  image_tensor = tf.image.decode_image(image_bytes)
  with tf.Session() as sess:
    x = sess.run(image_tensor)
  plt.imshow(x)

def get_all_classes(merged_df):
  label_confidence_pairs = merged_df['LabelConfidencePairs']
  labels_sets = label_confidence_pairs.apply(lambda x: set(get_labels(x)[0]))
  all_classes = set()
  for labels_set in labels_sets:
    all_classes = all_classes.union(labels_set)
  return all_classes

if __name__ == '__main__':
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  output_dir = FLAGS.output_dir

  dataset_dir = os.path.join(input_dir, dataset_name)
  print(dataset_dir)
  print(os.listdir(dataset_dir)[:10]) # TODO
  labels_df = get_labels_df(dataset_dir)
  merged_df = get_merged_df(labels_df)

  all_classes = get_all_classes(merged_df)

  # TODO (@zhengying-liu @Adrien): add info on real label names
  # classes_list = [str(i) for i in range(output_dim)]

  # new_dataset_name = 'image' + str(hash(dataset_name) % 10000)
  new_dataset_name = FLAGS.new_dataset_name

  features_labels_pairs_train =\
    get_features_labels_pairs(merged_df, subset='train')
  features_labels_pairs_test =\
    get_features_labels_pairs(merged_df, subset='test')

  row_count = 350 # -1
  col_count = 350 # -1
  output_dim = len(all_classes)
  sequence_size = 1
  num_examples_train = merged_df[merged_df['subset'] == 'train'].shape[0]
  num_examples_test = merged_df[merged_df['subset'] == 'test'].shape[0]

  dataset_formatter =  UniMediaDatasetFormatter(dataset_name,
                                                output_dir,
                                                features_labels_pairs_train,
                                                features_labels_pairs_test,
                                                output_dim,
                                                col_count,
                                                row_count,
                                                sequence_size=sequence_size, # for strides=2
                                                num_examples_train=num_examples_train,
                                                num_examples_test=num_examples_test,
                                                is_sequence_col='false',
                                                is_sequence_row='false',
                                                has_locality_col='true',
                                                has_locality_row='true',
                                                format='COMPRESSED',
                                                is_sequence='false',
                                                sequence_size_func=None,
                                                new_dataset_name=new_dataset_name,
                                                classes_list=None)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

  # Copy original info file to formatted dataset
  try:
      for info_file_type in ['_public', '_private']:
          info_filepath = os.path.join(input_dir, dataset_name, dataset_name + info_file_type + '.info')
          new_info_filepath = os.path.join(output_dir, new_dataset_name, new_dataset_name + info_file_type + '.info')
          copyfile(info_filepath, new_info_filepath)
  except Exception as e:
      print('Unable to copy info files')

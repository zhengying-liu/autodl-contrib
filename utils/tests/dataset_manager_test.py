# Author: Zhengying Liu
# Date: 3 Apr 2019

import pandas as pd
import os
import sys
import tensorflow as tf

def _HERE(*args):
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))

sys.path.append(_HERE('../'))
import dataset_manager

DEFAULT_DATASET_DIR = _HERE('../../formatted_datasets/Caucase')

def test_TFRecordFormatDataset():
  tf_format_dataset = dataset_manager.TFRecordFormatDataset(DEFAULT_DATASET_DIR)
  dataset_name = tf_format_dataset.get_dataset_name()
  # assert dataset_name == 'Hammer'
  tfrecord_dataset = tf_format_dataset.get_tfrecord_dataset()
  print(type(tfrecord_dataset))
  for subset in ['train', 'test']:
    print(tf_format_dataset.get_path_to_subset(subset))

  # iterator = tfrecord_dataset.make_one_shot_iterator()
  # next_element = iterator.get_next()

  print(tf_format_dataset)
  print(tf_format_dataset.__dict__)
  print(tf_format_dataset)

  tfrecord_dataset_map = tfrecord_dataset.map(tf_format_dataset._parse_function)
  iterator = tfrecord_dataset_map.make_one_shot_iterator()
  contexts, features = iterator.get_next()
  label_index = tf.sparse.to_dense(contexts['label_index'])
  label_score = tf.sparse.to_dense(contexts['label_score'])
  idx = contexts['id']
  with tf.Session() as sess:
    example = sess.run(contexts)
    print(example)
    print(type(example))

  classes_list = tf_format_dataset.get_classes_list()
  print('classes_list:', classes_list)

  contexts, features = tf_format_dataset.get_contexts_features()
  print(contexts)
  print(features)

  image_bytes = tf_format_dataset._get_bytes()
  print(type(image_bytes))
  print(image_bytes.shape)
  is_jpeg = tf.image.is_jpeg(image_bytes)
  with tf.Session() as sess:
    example = sess.run(is_jpeg)
    print(example)
    print(type(example))

  image_format = tf_format_dataset._get_image_format()
  print(image_format)

  # test_labels = tf_format_dataset.get_test_labels()
  # print(test_labels)
  # print(test_labels.shape)
  # print(dataset_manager.to_label_confidence_pairs(test_labels))

  index = tf_format_dataset.get_index(subset='train')
  with tf.Session() as sess:
    while True:
      try:
        value = sess.run(index)
        if value % 1000 == 0:
          print(value)
      except tf.errors.OutOfRangeError:
        print(value)
        break

  print(tf_format_dataset.get_example_shape())
  # lc_pairs_train = tf_format_dataset.get_label_confidence_pairs(subset='train')
  # lc_pairs_test = tf_format_dataset.get_label_confidence_pairs(subset='test')
  # print(lc_pairs_train)
  # print(lc_pairs_test)

def test_generate_file_hash():
  csv_file_path = os.path.join(DEFAULT_DATASET_DIR + '_file_format', 'labels.csv')
  dataset_manager.generate_file_hash(csv_file_path)

def test_get_labels_df():
  tf_format_dataset = dataset_manager.TFRecordFormatDataset(DEFAULT_DATASET_DIR)
  haha = tf_format_dataset.get_labels_df()
  haha.to_csv('haha.csv')

def test_labels_df_to_dict():
  labels_df = pd.read_csv('../../formatted_datasets/Caucase_file_format/labels_hash.csv')
  hash_labels_dict = dataset_manager.labels_df_to_dict(labels_df)
  print(hash_labels_dict)

def test_tfrecord_format_to_file_format():
  tf_format_dataset = dataset_manager.TFRecordFormatDataset(DEFAULT_DATASET_DIR)
  tf_format_dataset.tfrecord_format_to_file_format()

def test_compare_datasets():
  file_dataset_dir = DEFAULT_DATASET_DIR + '_file_format'
  tfrecord_dataset_dir = DEFAULT_DATASET_DIR
  dataset_manager.compare_datasets(file_dataset_dir, tfrecord_dataset_dir)

def main(*argv):
  # test_TFRecordFormatDataset()
  # test_tfrecord_format_to_file_format()
  # test_generate_file_hash()
  # test_get_labels_df()
  # test_labels_df_to_dict()
  test_compare_datasets()

if __name__ == '__main__':
  main(sys.argv)

# Author: Adrien and Zhengying
# Date: 12 Dec 2018

import tensorflow as tf
import os
import sys
sys.path.append('../')
from shutil import copyfile
from dataset_formatter import UniMediaDatasetFormatter
from PIL import Image
# Format utils contains functions shared by format_image, format_video, etc.
from format_utils import *

def get_features(dataset_dir, filename):
  """ Read a file
  """
  filepath = os.path.join(dataset_dir, filename)
  with open(filepath, 'rb') as f:
    image_bytes = f.read()
  features = [[image_bytes]]
  return features

def get_features_labels_pairs(merged_df, dataset_dir, subset='train'):
  def func(x):
    index, row = x
    filename = row['FileName']
    if 'LabelConfidencePairs' in row:
        labels = row['LabelConfidencePairs']
        confidence_pairs = True
    elif 'Labels' in row:
        labels = row['Labels']
        confidence_pairs = False
    else:
        raise Exception('No labels found, please check labels.csv file.')
    features = get_features(dataset_dir, filename) # read file
    labels = get_labels(labels, confidence_pairs=confidence_pairs) # read labels
    return features, labels

  g = merged_df[merged_df['subset'] == subset].iterrows
  features_labels_pairs = lambda:map(func, g())
  return features_labels_pairs

def show_image_from_bytes(image_bytes):
  image_tensor = tf.image.decode_image(image_bytes)
  with tf.Session() as sess:
    x = sess.run(image_tensor)
  plt.imshow(x)


def im_size(input_dir, filenames):
    """ Find images width and length
        -1 means not fixed size
    """
    s = set()
    for filename in filenames:
        im = Image.open(os.path.join(input_dir, filename))
        s.add(im.size)
    if len(s) == 1:
        row_count, col_count = next(iter(s))
    else:
        row_count, col_count = -1, -1
    print('Images size: {} x {}\n'.format(row_count, col_count))
    return row_count, col_count


def format_data(input_dir, output_dir, new_dataset_name, train_size=0.8,
                max_num_examples=None,
                num_channels=3,
                classes_list=None, output_dim=None, quick_check=False):
  print(input_dir)
  input_dir = os.path.normpath(input_dir)
  dataset_name = os.path.basename(input_dir)
  print('Some files in input directory:')
  print(os.listdir(input_dir)[:10])
  print()
  labels_df = get_labels_df(input_dir)
  merged_df = get_merged_df(labels_df, train_size=train_size)

  #if max_num_examples and max_num_examples<=4: # if quick check, it'll be the number of examples to format for each class
  if quick_check:
    # Need at least one example of each class (tensorflow)
    #merged_df = merged_df.sample(n=max_num_examples)
    if 'LabelConfidencePairs' in list(merged_df):
        merged_df = merged_df.groupby('LabelConfidencePairs').apply(lambda x: x.sample(n=1))
    elif 'Labels' in list(merged_df):
        merged_df = merged_df.groupby('Labels').apply(lambda x: x.sample(n=1))
    else:
        raise Exception('No labels found, please check labels.csv file.')

  all_classes = get_all_classes(merged_df)

  features_labels_pairs_train =\
    get_features_labels_pairs(merged_df, input_dir, subset='train')
  features_labels_pairs_test =\
    get_features_labels_pairs(merged_df, input_dir, subset='test')
  
  if output_dim is None:
      output_dim = len(all_classes)

  sequence_size = 1
  num_examples_train = merged_df[merged_df['subset'] == 'train'].shape[0]
  num_examples_test = merged_df[merged_df['subset'] == 'test'].shape[0]
  
  filenames = labels_df['FileName']
  row_count, col_count = im_size(input_dir, filenames)

  dataset_formatter =  UniMediaDatasetFormatter(dataset_name,
                                                output_dir,
                                                features_labels_pairs_train,
                                                features_labels_pairs_test,
                                                output_dim,
                                                col_count,
                                                row_count,
                                                sequence_size=sequence_size, # for strides=2
                                                num_channels=num_channels,
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
                                                classes_list=classes_list)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

  # Copy original info file to formatted dataset
  #try:
  #    for info_file_type in ['_public', '_private']:
  #        info_filepath = os.path.join(input_dir, dataset_name, dataset_name + info_file_type + '.info')
  #        new_info_filepath = os.path.join(output_dir, new_dataset_name, new_dataset_name + info_file_type + '.info')
  #        copyfile(info_filepath, new_info_filepath)
  #except Exception as e:
  #    print('Unable to copy info files')


if __name__ == '__main__':
  tf.flags.DEFINE_string('input_dir', '../../file_format/monkeys',
                         "Directory containing image datasets.")
  tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                         "Output data directory.")
  tf.flags.DEFINE_string('new_dataset_name', 'new_dataset',
                         "Basename of formatted dataset.")

  FLAGS = tf.flags.FLAGS
  input_dir = FLAGS.input_dir
  output_dir = FLAGS.output_dir
  new_dataset_name = FLAGS.new_dataset_name

  format_data(input_dir, output_dir, new_dataset_name)

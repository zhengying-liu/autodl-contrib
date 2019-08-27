# Author: Adrien and Zhengying
# Date: 12 Dec 2018

import tensorflow as tf
import os
import sys
sys.path.append('../')
from shutil import copyfile
from dataset_formatter import UniMediaDatasetFormatter
from PIL import Image
import cv2
# Format utils contains functions shared by format_image, format_video, etc.
from format_utils import *

def image_to_bytes(image, num_channels=3, tmp_filename='TMP-a78h2.jpg'):
    image = image[:, :, :num_channels] # delete useless channels
    # we have to do this because VideoCapture read frames as 3 channels images
    cv2.imwrite(tmp_filename, image)
    with open(tmp_filename, 'rb') as f:
      frame_bytes = f.read()
    return frame_bytes

def get_features(dataset_dir, filename, num_channels=3):
    """ Read a file
    """
    features = []
    filepath = os.path.join(dataset_dir, filename)
    vid = cv2.VideoCapture(filepath)
    success, image = vid.read()
    while success:
        features.append([image_to_bytes(image, num_channels=num_channels)])
        success, image = vid.read()
    os.remove('TMP-a78h2.jpg') # to clean
    return features

def get_features_labels_pairs(merged_df, dataset_dir, subset='train', num_channels=3):
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
    features = get_features(dataset_dir, filename, num_channels=num_channels) # read file
    labels = get_labels(labels, confidence_pairs=confidence_pairs) # read labels
    return features, labels

  g = merged_df[merged_df['subset'] == subset].iterrows
  features_labels_pairs = lambda:map(func, g())
  return features_labels_pairs

def show_video_from_bytes(video_bytes):
  #video_tensor = tf.image.decode_image(video_bytes)
  #with tf.Session() as sess:
  #  x = sess.run(video_tensor)
  #plt.imshow(x)
  pass

def im_size(input_dir, filenames):
    """ Find videos width and length
        -1 means not fixed size
    """
    s = set()
    for filename in filenames:
        vid = cv2.VideoCapture(os.path.join(input_dir, filename))
        _, image = vid.read()
        s.add((image.shape[0], image.shape[1]))

    if len(s) == 1:
        row_count, col_count = next(iter(s))
    else:
        row_count, col_count = -1, -1
    print('Videos frame size: {} x {}\n'.format(row_count, col_count))
    return row_count, col_count

def seq_size(input_dir, filenames):
    """ Find videos width and length
        -1 means not fixed size
    """
    s = set()
    for filename in filenames:
        n_frames = 0
        vid = cv2.VideoCapture(os.path.join(input_dir, filename))
        success, _ = vid.read()
        while(success):
            n_frames += 1
            success, _ = vid.read()
        s.add(n_frames)

    if len(s) == 1:
        sequence_size = next(iter(s))
    else:
        sequence_size = -1
    print('Videos sequence size: {}\n'.format(sequence_size))
    return sequence_size

def format_data(input_dir, output_dir, new_dataset_name, train_size=0.8,
                max_num_examples=None,
                num_channels=3,
                classes_list=None):
  print(input_dir)
  input_dir = os.path.normpath(input_dir)
  dataset_name = os.path.basename(input_dir)
  print('Some files in input directory:')
  print(os.listdir(input_dir)[:10])
  print()
  labels_df = get_labels_df(input_dir)
  merged_df = get_merged_df(labels_df, train_size=train_size)

  if max_num_examples and max_num_examples<=4: # if quick check, it'll be the number of examples to format for each class
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
    get_features_labels_pairs(merged_df, input_dir, subset='train', num_channels=num_channels)
  features_labels_pairs_test =\
    get_features_labels_pairs(merged_df, input_dir, subset='test', num_channels=num_channels)

  output_dim = len(all_classes)
  num_examples_train = merged_df[merged_df['subset'] == 'train'].shape[0]
  num_examples_test = merged_df[merged_df['subset'] == 'test'].shape[0]

  filenames = labels_df['FileName']
  row_count, col_count = im_size(input_dir, filenames)
  sequence_size = seq_size(input_dir, filenames)

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
                         "Directory containing video datasets.")
  tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                         "Output data directory.")
  tf.flags.DEFINE_string('new_dataset_name', 'new_dataset',
                         "Basename of formatted dataset.")

  FLAGS = tf.flags.FLAGS
  input_dir = FLAGS.input_dir
  output_dir = FLAGS.output_dir
  new_dataset_name = FLAGS.new_dataset_name

  format_data(input_dir, output_dir, new_dataset_name)

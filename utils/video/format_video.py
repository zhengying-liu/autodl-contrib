# Author: Zhengying LIU
# Creation date: 1 Oct 2018
# Description: format video datasets to TFRecords (SequenceExample proto)
#   for AutoDL challenge.

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
from dataset_formatter import UniMediaDatasetFormatter

import re
from pprint import pprint
from sklearn.utils import shuffle # For shuffling datasets

# CV packages
import cv2 # Run `pip install opencv-python` to install
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tf.flags.DEFINE_string('input_dir', '../../raw_datasets/video/',
                       "Directory containing text datasets.")

tf.flags.DEFINE_string('dataset_name', 'kth', "Basename of dataset.")

tf.flags.DEFINE_string('output_dir', '../../formatted_datasets/',
                       "Output data directory.")

tf.flags.DEFINE_string('tmp_dir', "/tmp/", "Temporary directory.")

FLAGS = tf.flags.FLAGS

verbose = False

def get_kth_sequence_df(kth_dir):
  info_filename = os.path.join(kth_dir, '00sequences.txt')
  # Regular expression for lines. A typical line looks like:
  #   person13_handclapping_d4	frames	1-132, 133-234, 235-335, 336-443
  line_pattern = r'(person\d+_\w+_d\d)\s+frames\s+\d+-\d+(?:,\s+\d+-\d+)+'
  begin_end_pair_pattern = re.compile(r'\d+-\d+')

  counter = 0
  basenames = []
  begins = []
  ends = []
  with open(info_filename, 'r') as f:
    lines = f.readlines()
    relevant_lines = [x for x in lines if '\t' in x]
    for line in relevant_lines:
      res = re.match(line_pattern, line)
      if res:
        basename = res.groups()[0]
        pairs = begin_end_pair_pattern.findall(line)
        for pair in pairs:
          li_split = pair.split('-')
          begin = int(li_split[0]) - 1 # Data provider starts from 1 but I prefer from 0
          end = int(li_split[1])
          # Update lists
          basenames.append(basename)
          begins.append(begin)
          ends.append(end)
          counter += 1

  sequence_df = pd.DataFrame({'basename': basenames,
                              'begin': begins,
                              'end': ends})
  return sequence_df

def get_kth_info_df(kth_dir, tmp_dir='/tmp/', from_scratch=False):
  csv_filepath = os.path.join(tmp_dir, 'kth_files_info.csv')
  if not from_scratch and os.path.isfile(csv_filepath):
    kth_df = pd.read_csv(csv_filepath)
    print("Successfully loaded existing info table. Now life is easier.")
    return kth_df
  else:
    print("Couldn't load existing info table. Now building from scatch...")
    train_id = [11, 12, 13, 14, 15, 16, 17, 18]
    valid_id = [19, 20, 21, 23, 24, 25, 1, 4]
    test_id = [22, 2, 3, 5, 6, 7, 8, 9, 10]
    train_id = train_id + valid_id # Merge training set and validation set

    def get_subset(person_id):
      """
      Returns:
        'train' or 'test'
      """
      if person_id in train_id:
        return 'train'
      elif person_id in test_id:
        return 'test'
      else:
        raise ValueError(f"Wrong person id {person_id}!")

    path = kth_dir
    li = []
    for dirpath, dirnames, filenames in os.walk(path):
      for filename in filenames:
        if filename.endswith('.avi'):
          basename, person_id, action, remark = parse_filename(filename)
          video_filepath = os.path.join(dirpath, filename)
          subset = get_subset(person_id)
          li.append((basename, person_id, action, remark, video_filepath, subset))
    kth_df = pd.DataFrame({'basename':        [x[0] for x in li],
                           'person_id':       [x[1] for x in li],
                           'action':          [x[2] for x in li],
                           'remark':          [x[3] for x in li],
                           'video_filepath':  [x[4] for x in li],
                           'subset':          [x[5] for x in li]})
    kth_df['action'] = kth_df['action'].astype('category')
    kth_df['action_num'] = kth_df['action'].cat.codes
    kth_df.to_csv(csv_filepath, index=False)
    return kth_df

def get_merged_df(sequence_df, kth_df):
  merged_df = pd.merge(sequence_df, kth_df, on='basename', how='left')
  merged_df = shuffle(merged_df, random_state=42)
  return merged_df

def video_to_3d_features(video_filepath):
  cap = cv2.VideoCapture(video_filepath)
  features = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
      break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_flattened = gray.flatten()
    features.append(gray_flattened)
  cap.release()
  cv2.destroyAllWindows()
  features = np.array(features)
  return features

def get_features_labels_pairs(merged_df, subset='train'):
  def func(x):
    index, row = x
    video_filepath = row['video_filepath']
    features_full = video_to_3d_features(video_filepath)
    begin = row['begin']
    end = row['end']
    features = features_full[begin:end]
    labels = [row['action_num']]
    return features, labels
  g = merged_df[merged_df['subset'] == subset].iterrows()
  features_labels_pairs = map(func, g)
  return features_labels_pairs

def parse_filename(filename):
  """
  Args:
    filename: e.g. 'person01_running_d1_uncomp.avi'
  Returns:
    e.g. 'person01_running_d1', '01', 'running', 'd1'
  """
  FILENAME_PATTERN = r'(person(\d+)_(\w+)_(d\d))_uncomp.avi'
  res = re.match(FILENAME_PATTERN, filename)
  if res:
    basename, person_id, action, remark = res.groups()
    person_id = int(person_id)
    return basename, person_id, action, remark
  else:
    raise ValueError("Filename not in good pattern!")

def play_video_from_file(filename):
  fig, ax = plt.subplots()
  vidcap = cv2.VideoCapture(filename)
  success,image = vidcap.read()
  image = np.array(image).astype(float)
  screen = plt.imshow(image)
  def init():  # only required for blitting to give a clean slate.
      screen.set_data(np.empty(image.shape))
      return screen,
  def animate(i):
      if vidcap.isOpened():
        success,image = vidcap.read()
        image = np.array(image).astype(float)
        screen.set_data(image)
      return screen,
  ani = animation.FuncAnimation(
      fig, animate, init_func=init, interval=40, blit=True, save_count=50, repeat=False) # interval=40 because 25fps
  plt.show()
  return ani, plt

if __name__ == '__main__':
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  output_dir = FLAGS.output_dir
  tmp_dir = FLAGS.tmp_dir
  kth_dir = '../../raw_datasets/video/kth/'
  sequence_df = get_kth_sequence_df(kth_dir)
  kth_df = get_kth_info_df(kth_dir, tmp_dir=tmp_dir, from_scratch=False)
  merged_df = get_merged_df(sequence_df, kth_df)
  features_labels_pairs_train =\
    get_features_labels_pairs(merged_df, subset='train')
  features_labels_pairs_test =\
    get_features_labels_pairs(merged_df, subset='test')

  output_dim = 6
  row_count = 120
  col_count = 160
  dataset_formatter =  UniMediaDatasetFormatter(dataset_name,
                                                output_dir,
                                                features_labels_pairs_train,
                                                features_labels_pairs_test,
                                                output_dim,
                                                col_count,
                                                row_count,
                                                sequence_size=2000,
                                                num_examples_train=1528,
                                                num_examples_test=863,
                                                is_sequence_col='false',
                                                is_sequence_row='false',
                                                has_locality_col='true',
                                                has_locality_row='true',
                                                format='DENSE',
                                                is_sequence='false',
                                                sequence_size_func=max)

  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

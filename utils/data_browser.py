# Author: Zhengying LIU
# Date: 3 Nov 2018
"""Visualize examples and labels for given AutoDL dataset."""

import tensorflow as tf
import os, sys
import numpy as np
import cv2 # Run `pip install opencv-python` to install
import matplotlib.pyplot as plt
import matplotlib.animation as animation

STARTING_KIT_DIR = '../../autodl/codalab_competition_bundle/AutoDL_starting_kit'
INGESTION_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program')
SCORING_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_scoring_program')
CODE_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_sample_code_submission')
for dir in [INGESTION_DIR, SCORING_DIR, CODE_DIR]:
  sys.path.append(dir)
from dataset import AutoDLDataset

def read_data(dataset_dir):
  files = os.listdir(dataset_dir)
  data_files = [x for x in files if x.endswith('.data')]
  assert(len(data_files) == 1)
  dataset_name = data_files[0][:-5]
  solution_files = [x for x in files if x.endswith('.solution')]
  with_test_solution = None # With or without solution
  if len(solution_files)==1:
    solution_dataset_name = solution_files[0][:-9]
    if solution_dataset_name == dataset_name:
      with_test_solution = True
    else:
      raise ValueError("Wrong dataset name. Should be {} but got {}."\
                       .format(dataset_name, solution_dataset_name))
  elif len(solution_files)==0:
    with_test_solution = features_labels_pairs_test
  else:
    return ValueError("Multiple solution files found: {}".format(solution_files))
  D_train = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                       "train"))
  D_test = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                      "test"))
  label_to_index_map = D_train.get_metadata().get_label_to_index_map()
  classes_list = [None] * len(label_to_index_map)
  for label in label_to_index_map:
    index = label_to_index_map[label]
    classes_list[index] = label
  return D_train, D_test, classes_list

def play_video_from_features(tensor_3d, interval=80, label_confidence_pairs=None):
  fig, ax = plt.subplots()
  image = tensor_3d[0]
  screen = plt.imshow(image, cmap='gray')
  def init():  # only required for blitting to give a clean slate.
      screen.set_data(np.empty(image.shape))
      return screen,
  def animate(i):
      if i < len(tensor_3d):
        image = tensor_3d[i]
        screen.set_data(image)
      return screen,
  ani = animation.FuncAnimation(
      fig, animate, init_func=init, interval=interval, blit=True, save_count=50, repeat=False) # interval=40 because 25fps
  plt.title('Labels: ' + str(label_confidence_pairs))
  plt.show()
  return ani, plt

def infer_domain(dataset_dir):
  pass

if __name__ == '__main__':
  dataset_dir = '../formatted_datasets/kreatur'
  D_train, D_test, classes_list = read_data(dataset_dir)
  print(classes_list)
  dataset = D_train.get_dataset()
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  idx = np.random.choice(range(100))
  with tf.Session() as sess:
    for i in range(idx):
      tensor_3d, labels = sess.run(next_element)
    label_confidence_pairs = {classes_list[index]: confidence for index, confidence in enumerate(labels) if confidence != 0}
    play_video_from_features(tensor_3d, interval=80, label_confidence_pairs=label_confidence_pairs)

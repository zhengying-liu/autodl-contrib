# Author: Zhengying LIU
# Date: 3 Nov 2018
"""Visualize examples and labels for given AutoDL dataset.

Usage:
  `python data_browser.py -input_dir=../formatted_datasets/ -dataset_name=itwas`
"""

import os
import sys

import tensorflow as tf
import numpy as np
# import cv2 # Run `pip install opencv-python` to install
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tf.logging.set_verbosity(tf.logging.INFO)

STARTING_KIT_DIR = '../../autodl/codalab_competition_bundle/AutoDL_starting_kit'
INGESTION_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program')
SCORING_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_scoring_program')
CODE_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_sample_code_submission')
for d in [INGESTION_DIR, SCORING_DIR, CODE_DIR]:
  sys.path.append(d)
from dataset import AutoDLDataset # pylint: disable=wrong-import-position, import-error

tf.flags.DEFINE_string('input_dir', '../formatted_datasets/',
                       "Directory containing datasets.")

tf.flags.DEFINE_string('dataset_name', 'itwas', "Basename of dataset.")

FLAGS = tf.flags.FLAGS


class DataBrowser(object):
  """A class for visualizing datasets."""

  def __init__(self, dataset_dir):
    self.dataset_dir = dataset_dir
    self.domain = self.infer_domain()
    self.d_train, self.d_test, self.other_info = self.read_data()

  def read_data(self):
    """Given a dataset directory, read and return training/test set data as
    `AutoDLDataset` objects, along with other infomation.

    Args:
      dataset_dir: a string indicating the absolute or relative path of a
        formatted AutoDL dataset.
    Returns:
      d_train, d_test: 2 'AutoDLDataset' objects, containing training/test data.
      other_info: a dict containing some additional info on the dataset, e.g.
      the metadata on the column names and class names (contained in
        `label_to_index_map`).
    """
    dataset_dir = self.dataset_dir
    files = os.listdir(dataset_dir)
    data_files = [x for x in files if x.endswith('.data')]
    assert len(data_files) == 1
    dataset_name = data_files[0][:-5]
    solution_files = [x for x in files if x.endswith('.solution')]
    with_solution = None # With or without solution (i.e. training or test)
    if len(solution_files) == 1:
      solution_dataset_name = solution_files[0][:-9]
      if solution_dataset_name == dataset_name:
        with_solution = True
      else:
        raise ValueError("Wrong dataset name. Should be {} but got {}."\
                         .format(dataset_name, solution_dataset_name))
    elif not solution_files:
      with_solution = False
    else:
      return ValueError("Multiple solution files found:" +\
                        " {}".format(solution_files))
    d_train = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                         "train"))
    d_test = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                        "test"))
    other_info = {}
    other_info['with_solution'] = with_solution
    label_to_index_map = d_train.get_metadata().get_label_to_index_map()
    if label_to_index_map:
      classes_list = [None] * len(label_to_index_map)
      for label in label_to_index_map:
        index = label_to_index_map[label]
        classes_list[index] = label
      other_info['classes_list'] = classes_list
    else:
      tf.logging.info("No label_to_index_map found in metadata. Labels will "
                      "only be represented by integers.")
    self.d_train, self.d_test, self.other_info = d_train, d_test, other_info
    return d_train, d_test, other_info

  def infer_domain(self):
    """Infer the domain from the shape of 3."""
    d_train, _, _ = self.read_data()
    metadata = d_train.get_metadata()
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    domain = None
    if sequence_size == 1:
      if row_count == 1 or col_count == 1:
        domain = "tabular"
      else:
        domain = "image"
    else:
      if row_count == 1 and col_count == 1:
        domain = "speech"
      elif row_count == 1 or col_count == 1:
        domain = "text"
      else:
        domain = "video"
    self.domain = domain
    tf.logging.info("The inferred domain of the dataset is: {}.".format(domain))
    return domain

  @classmethod
  def show_video(cls, tensor_3d, interval=80, label_confidence_pairs=None):
    """Visualize a video represented by `tensor_3d` using `interval` ms.
    This means that frames per second (fps) is equal to 1000/`interval`.
    """
    fig, _ = plt.subplots()
    image = tensor_3d[0]
    screen = plt.imshow(image, cmap='gray')
    def init():  # only required for blitting to give a clean slate.
      """Initialize the first screen"""
      screen.set_data(np.empty(image.shape))
      return screen,
    def animate(i):
      """Some kind of hooks for `animation.FuncAnimation` I think."""
      if i < len(tensor_3d):
        image = tensor_3d[i]
        screen.set_data(image)
      return screen,
    animation.FuncAnimation(
        fig, animate, init_func=init, interval=interval,
        blit=True, save_count=50, repeat=False) # interval=40 because 25fps
    plt.title('Labels: ' + str(label_confidence_pairs))
    plt.show()
    return plt

  @classmethod
  def show_image(cls, tensor_3d, label_confidence_pairs=None):
    """Visualize a image represented by `tensor_3d` in grayscale."""
    image = tensor_3d[0]
    plt.imshow(image, cmap='gray')
    plt.title('Labels: ' + str(label_confidence_pairs))
    plt.show()
    return plt

  @classmethod
  def get_nth_element(cls, autodl_dataset, num):
    """Get n-th element in `autodl_dataset` using iterator."""
    dataset = autodl_dataset.get_dataset()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      for _ in range(num):
        tensor_3d, labels = sess.run(next_element)
    return tensor_3d, labels

  @property
  def show(self):
    """Return corresponding show method according to inferred domain."""
    domain = self.domain
    if domain == 'image':
      return DataBrowser.show_image
    elif domain == 'video':
      return DataBrowser.show_video
    else:
      raise NotImplementedError("Show method not implemented for domain: " +\
                                 "{}".format(domain))

  def show_an_example(self, max_range=1000):
    """Visualize an example whose index is randomly chosen in the interval
    [0, `max_range`).
    """
    idx = np.random.randint(0, max_range)
    tensor_3d, labels = DataBrowser.get_nth_element(self.d_train, idx)
    if 'classes_list' in self.other_info:
      c_l = self.other_info['classes_list']
      label_conf_pairs = {c_l[idx]: c for idx, c in enumerate(labels) if c != 0}
    else:
      label_conf_pairs = {idx: c for idx, c in enumerate(labels) if c != 0}
    self.show(tensor_3d, label_confidence_pairs=label_conf_pairs)

def main(*argv):
  """Do you really need a docstring?"""
  del argv
  input_dir = FLAGS.input_dir
  dataset_name = FLAGS.dataset_name
  print("Start visualizing process for dataset: {}...".format(dataset_name))
  dataset_dir = os.path.join(input_dir, dataset_name)
  data_browser = DataBrowser(dataset_dir)
  num_examples_to_visualize = input("Please enter the number of examples " +
                                    "that you want to visualize: ")
  num_examples_to_visualize = min(10, int(num_examples_to_visualize))
  for i in range(num_examples_to_visualize):
    print("Visualizing example {}.".format(i+1) +
          " Close the corresponding window to continue...")
    data_browser.show_an_example()

if __name__ == '__main__':
  main()

################################################################################
# Name:           Data Leakage Analysis Tool
# Author:         Zhengying Liu
# Creation date:  14 May 2019
# Usage:          python data_leakage_analysis.py -dataset_dir=<dataset_dir>
# Version:        v20190514
# Description:
#   This program analyzes if there is data leakage in a given dataset.
#   A typical procedure goes as follows:
#   1) Run ResNet-50 and Inception on a given dataset and recuperate
#   preprocessed data (the representation of the second-last-layer); save that
#   for future use (will be useful for several things) as new preprocessed
#   TF-records datasets.
#   2) create pages or images with pairs of examples that are among the closest
#   across {training, test} pairs in Eucliean distance of those representations.
#   For control, do the same for {training, training} pairs and {test, test}
#   pairs. Indicate their distances.
#   3) Create 3 graphs of cumulative density of number of pairs as a function of
#   distance for {training, test} pairs and {training, training} pairs and
#   {test, test} pairs.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE,
# AND THE WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL
# PROPERTY RIGHTS. IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE
# LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
# SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE
# FOR THE CHALLENGE.
################################################################################

from scipy.io import loadmat
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
# Our packages
REPO_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(REPO_DIR)
from utils.dataset_formatter import UniMediaDatasetFormatter
from utils.dataset_formatter import label_dense_to_sparse
import utils.dataset_manager as dm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODULE_URLS = {
  'inception_v3': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
  'resnet_v2_50': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
  'nasnet_large': 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1'
}

def get_module(module_name='inception_v3'):
  """Get a TensorFlow hub module according to name."""
  if module_name not in MODULE_URLS:
    raise ValueError("{} is not a valid module name. Valid module names are " +
                     "{}.".format(list(MODULE_URLS)))
  module_url = MODULE_URLS[module_name]
  module = hub.Module(module_url)
  return module

def adjust_image(tensor_4d, expected_image_size=(299, 299)):
  """Resize images to expected size.

  Args:
    tensor_4d: a 4-D Tensor of shape
      [sequence_size, row_count, col_count, num_channels]
    expected_image_size: a 2-tuple of integers
  Returns:
    imgs: a Tensor of shape [sequence_size, R, C, 3] where
      expected_image_size = (R, C)
  """
  imgs = tf.squeeze(tensor_4d, axis=[-4])
  imgs = tf.image.resize_images(imgs, expected_image_size)
  # Convert to RGB image if needed
  if imgs.shape[-1] == 1:
    imgs = tf.image.grayscale_to_rgb(imgs) # MNIST
  elif imgs.shape[-1] >= 4:
    imgs = imgs[:, :, :3]
  elif imgs.shape[-1] != 3:
    raise ValueError("Got {} channels but only support [1, 3, 4]."\
                     .format(imgs.shape[-1]))
  return imgs

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[1], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def get_prepro_func(expected_image_size=None, num_frames=None):
  """Get a preprocessing function using given image size and number of frames.
  """
  def func(example, expected_image_size, num_frames):
    if expected_image_size:
      example = adjust_image(example, expected_image_size)
    if num_frames:
      example = crop_time_axis(example, num_frames)
    return example
  prepro_func = lambda x: func(x, expected_image_size, num_frames)
  return prepro_func

def get_features_labels_pairs(dataset_dir, prepro_func=None,
                              module=None, subset='train'):
  """Given an AutoDL dataset, get an generator that generates (example, labels)
  pairs, where `example` is preprocessed using `prepro_func` and `module`.
  """
  batch_size = 100
  if module is None:
    module_name = 'inception_v3'
    module = get_module(module_name)
    expected_image_size = hub.get_expected_image_size(module)
    prepro_func = get_prepro_func(expected_image_size=expected_image_size)
  if prepro_func is None:
    prepro_func = lambda x: x
  raw_dataset = dm.TFRecordFormatDataset(dataset_dir)
  autodl_dataset = raw_dataset.get_autodl_dataset(subset=subset)
  tfrecord_dataset = autodl_dataset.get_dataset()
  preprocessed_dataset = tfrecord_dataset.map(
    lambda *x: (prepro_func(x[0]), x[1]))
  preprocessed_dataset = preprocessed_dataset.batch(batch_size)
  iterator = preprocessed_dataset.make_one_shot_iterator()
  example, labels = iterator.get_next()
  logger.info("Example shape before applying pretrained model: {}"\
              .format(example.shape))
  example = module(example)
  logger.info("Example shape after applying pretrained model: {}"\
              .format(example.shape))
  li_examples = []
  li_labels = []
  count = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
      try:
        ele = sess.run((example, labels))
        li_examples.append(ele[0])
        if subset == 'train':
          label_confidence_pairs = [label_dense_to_sparse(x) for x in ele[1]]
          li_labels += label_confidence_pairs
        count += 1
        if count % 10 == 1:
          logger.info("Preprocessed {} examples.".format(count * batch_size))
      except tf.errors.OutOfRangeError:
        break
  if subset == 'test':
    li_labels = raw_dataset.get_test_labels()
    func = lambda li: ([x[0] for x in li], [x[1] for x in li])
    li_labels = list(map(func, li_labels))
    if li_labels is None:
      raise ValueError("No solution file found. " +
                       "Please put one solution file at {}."\
                       .format(dataset_dir))
  li_examples = np.concatenate(li_examples, axis=0)
  li_examples = [[x] for x in li_examples]
  generator = lambda:zip(li_examples, li_labels)
  return generator

def format_preprocessed_dataset(dataset_dir):
  features_labels_pairs_train =\
    get_features_labels_pairs(dataset_dir, subset='train')
  features_labels_pairs_test =\
    get_features_labels_pairs(dataset_dir, subset='test')
  raw_dataset = dm.TFRecordFormatDataset(dataset_dir)
  dataset_name = raw_dataset.get_dataset_name()
  new_dataset_name = raw_dataset.get_dataset_name() + '_preprocessed'
  if dataset_dir.endswith('/'):
    dataset_dir = dataset_dir[:-1]
  output_dir = os.path.dirname(dataset_dir)
  output_dim = raw_dataset.get_output_size()
  metadata_train = raw_dataset.get_autodl_dataset(subset='train').get_metadata()
  sequence_size = 1
  row_count = 1
  col_count = None
  for x in features_labels_pairs_train():
    # x = (example, labels) where example is a list of arrays
    col_count = x[0][0].shape[-1]
    break
  num_channels = 1
  num_examples_train = raw_dataset.get_num_examples(subset='train')
  num_examples_test = raw_dataset.get_num_examples(subset='test')
  classes_list = raw_dataset.get_classes_list()

  dataset_formatter = UniMediaDatasetFormatter(
                           dataset_name,
                           output_dir,
                           features_labels_pairs_train,
                           features_labels_pairs_test,
                           output_dim,
                           col_count,
                           row_count,
                           sequence_size=sequence_size,
                           num_channels=num_channels,
                           num_examples_train=num_examples_train,
                           num_examples_test=num_examples_test,
                           is_sequence_col='false',
                           is_sequence_row='false',
                           has_locality_col='false',
                           has_locality_row='false',
                           format='DENSE',
                           is_sequence='false',
                           sequence_size_func=None,
                           new_dataset_name=new_dataset_name,
                           classes_list=classes_list,
                           is_label_array=False)
  dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

def test_get_features_labels_pairs(*argv):
  dataset_dir = '../../formatted_datasets/miniciao'
  features_labels_pairs = get_features_labels_pairs(dataset_dir)
  print(features_labels_pairs[:10])
  # return features_labels_pairs

def test_get_feature_labels_arrays(*argv):
  dataset_dir = '../../formatted_datasets/miniciao'
  feature, labels = get_feature_labels_arrays(dataset_dir)
  print("feature.shape:", feature.shape)
  print("labels.shape:", labels.shape)
  # return features_labels_pairs

def test_format_preprocessed_dataset(*argv):
  dataset_dir = '../../formatted_datasets/Saturn'
  # dataset_dir = '../../formatted_datasets/miniciao'
  features_labels_pairs = format_preprocessed_dataset(dataset_dir)

def main(*argv):
  # test_get_features_labels_pairs(*argv)
  test_format_preprocessed_dataset(*argv)
  # test_get_feature_labels_arrays()

if __name__ == '__main__':
  default_dataset_dir = '../../formatted_datasets/miniciao'

  tf.flags.DEFINE_string('dataset_dir', default_dataset_dir,
                        "Directory containing the content (e.g. adult.data/ + "
                        "adult.solution) of an AutoDL dataset. Specify this "
                        "argument if you want to test on a different dataset.")

  FLAGS = tf.flags.FLAGS
  dataset_dir = FLAGS.dataset_dir

  main(sys.argv)

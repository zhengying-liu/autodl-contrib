# Author: LIU Zhengying
# Creation Date: 18 June 2018
"""
A class to facilitate the management of datasets for AutoDL. It can keep
track of the files of each component (training set, test set, metadata, etc)
of a dataset, and make manipulations on them such as format transformation,
train/test split, example-label separation, checking, etc.

[IMPORTANT] To use this class, one should be very careful about file names! So
try not to modify file names manually.
Of course, originally this class is only reserved for organizers or their
collaborators, since the participants will only see the result of this class.
"""

import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import yaml

# Ugly section to import necessary packages in autodl
# To be replaced using Python packaging
def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))
BUNDLE_DIR = _HERE('../../autodl/codalab_competition_bundle/')
STARTING_KIT_DIR = os.path.join(BUNDLE_DIR, 'AutoDL_starting_kit')
INGESTION_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program')
sys.path.append(INGESTION_DIR)

from dataset import AutoDLDataset
from data_browser import DataBrowser

def to_label_confidence_pairs(confidences):
  """Convert a dense array of numbers in [0,1] to sparse label-confidence pairs.

  Args:
    confidences: a 'numpy.ndarray' of dtype='float' and
      shape=(num_examples, num_classes)
  Returns:
    a list of lists of (label, confidence) pairs.
  """
  return [[(l, c) for l,c in enumerate(labels_proba) if c > 0]
          for labels_proba in confidences]

class TFRecordFormatDataset(object):

  def __init__(self, dataset_dir):
    self.dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    self.dataset_name = self.get_dataset_name()
    self.domain = self.get_domain()

  def get_dataset_name(self):
    """Get the name of the dataset.

    Returns:
      a 'string' object.
    """
    files = os.listdir(self.dataset_dir)
    data_files = [x for x in files if x.endswith('.data')]
    if len(data_files) != 1:
      raise ValueError("Multiple or zero '.data' files or folders found: {}."\
                       .format(data_files))
    dataset_name = data_files[0][:-5]
    return dataset_name

  def get_path_to_subset(self, subset='train'):
    """Get the path to `subset` (can be 'train' or test).

    For example, givev a valid TFRecord Format dataset of directory 'adult/',
    then return 'adult/adult.data/train/' for the subset 'train'.
    """
    if not subset in ['train', 'test']:
      raise ValueError("`subset` should be 'train' or 'test'. But got '{}'."\
                       .format(subset))
    return os.path.join(self.dataset_dir, self.dataset_name + '.data',
                        subset)

  def get_tfrecord_dataset(self, subset='train'):
    """
    Returns:
      A raw tf.data.TFRecordDataset corresponding to `subset`.
    """
    subset_path = self.get_path_to_subset(subset)
    glob_pattern = os.path.join(subset_path, 'sample*')
    files = tf.gfile.Glob(glob_pattern)
    if not files:
        raise IOError("Unable to find training files. data_pattern='" +
                      dataset_file_pattern(self.dataset_name_) + "'.")
    return tf.data.TFRecordDataset(files)

  def get_autodl_dataset(self, subset='train'):
    subset_path = self.get_path_to_subset(subset)
    return AutoDLDataset(subset_path)

  def get_output_size(self):
    d_train = self.get_autodl_dataset(subset='train')
    metadata = d_train.get_metadata()
    output_dim = metadata.get_output_size()
    return output_dim

  def get_num_examples(self, subset='train'):
    d = self.get_autodl_dataset(subset=subset)
    return d.get_metadata().size()

  def get_example_shape(self):
    d_train = self.get_autodl_dataset(subset='train').get_dataset()
    example, labels = d_train.make_one_shot_iterator().get_next()
    return example.shape

  def get_domain(self):
    """Infer the domain.

    Returns:
      a string in ['tabular', 'image', 'speech', 'text', 'video'].
    """
    d_train = self.get_autodl_dataset(subset='train')
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
    return domain

  def get_classes_list(self):
    """Get list of text label names.

    Returns:
      a list of strings. `None` if not exists.
    """
    d_train = self.get_autodl_dataset(subset='train')
    metadata = d_train.get_metadata()
    output_dim = metadata.get_output_size()
    label_to_index_map = metadata.get_label_to_index_map()
    if not label_to_index_map:
      return None
    classes_list = [None] * output_dim
    for label, index in label_to_index_map.items():
      classes_list[index] = label
    return classes_list

  def _parse_function(self, sequence_example_proto):
    """Parse a SequenceExample in the AutoDL/TensorFlow format.

    Args:
      sequence_example_proto: a SequenceExample with "x_dense_input" or sparse
        input or compressed input representation
    Returns:
      A tuple of (contexts, features) where `contexts` is a dictionary of 3
        Tensor objects of keys 'id', 'label_index', 'label_score' and
        features a dictionary containing key '0_dense_input' for DENSE,
        '0_compressed' for COMPRESSED or '0_sparse_col_index',
        '0_sparse_row_index' and '0_sparse_value' for SPARSE.
    """
    autodl_dataset = self.get_autodl_dataset(subset='train')
    sequence_features = {}
    for i in range(autodl_dataset.metadata_.get_bundle_size()):
      if autodl_dataset.metadata_.is_sparse(i):
        sequence_features[autodl_dataset._feature_key(
            i, "sparse_col_index")] = tf.VarLenFeature(tf.int64)
        sequence_features[autodl_dataset._feature_key(
            i, "sparse_row_index")] = tf.VarLenFeature(tf.int64)
        sequence_features[autodl_dataset._feature_key(
            i, "sparse_value")] = tf.VarLenFeature(tf.float32)
      elif autodl_dataset.metadata_.is_compressed(i):
        sequence_features[autodl_dataset._feature_key(
            i, "compressed")] = tf.VarLenFeature(tf.string)
      else:
        sequence_features[autodl_dataset._feature_key(
            i, "dense_input")] = tf.FixedLenSequenceFeature(
                autodl_dataset.metadata_.get_tensor_size(i), dtype=tf.float32)
    contexts, features = tf.parse_single_sequence_example(
        sequence_example_proto,
        context_features={
            # "id": tf.VarLenFeature(tf.int64),
            "id": tf.FixedLenFeature([], tf.int64),
            "label_index": tf.VarLenFeature(tf.int64),
            "label_score": tf.VarLenFeature(tf.float32)
        },
        sequence_features=sequence_features)

    return contexts, features

  def get_contexts_features(self, subset='train'):
    """Read raw TFRecords in training set or test set by parsing
    SequenceExample proto.

    Returns:
      A tuple of (contexts, features) where `contexts` is a dictionary of 3
        Tensor objects of keys 'id', 'label_index', 'label_score' and
        features a dictionary containing key '0_dense_input' for DENSE format,
        '0_compressed' for COMPRESSED format or '0_sparse_col_index',
        '0_sparse_row_index' and '0_sparse_value' for SPARSE format.
    """
    c_name = 'contexts_' + subset
    f_name = 'features_' + subset
    if hasattr(self, c_name) and hasattr(self, f_name):
      return getattr(self, c_name), getattr(self, f_name)
    else:
      tfrecord_dataset = self.get_tfrecord_dataset(subset=subset)
      tfrecord_dataset = tfrecord_dataset.map(self._parse_function)
      iterator = tfrecord_dataset.make_one_shot_iterator()
      contexts, features = iterator.get_next()
      setattr(self, c_name, contexts)
      setattr(self, f_name, features)
      return contexts, features

  def _get_bytes(self, subset='train'):
    """Get raw bytes of the images. Only for COMPRESSED format.

    Returns:
      a 0-D tensor of bytes.
    """
    assert self.get_autodl_dataset().get_metadata().is_compressed(0)
    contexts, features = self.get_contexts_features(subset=subset)
    bytes_tensor = features['0_compressed'].values
    return bytes_tensor[0]

  def _get_image_format(self):
    """Infer image format from bytes.

    Returns:
      a string in ['jpg', 'png', 'bmp', 'gif', 'Unknown'].
    """
    image_bytes = self._get_bytes()
    is_jpeg = tf.image.is_jpeg(image_bytes)
    def is_png(contents):
      return contents[:3] == b'\211PN'
    def is_bmp(contents):
      return contents[:2] == 'BM'
    def is_gif(contents):
      return contents[:3] == b'\x47\x49\x46'
    with tf.Session() as sess:
      bytes_value = sess.run(image_bytes)
      if sess.run(is_jpeg):
        return 'jpg'
      elif is_png(bytes_value):
        return 'png'
      elif is_bmp(bytes_value):
        return 'bmp'
      elif is_gif(bytes_value):
        return 'gif'
      else:
        return 'Unknown'

  def get_index(self, subset='train'):
    """Get a 0-D tensor of the id of examples."""
    contexts, features = self.get_contexts_features(subset=subset)
    return contexts['id']

  # def get_label_confidence_pairs(self, subset='train'):
  #   """Get list of label-confidence pairs lists if exists.
  #
  #   Returns:
  #     a list of lists of label-confidence pairs or `None`.
  #   """
  #   if subset == 'train':
  #     label_confidence_pairs = []
  #     contexts, features = self.get_contexts_features(subset='train')
  #     label_index = contexts['label_index'].values
  #     label_score = contexts['label_score'].values
  #     while True:
  #       with tf.Session() as sess:
  #         try:
  #           index_value, score_value = sess.run((label_index, label_score))
  #           label_confidence_pairs.append(zip(index_value, score_value))
  #         except tf.errors.OutOfRangeError:
  #           break
  #     return label_confidence_pairs
  #   else:
  #     if not subset == 'test':
  #       raise ValueError("`subset` should be in ['train', 'test'] " +
  #                        "but got {}.".format(subset))
  #     path_to_solution = os.path.join(self.dataset_dir,
  #                                     self.dataset_name + '.solution')
  #     if not os.path.exists(path_to_solution):
  #       return None
  #     else:
  #       solution_array = np.loadtxt(path_to_solution)
  #       label_confidence_pairs = to_label_confidence_pairs(solution_array)
  #       return label_confidence_pairs


  def get_train_labels(self):
    contexts, features = self.get_contexts_features(subset='train')
    label_index = contexts['label_index'].values
    label_score = contexts['label_score'].values
    return label_index, label_score

  def get_test_labels(self):
    """Get test solution as NumPy array if exists.

    Returns:
      a list of lists of label-confidence pairs or `None`.
    """
    path_to_solution = os.path.join(self.dataset_dir,
                                    self.dataset_name + '.solution')
    if not os.path.exists(path_to_solution):
      return None
    else:
      solution_array = np.loadtxt(path_to_solution)
      label_confidence_pairs = to_label_confidence_pairs(solution_array)
      return label_confidence_pairs

  def tfrecord_format_to_file_format(self, new_dataset_name=None):
    """Generate a dataset in File Format.

    Args:
      tfdataset_dir: path to dataset in TFRecord Format.
      new_dataset_name: name of the created dataset, is also the folder name

    Returns:
      Create a folder `new_dataset_name` in the parent directory of
        `dataset_dir`.

    For more information on File Format, see:
      https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format
    """
    if self.domain != 'image':
      raise NotImplementedError("This functionality is not implemented for " +
                              "the domain {} yet.".format(browser.domain))

    dataset_name = self.dataset_name
    if new_dataset_name is None:
      new_dataset_name = dataset_name + '_file_format'

    new_dataset_dir = os.path.abspath(os.path.join(self.dataset_dir,
                          os.pardir, new_dataset_name))
    os.makedirs(new_dataset_dir, exist_ok=True)

    output_dim = self.get_output_size()

    classes_list = self.get_classes_list()
    if classes_list:
      label_name_file = os.path.join(new_dataset_dir, 'label.name')
      with open(label_name_file, 'w') as f:
        f.write('\n'.join(classes_list) + '\n')
    else:
      classes_list = range(output_dim)

    file_names = []
    label_confidence_pairs = []
    subsets = []
    indices = []

    total_num_examples = self.get_num_examples(subset='train') +\
                         self.get_num_examples(subset='test')
    le_n = len(str(total_num_examples))

    image_format = self._get_image_format()

    for subset in ['test', 'train']:
      image_bytes_tensor = self._get_bytes(subset)
      index_tensor = self.get_index(subset)

      if subset == 'test':
        count = 0
        label_confidence_pairs_test = self.get_test_labels()
        print("Number of test examples:", len(label_confidence_pairs_test))
        with tf.Session() as sess:
          while True:
            try:
              # image_bytes = sess.run(image_bytes_tensor)
              # index = sess.run(index_tensor)
              # print("index:", index)
              image_bytes, index = sess.run((image_bytes_tensor,
                                             index_tensor))
              if count % 100 == 0:
                print("Writing {}-th example for subset {}... Index: {}"\
                      .format(count, subset, index))
              index_score_list = label_confidence_pairs_test[count]

              string_list = [str(l) + ' ' + str(c) for l, c in index_score_list]
              labels_list = [str(classes_list[l]) for l, c in index_score_list]
              labels_str = '-'.join(labels_list)
              label_confidence_pairs_str = ' '.join(string_list)
              file_name = str(index).zfill(le_n) + '_' + labels_str +\
                          '_' + subset + '.' + image_format
              file_path = os.path.join(new_dataset_dir, file_name)
              with open(file_path, 'wb') as f:
                f.write(image_bytes)
              label_confidence_pairs.append(label_confidence_pairs_str)
              file_names.append(file_name)
              subsets.append(subset)
              indices.append(index)
              count += 1
            except tf.errors.OutOfRangeError:
              print("Number of last example in subset {}: {}"\
                    .format(subset, count))
              break
      else: # subset == 'train'
        count = 0
        label_index, label_score = self.get_train_labels()
        with tf.Session() as sess:
          while True:
            try:
              image_bytes, index, label_index_v, label_score_v =\
                  sess.run((image_bytes_tensor,
                            index_tensor,
                            label_index,
                            label_score))
              index_score_list = list(zip(label_index_v, label_score_v))
              if count % 100 == 0:
                print("Writing {}-th example for subset {}... Index: {}"\
                      .format(count, subset, index))
                # print("index_score_list:", index_score_list)

              string_list = [str(l) + ' ' + str(c) for l, c in index_score_list]
              labels_list = [str(classes_list[l]) for l, c in index_score_list]
              labels_str = '-'.join(labels_list)
              label_confidence_pairs_str = ' '.join(string_list)
              file_name = str(index).zfill(le_n) + '_' + labels_str +\
                          '_' + subset + '.' + image_format
              file_path = os.path.join(new_dataset_dir, file_name)
              with open(file_path, 'wb') as f:
                f.write(image_bytes)
              label_confidence_pairs.append(label_confidence_pairs_str)
              file_names.append(file_name)
              subsets.append(subset)
              indices.append(index)
              count += 1
            except tf.errors.OutOfRangeError:
              print("Number of last example in subset {}: {}"\
                    .format(subset, count))
              break

    labels_df = pd.DataFrame({'FileName': file_names,
                              'LabelConfidencePairs': label_confidence_pairs,
                              'Subset': subsets,
                              'Index': indices})
    labels_file_name = 'labels.csv'
    labels_file_path = os.path.join(new_dataset_dir, labels_file_name)
    labels_df.to_csv(labels_file_path, index=False)

    return None


class DatasetManager(object):

  # 3 possible dataset formats
  DATASET_FORMATS = ['matrix', 'file', 'tfrecord']

  def __init__(self, dataset_dir, dataset_name=None):

    # Important (and compulsory) attributes
    self._dataset_info = {} # contains almost all useful info on this dataset
    self._dataset_dir = ""  # Absolute path to dataset directory
    self._dataset_name = ""

    # Assert `dataset_dir` is valid
    if os.path.isdir(dataset_dir):
      self._dataset_dir = dataset_dir
    else:
      raise ValueError(
        "Failed to create dataset manager. {} is not a directory!"\
                       .format(dataset_dir))

    self._path_to_yaml = os.path.join(self._dataset_dir, DATASET_INFO_FILENAME)

    # Set or infer dataset name
    if dataset_name:
      self._dataset_name = dataset_name
    else:
      self._dataset_name = os.path.basename(os.path.dirname(dataset_dir))

    # Load or infer dataset info
    if os.path.exists(self._path_to_yaml):
      self.load_dataset_info()
      # If loaded void YAML, infer dataset info anyway
      if not self._dataset_info:
        self.infer_dataset_info()
      if self._dataset_name != self._dataset_info['dataset_name']:
        print("WARNING: inconsistent dataset names!")
    else:
      self.infer_dataset_info()

  def save_dataset_info(self):
    with open(self._path_to_yaml, 'w') as f:
      print("Saving dataset info to the file {}..."\
            .format(self._path_to_yaml), end="")
      yaml.dump(self._dataset_info, f)
      print("Done!")

  def load_dataset_info(self):
    """Load dataset info from the file <DATASET_INFO>"""
    assert(os.path.exists(self._path_to_yaml))
    with open(self._path_to_yaml, 'r') as f:
      print("Loading dataset info from file `{}`..."\
            .format(self._path_to_yaml), end="")
      self._dataset_info = yaml.load(f)
      print("Done!")

  def get_dataset_info(self):
    return self._dataset_info

  def generate_default_dataset_info(self):
    default_dataset_info =\
        {'dataset_name': self._dataset_name,
        'dataset_format': None, # "matrix", "file" or "tfrecord"
        'domain': None, # extension name or 'text', 'image', 'video' etc
        'metadata': None,
        'train_test_split_done': False,
        'training_data': {'examples': [],
                          'labels': [],
                          'num_examples': None,
                          'num_labels': None,
                          },
        'test_data': {'examples': [],
                      'labels': [],
                      'num_examples': None,
                      'num_labels': None,
                      },
        'integrity_check_done': False,
        'donor_info': {}
        }
    return default_dataset_info

  def infer_dataset_format(self):
    """Infer dataset format according to file names and extension names in
    dataset directory.
    """

    self._dataset_info = self.generate_default_dataset_info()

    files = os.listdir(self._dataset_dir)
    extensions = [os.path.splitext(file)[1] for file in files]

    # Matrix format
    if '.data' in extensions and '.solution' in extensions:
      self._dataset_info['dataset_format'] = DATASET_FORMATS[0]
      return
    if any(['example' in x and '.csv' in x for x in files]):
      self._dataset_info['dataset_format'] = DATASET_FORMATS[0]
      return

    # File format
    if any(['label' in x and 'file_format' in x for x in files]):
      self._dataset_info['dataset_format'] = DATASET_FORMATS[1]
      return
    for file in files:
      if 'label' in file:
        print("Possible label file found: {}".format(file))
        abspath = os.path.join(self._dataset_dir, file)
        with open(abspath, 'rb') as f:
          first_line = f.readline()
          # if "FileName" appears in the first line of the label file
          if b'filename' in first_line.lower():
            self._dataset_info['dataset_format'] = DATASET_FORMATS[1]
            return

    # TFRecord format
    if '.tfrecord' in extensions:
      self._dataset_info['dataset_format'] = DATASET_FORMATS[2]
      return
    if '.textproto' in extensions:
      self._dataset_info['dataset_format'] = DATASET_FORMATS[2]
      return

    # Else
    raise ValueError("""Oops! Cannot infer dataset format... Make sure that
      you followed file naming rules at
      https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files
    """)

  def get_dataset_format(self):
    return self._dataset_info['dataset_format']

  def infer_dataset_info(self):
    # Infer dataset format
    try:
      self.infer_dataset_format()
      dataset_format = self.get_dataset_format()
      print("Inferred dataset format: ", dataset_format)
    except:
      print("Cannot infer dataset format...")
      print("Please indicate the format of your dataset from the following:")
      print("0. Matrix Format  1. File Format  2. TFRecord Format")
      print("(Please refer to https://github.com/zhengying-liu/autodl-contrib#3-possible-formats)")
      while True:
        answer = input("Your answer (type 0, 1 or 2): ")
        if answer in ['0', '1', '2']:
          dataset_format = DATASET_FORMATS[int(answer)]
          self._dataset_info['dataset_format'] = dataset_format
          print("Indicated dataset format: ", dataset_format)
          break
        print("Invalid answer! Please try again...")

    # Infer dataset info
    print("Trying to infer dataset info under {} format..."\
          .format(dataset_format))
    if dataset_format == DATASET_FORMATS[0]: # Matrix format
      self.infer_matrix_dataset_info()
    elif dataset_format == DATASET_FORMATS[1]: # File format
      self.infer_file_dataset_info()
    elif dataset_format == DATASET_FORMATS[2]: # TFRecord format
      self.infer_tfrecord_dataset_info()
    else: # Not possible
      raise ValueError("The format {} does not exist!".format(dataset_format))

    print("Successfully inferred dataset info in {} format."\
          .format(dataset_format))
    print("Congratulations! Your dataset is valid for a contribution!")
    self._dataset_info['dataset_format'] = dataset_format
    self.collect_donor_info()
    self.integrity_check_done()
    self.save_dataset_info()
    print("Now you can find inferred dataset info in the file `{}`."\
          .format(DATASET_INFO_FILENAME))

  def collect_donor_info(self):
    donor_name = input("Please enter your name as data donor: ")
    self._dataset_info['donor_info']['name'] = donor_name

  def integrity_check_done(self):
    self._dataset_info['integrity_check_done'] = True

  def infer_matrix_dataset_info(self):
    # TODO
    raise NotImplementedError("Sorry, the integrity check of matrix format "
                              "will be implemented soon.")

  def infer_file_dataset_info(self):
    dataset_info = self.generate_default_dataset_info()
    files = os.listdir(self._dataset_dir)

    # Count the number of files in each extension and sort by number of files
    extensions = {}
    for file in files:
      ext = os.path.splitext(file)[1]
      if ext in extensions:
        extensions[ext] += 1
      else:
        extensions[ext] = 1
    extensions = sorted(list(extensions.items()), key=lambda x: -x[1])
    if len(extensions) > 1:
      print("Inferring extension from file names...")
      print("Possible extension: ", extensions[0][0])
      print("Possible number of examples: ", extensions[0][1])

    # Find the file containing labels
    def is_label_file(filename):
      return 'label' in filename and filename.endswith('.csv')
    files_label = []
    for file in files:
      if is_label_file(file):
        files_label.append(file)
    if len(files_label) < 1:
      raise ValueError("Expected 1 file containing labels. "
                       "But 0 was found.")
    elif len(files_label) > 1:
      raise ValueError("Expected 1 file containing labels. "
                       "But {} were found. They are {}"\
                       .format(len(files_label), files_label))
    file_label = files_label[0]
    labels_df = pd.read_csv(os.path.join(self._dataset_dir, file_label))

    # Compare number of rows and number of files
    n_rows = labels_df.shape[0]
    n_files = extensions[0][1]
    ext = extensions[0][0]
    if n_rows == n_files:
      dataset_info['domain'] = ext
      dataset_info['training_data']['examples'] = ['*' + ext]
      dataset_info['training_data']['labels'] = files_label
      dataset_info['training_data']['num_examples'] = n_files
      dataset_info['training_data']['num_labels'] = n_rows

      self._dataset_info = dataset_info
    else:
      print("WARNING: inconsistent number of files found! {} files with "
            "extension {} are found "
            "but {} rows of labels are found in the file "
            "{}.".format(n_files, ext, n_rows, file_label))

  def infer_tfrecord_dataset_info(self):
    dataset_info = self.generate_default_dataset_info()

    def is_sharded(path_to_tfrecord):
      return "-of-" in path_to_tfrecord

    files = os.listdir(self._dataset_dir)
    metadata_files = [x for x in files if 'metadata' in x]
    training_data_files = [x for x in files if 'train' in x]
    test_data_files = [x for x in files if 'test' in x]

    # Infer metadata
    if len(metadata_files) > 1:
      raise ValueError("More than 1 metadata files are found."
                       "Couldn't infer metadata.")
    elif len(metadata_files) < 1:
      dataset_info['metadata'] = None
    else:
      dataset_info['metadata'] = metadata_files[0]

    # Infer training data
    training_examples_files = [x for x in training_data_files if 'example' in x]
    training_labels_files = [x for x in training_data_files if 'label' in x]
    if len(training_labels_files) > 0:
      dataset_info['training_data']['labels_separated'] = True
      dataset_info['training_data']['examples'] = training_examples_files
      dataset_info['training_data']['labels'] = training_labels_files
    else:
      dataset_info['training_data']['labels_separated'] = False
      dataset_info['training_data']['examples'] = training_data_files
      dataset_info['training_data']['labels'] = training_data_files

    # Infer test data
    test_examples_files = [x for x in test_data_files if 'example' in x]
    test_labels_files = [x for x in test_data_files if 'label' in x]

    if test_labels_files: # if independent label files exist
      dataset_info['test_data']['labels_separated'] = True
      dataset_info['test_data']['examples'] = test_examples_files
      dataset_info['test_data']['labels'] = test_labels_files
    else:
      dataset_info['test_data']['labels_separated'] = False
      dataset_info['test_data']['examples'] = test_data_files
      dataset_info['test_data']['labels'] = test_data_files

    self._dataset_info = dataset_info

  def check_integrity(self):
    """Check that all file paths are valid and the  dataset is a valid dataset
    with everything and all. This parsing process could be a bit tedious.
    """
    pass

  def convert_AutoML_format_to_tfrecord(self, *arg, **kwarg):
    """Convert a dataset in AutoML format to TFRecord format.

    This facilitates the process of generating new datasets in TFRecord format,
    since there exists a big database of datasets in AutoML format.
    """
    pass

  def convert_file_format_to_tfrecord(self, *arg, **kwarg):
    """Convert a dataset in AutoML format to file format.

    This facilitates the process of generating new datasets in file format,
    since there exists a big database of datasets in AutoML format.
    """
    pass

  def train_test_split(self):
    """Split the dataset to have training data and test data
    """
    pass

  def remove_all_irrelevant_files_in_dataset_dir(self):
    pass

  def separate_labels_from_examples(self, part='test'):
    pass

def main(argv):
  if len(argv) > 1:
    dataset_dir = argv[1]
  else:
    dataset_dir = input("Please enter the path to your dataset: ")
  dm = DatasetManager(dataset_dir=dataset_dir)

if __name__ == '__main__':
  main(sys.argv)

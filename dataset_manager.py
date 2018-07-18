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

import yaml
import os
import sys
import tensorflow as tf
import pandas as pd

debug = True

# Important YAML file assigned to save and load dataset info.
# This file can be edited at first but should be automatically generated
# once processed. So be really careful when using this file.
DATASET_INFO_FILENAME = 'dataset_info.yaml'

# 3 possible dataset formats given by data donors
DATASET_FORMATS = ['matrix', 'file', 'tfrecord']


class DatasetManager(object):

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

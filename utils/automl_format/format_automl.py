# Author: Zhengying LIU
# Creation date: 20 Sep 2018
# Description: format datasets in AutoML format to TFRecords for AutoDL

verbose = False

class AutoMLMetadata():
  def __init__(self, dataset_name=None, sample_count=None, output_dim=None, set_type='train'):
    self.dataset_name = dataset_name
    self.sample_count = sample_count
    self.output_dim = output_dim
    self.set_type = set_type
  def __str__(self):
    return "AutoMLMetadata: {}".format(self.__dict__)
  def __repr__(self):
    return "AutoMLMetadata: {}".format(self.__dict__)

def is_sparse(obj):
  return scipy.sparse.issparse(obj)

def binary_to_multilabel(binary_label):
  return np.stack([1 - binary_label, binary_label], axis=1)

def regression_to_multilabel(regression_label, get_threshold=np.median):
  threshold = get_threshold(regression_label)
  binary_label = (regression_label > threshold)
  return binary_to_multilabel(binary_label)

def prepare_metadata_features_and_labels(input_dir, dataset_name, set_type='train'):
  D = DataManager(dataset_name, input_dir, replace_missing=False, verbose=verbose)
  data_format = D.info['format']
  task = D.info['task']
  if set_type == 'train':
    # Fetch features
    X_train = D.data['X_train']
    X_valid = D.data['X_valid']
    Y_train = D.data['Y_train']
    Y_valid = D.data['Y_valid']
    if is_sparse(X_train):
      concat = scipy.sparse.vstack
    else:
      concat = np.concatenate
    features = concat([X_train, X_valid])
    # Fetch labels
    labels = np.concatenate([Y_train, Y_valid])
  elif set_type == 'test':
    features = D.data['X_test']
    labels = D.data['Y_test']
  else:
    raise ValueError("Wrong set type, should be `train` or `test`!")
  # when the task if binary.classification or regression, transform it to multilabel
  if task == 'regression':
    labels = regression_to_multilabel(labels)
  elif task == 'binary.classification':
    labels = binary_to_multilabel(labels)
  # Generate metadata
  metadata = AutoMLMetadata(dataset_name=D.info['name'],
                            sample_count=features.shape[0],
                            output_dim=labels.shape[1],
                            set_type=set_type)
  return metadata, features, labels

if __name__ == '__main__':
  input_dir = '../../datasets/automl/' # Change this to the directory containing AutoML datasets

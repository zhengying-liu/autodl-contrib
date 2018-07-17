import numpy as np
import pandas as pd
import os

def generate_text(average_length=100, max_index=2000):
  length = np.random.poisson(average_length) + 1
  list_of_words = np.random.randint(max_index, size=length)
  return ' '.join(map(str, list_of_words)) + '\n'

def get_filename(index, ext='.txt', dataset_name=""):
  return dataset_name + str(index).zfill(5) + ext

def generate_distribution(length):
  assert(length > 0)
  dist = np.random.rand(length)
  dist = dist / sum(dist)
  return dist

def generate_label_confidence_pair(num_classes=103, average_num_labels=2):
  num_labels = np.random.poisson(average_num_labels) + 1
  index_list = range(num_classes)
  labels = np.random.choice(index_list, size=num_labels, replace=False)
  proba_dist = generate_distribution(num_labels)
  pairs = ["{} {:.4f}".format(x, y) for x,y in zip(labels, proba_dist)]
  return ' '.join(pairs)

def generate_label_file(filenames, num_classes=103,
                        dataset_name='minitxt'):
  filename_labels = dataset_name + '_labels_file_format.csv'
  num_examples = len(filenames)
  lc_pairs = [generate_label_confidence_pair(num_classes=num_classes)\
              for i in range(num_examples)]
  df = {"FileName": filenames, "LabelConfidencePairs": lc_pairs}
  df = pd.DataFrame(df)
  path = os.path.join(dataset_name, filename_labels)
  df.to_csv(path, index=None)

def generate_example_files(filenames, average_length=100,
                           max_index=2000, dataset_dir='minitxt/'):
  filenames = [os.path.join(dataset_dir, x) for x in filenames]
  for filename in filenames:
    with open(filename, 'w') as f:
      text = generate_text(average_length=average_length,
                           max_index=max_index)
      f.write(text)

def generate_dataset(dataset_name,
                     num_examples=30,
                     num_classes=103,
                     average_length=100,
                     num_tokens=2000,
                     average_num_labels=2):
  try:
    os.mkdir(dataset_name)
  except:
    print("WARNING: the directory `{}` already exists!".format(dataset_name))
  filenames = [get_filename(index, dataset_name=dataset_name) \
               for index in range(num_examples)]
  generate_example_files(filenames,
                         average_length=average_length,
                         max_index=num_tokens,
                         dataset_dir=dataset_name)
  generate_label_file(filenames,
                      num_classes=num_classes,
                      dataset_name=dataset_name)


if __name__ == '__main__':
  np.random.seed(seed=42)
  generate_dataset('randomtext')

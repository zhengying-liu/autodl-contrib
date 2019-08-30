# Convert AutoNLP datasets into TFRecords format
# Date: 27 Aug 2019

# Usage: `python3 nlp_to_tfrecords.py path/to/dataset`

# Input format files tree:
# ├── AutoNLP Dataset (name)
#     ├── name.solution (test solution)
#     ├── name.data
#         ├── meta.json
#         ├── train.data (each line is a string representing one example)
#         ├── test.data
#         ├── train.solution

import os
from sys import argv, path
import json
import pandas as pd
path.append('../')
from dataset_formatter import UniMediaDatasetFormatter

def get_language(filename):
    with open(filename) as json_file:
        info = json.load(json_file)
        language = info['language']
    return language

def read_file(filename):
    f = open(filename, 'r')
    output = f.read().split('\n')
    if '' in output:
        output.remove('')
    f.close()
    return output

def create_vocabulary(data, language='EN'):
    vocabulary = dict()
    i = 0
    for row in data:
        if language != 'ZH':
            row = row.split(' ')
        for token in row:
            # Split (EN or ZH)
            if token not in vocabulary:
                vocabulary[token] = i
                i += 1
    return vocabulary

def get_features(row, vocabulary, language='EN'):
    features = []
    if language != 'ZH':
        row = row.split(' ')
    for e in row:
        features.append(vocabulary[e])
    return [features]

def get_labels(row):
    labels = row.split(' ')
    return list(map(int, labels))

def get_features_labels_pairs(data, solution, vocabulary, language):
    # Function that returns a generator of pairs (features, labels)
    def func(i):
        features = get_features(data[i], vocabulary, language)
        labels = get_labels(solution[i])
        return features, labels
    g = iter(range(len(data)))
    features_labels_pairs = lambda:map(func, g)
    return features_labels_pairs

def get_output_dim(solution):
    return len(solution[0].split(' '))

if __name__=="__main__":

    if len(argv)==2:
        input_dir = argv[1]
        input_dir = os.path.normpath(input_dir)
        name = os.path.basename(input_dir)
        output_dir = input_dir + '_formatted'
    else:
        print('Please enter a dataset directory. Usage: `python3 nlp_to_tfrecords path/to/dataset`')
        exit()

    # Read data
    language = get_language(os.path.join(input_dir, name+'.data', 'meta.json'))
    train_data = read_file(os.path.join(input_dir, name+'.data', 'train.data'))
    train_solution = read_file(os.path.join(input_dir, name+'.data', 'train.solution'))
    test_data = read_file(os.path.join(input_dir, name+'.data', 'test.data'))
    test_solution = read_file(os.path.join(input_dir, name+'.solution'))

    # Create vocabulary
    vocabulary = create_vocabulary(train_data+test_data, language)

    # Convert data into sequences of integers
    features_labels_pairs_train = get_features_labels_pairs(train_data, train_solution, vocabulary, language)
    features_labels_pairs_test = get_features_labels_pairs(test_data, test_solution, vocabulary, language)

    # Write data in TFRecords and vocabulary in metadata
    output_dim = get_output_dim(train_solution)
    col_count, row_count = 1, 1
    sequence_size = -1
    num_channels = len(vocabulary)
    num_examples_train = len(train_data)
    num_examples_test = len(test_data)
    new_dataset_name = 'Pierpoljak'
    classes_list = None
    dataset_formatter =  UniMediaDatasetFormatter(name,
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
                                                  format='DENSE',
                                                  label_format='DENSE',
                                                  is_sequence='false',
                                                  sequence_size_func=None,
                                                  new_dataset_name=new_dataset_name,
                                                  classes_list=classes_list)
    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

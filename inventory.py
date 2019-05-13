# Makes the inventory and compute statistics of formatted AutoDL datasets.
# Author: Adrien Pavao
# Date: 10 May 2019

# Remarks about TFrecords format:
# - Not easy to loop over data and labels
# - Train and test labels are separated in an unintuitive way

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
INGESTION_PATH = '../autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset

DOMAINS = ['image', 'video']

class Stats():

    def __init__(self, name, subset, subset2=None):
        """  subset is an AutoDLDataset
             return a dict for statistics
             could be a class or directly in AutoDLDataset
        """
        self.name = name
        metadata = subset.metadata_
        if subset2 is None:
            self.size = metadata.size()
        else:
            self.size = metadata.size() + subset2.metadata_.size()
        self.tensor_shape = metadata.get_tensor_shape()
        self.output_size = metadata.get_output_size()
        self.num_channels = metadata.get_num_channels()
        # Initialize variables that needs the loop over data
        self.is_multilabel = False
        self.labels_sum = np.zeros(self.output_size)
        self.ones_sum = 0
        self.average_labels = None
        self.min_cardinality_label = None

    def to_string(self):
        return'{},{},{},{},{},{},{},{}\n'.format(self.name,self.size,
                                                 str(self.tensor_shape).replace(',', ';'), # let's avoid commas because of CSV format
                                                 self.output_size,self.num_channels,
                                                 self.is_multilabel, self.average_labels,
                                                 self.min_cardinality_label)

    def update(self):
        """ Update average and minimum from already computed statitics.
        """
        # class balance TODO
        # average #labels per example or distribution of #labels per example
        self.average_labels = float(self.ones_sum) / self.size
        # min cardinality of classes per label
        self.min_cardinality_label = min(self.labels_sum)

def load_dataset(input_dir, name):
    """ Load a TFRecords dataset (AutoDL format).
    """
    input_dir = os.path.join(input_dir, name)
    test_labels_file = os.path.join(input_dir, name+'.solution')
    test_labels = np.array(pd.read_csv(test_labels_file, header=None, sep=' '))
    data_dir = name + '.data'
    train = AutoDLDataset(os.path.join(input_dir, data_dir, 'train'))
    test = AutoDLDataset(os.path.join(input_dir, data_dir, 'test'))
    return name, train, test, test_labels

def compute_statistics(dataset):
    """ Compute statitics of the dataset, with and without train/test split.
     Raise a warning if pathological case,
     e.g. no ex. of one class in one label column after the split.
    """
    name, train, test, test_labels = dataset
    train_stats = Stats(name + ' train', train) # read metadata and initialize variables
    test_stats = Stats(name + ' test', test)
    all_stats = Stats(name, train, test) # stats before train/test split

    # loop over train set (labels are attached with data points)
    train_dataset = train.get_dataset()
    iterator = train_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for _ in range(train_stats.size):
         _, labels = sess.run(next_element)
         one_num = np.count_nonzero(labels)
         if (not train_stats.is_multilabel) and (one_num > 1):
             train_stats.is_multilabel = True # is multilabel
         train_stats.ones_sum += one_num
         train_stats.labels_sum += labels
    train_stats.update()

    # loop over test set (labels are in solution file)
    for labels in test_labels:
        one_num = np.count_nonzero(labels)
        if (not test_stats.is_multilabel) and (one_num > 1):
            test_stats.is_multilabel = True # is multilabel
        test_stats.ones_sum += one_num
        test_stats.labels_sum += labels
    test_stats.update()

    # re-compute for the dataset before train/test split
    all_stats.is_multilabel =  train_stats.is_multilabel or test_stats.is_multilabel
    all_stats.ones_sum = train_stats.ones_sum + test_stats.ones_sum
    all_stats.labels_sum = train_stats.labels_sum + test_stats.labels_sum
    all_stats.update()

    return all_stats, train_stats, test_stats

def get_folders(input_dir):
    folders = os.listdir(input_dir)
    # remove hidden files and files with extension (e.g. .zip)
    folders = [x for x in folders if not '.' in x]
    return folders

def write_csv(filename):
    """ Loop over formatted datasets
    """
    output = open(filename, 'w')
    output.write('name,size,tensor_shape,output_size,num_channels,is_multilabel,average_labels,min_cardinality_label\n')
    for domain in DOMAINS:
        print('\nDomain: {}\n'.format(domain))
        input_dir = '../autodl-data/{}/formatted_datasets'.format(domain)
        folders = get_folders(input_dir)
        # for each dataset
        for name in folders:
            print(name)
            # for each set (all, train, test)
            for stats in compute_statistics(load_dataset(input_dir, name)):
                row = stats.to_string()
                print(row)
                output.write(row)
    output.close()

def main():
    write_csv('inventory.csv')

if __name__ == "__main__":
    main()

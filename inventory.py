# Make the inventory and compute statistics of formatted AutoDL datasets.
# Author: Adrien Pavao
# Date: 10 May 2019

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from math import sqrt

INGESTION_PATH = '../autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset

DOMAINS = ['image'] #, 'video', 'text', 'time']
HEADER = 'name,domain,size,train_ratio,tensor_shape,output_size\n'
OUTPUT_FILE = 'inventory.csv'

def compute_statistics(dataset, domain='unknown'):
    """ Compute statitics of the dataset.
    
        :param dataset: return of load_dataset function
        :return: a string (row) of statistics
    """
    name, train, test, test_labels = dataset
    metadata = train.metadata_
    # TODO is_multilabel
    train_size, test_size = metadata.size(), test.metadata_.size()
    size = train_size + test_size
    tensor_shape = metadata.get_tensor_shape()
    tensor_shape = str(tensor_shape).replace(',', ';') # let's avoid commas because of CSV format
    return'{},{},{},{},{},{}\n'.format(name,
                                       domain,
                                       size,
                                       round(train_size/size, 3),
                                       tensor_shape,
                                       metadata.get_output_size())

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

def get_folders(input_dir):
    """ Return the list of folders in a given directory.
    """
    folders = os.listdir(input_dir)
    # remove hidden files and files with extension (e.g. .zip)
    folders = [x for x in folders if not '.' in x]
    return folders

def write_csv(filename):
    """ Loop over formatted datasets.
    """
    output = open(filename, 'w')
    output.write(HEADER)
    for domain in DOMAINS:
        print('\nDomain: {}\n'.format(domain))
        input_dir = '../autodl-data/{}/formatted_datasets'.format(domain)
        folders = get_folders(input_dir)
        # for each dataset
        for name in folders:
            print(name)
            try:
                row = compute_statistics(load_dataset(input_dir, name), domain=domain)
                print(row)
                output.write(row)
            except:
                print('FAILED.')
    output.close()
    
def print_statistics(input_dir, name):
    """ Mini version of write_csv.
    """
    row = compute_statistics(load_dataset(input_dir, name))
    print(HEADER)
    print(row)

def main():
    write_csv(OUTPUT_FILE)
    #print_statistics('../autodl-data/image/formatted_datasets', 'munster')

if __name__ == "__main__":
    main()

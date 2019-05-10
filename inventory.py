import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
INGESTION_PATH = '../autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset

DOMAINS = ['image', 'video']

def load_dataset(input_dir, name, padding=False, batch_size=1):
    """ Load a TFRecords dataset (AutoDL format).
    """
    input_dir = os.path.join(input_dir, name)
    data_dir = name + '.data'
    train = AutoDLDataset(os.path.join(input_dir, data_dir, 'train'))
    test = AutoDLDataset(os.path.join(input_dir, data_dir, 'test'))
    return train, test, name

def compute_statistics(dataset):
     """ name,tensor_shape,sample_num,label_num,class_balance
         average #labels per example or distribution of #labels per example

         min cardinality of classes per label
         - after split:
          training/test split
          fraction of examples in training and test
          recompute the same stats as before [raise a warning if pathological case, e.g. no ex. of one class in one label column
          compute the error bars of the baseline methods
     """
     train, test, name = dataset
     metadata = train.metadata_
     test_metadata = test.metadata_
     tensor_shape = metadata.get_tensor_shape()

     #data_train = train.get_dataset()
     #data_test = test.get_dataset()
     train_size = metadata.size()
     test_size = test_metadata.size()
     output_size = metadata.get_output_size()
     num_channels = metadata.get_num_channels()
     return [name, tensor_shape, output_size, train_size, test_size, train_size+test_size, num_channels]

def to_string(statistics):
    for i in range(len(statistics)):
        statistics[i] = str(statistics[i])
    return ','.join(statistics)+'\n'

def get_folders(input_dir):
    folders = os.listdir(input_dir)
    # remove hidden files and files with extension (e.g. .zip)
    folders = [x for x in folders if not '.' in x]
    return folders

def write_csv(filename):
    """ Loop over formatted datasets
    """
    output = open(filename, 'w')
    output.write('name,tensor_shape,output_size,train_size,test_size,size,num_channels\n')
    for domain in DOMAINS:
        print('\nDomain: {}\n'.format(domain))
        input_dir = '../autodl-data/{}/formatted_datasets'.format(domain)
        folders = get_folders(input_dir)
        for name in folders:
            row = to_string(compute_statistics(load_dataset(input_dir, name)))
            print(row)
            output.write(row)
    output.close()

def main():
    write_csv('inventory.csv')

if __name__ == "__main__":
    main()

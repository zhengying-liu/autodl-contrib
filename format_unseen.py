# Usage: `python3 format_unseen.py path/to/dataset output_dim path/to/ouput/folder`

from sys import argv, path
import argparse
import glob, os, yaml
import tensorflow as tf
path.append('utils')
path.append('utils/image')
path.append('utils/video')
path.append('utils/series')
path.append('utils/automl_format')
path.append('utils/text')
STARTING_KIT_DIR = '../autodl/codalab_competition_bundle/AutoDL_starting_kit'
LOG_FILE = 'baseline_log.txt'
path.append(STARTING_KIT_DIR)
path.append(os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program'))
import dataset_manager
from dataset_formatter import UniMediaDatasetFormatter
from data_manager import DataManager

import pandas as pd
import format_image
import format_video
import format_series
import format_automl_new as format_tabular
import nlp_to_tfrecords
import shutil 

import run_local_test
import data_browser
import re

verbose = False

def format_data(input_dir, output_dir, fake_name, effective_sample_num,
                train_size=0.65,
                num_channels=3,
                classes_list=None,
                domain='image', output_dim=None, input_name=None):
    """ Transform data into TFRecords
    """
    print('format_data: Formatting... {} samples'.format(effective_sample_num))
    if effective_sample_num != 0:
        if domain == 'image':
            format_image.format_data(input_dir, output_dir, fake_name,
                                     train_size=0,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list, output_dim=output_dim)
        elif domain == 'video':
            format_video.format_data(input_dir, output_dir, fake_name,
                                     train_size=0,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list, output_dim=output_dim)
        elif domain == 'series':
            format_series.format_data(input_dir, output_dir, fake_name,
                                     train_size=0,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list, output_dim=output_dim)

        elif domain == 'tabular':
            D = DataManager(input_name, input_dir, replace_missing=False, verbose=verbose)
            new_dataset_name = "unlabelled"

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            dataset_dir = os.path.join(output_dir, new_dataset_name)

            if not os.path.isdir(dataset_dir):
                os.mkdir(dataset_dir)

            # Format test set
            set_type = 'test'

            filepath = os.path.join(dataset_dir, "sample-unlabelled.tfrecord")
            metadata, features, labels = format_tabular._prepare_metadata_features_and_labels(D, set_type=set_type)
            format_tabular.convert_vectors_to_sequence_example(filepath, metadata, features, labels, D.info,
                                      max_num_examples=effective_sample_num)
        elif domain == 'text':
            name=fake_name

            language = nlp_to_tfrecords.get_language(os.path.join(input_dir, 'meta.json'))
            train_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, 'train.data'))
            train_solution = nlp_to_tfrecords.read_file(os.path.join(input_dir, 'train.solution'))
            test_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, 'test.data'))
            test_solution = nlp_to_tfrecords.read_file(os.path.join(input_dir, 'test.solution'))

            # Create vocabulary
            vocabulary = nlp_to_tfrecords.create_vocabulary(train_data+test_data, language)

            # Convert data into sequences of integers
            features_labels_pairs_train = nlp_to_tfrecords.get_features_labels_pairs(train_data, train_solution, 
                                                                    vocabulary, language, format=format)
            features_labels_pairs_test = nlp_to_tfrecords.get_features_labels_pairs(test_data, test_solution, 
                                                                   vocabulary, language, format=format)

            # Write data in TFRecords and vocabulary in metadata
            output_dim = nlp_to_tfrecords.get_output_dim(train_solution)
            col_count, row_count = 1, 1
            sequence_size = -1
            num_channels = 1 #len(vocabulary)
            num_examples_train = len(train_data)
            num_examples_test = len(test_data)
            new_dataset_name = name # same name
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
                                                          classes_list=classes_list,
                                                          channels_dict=vocabulary)
            dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

        print('format_data: done.')

if __name__=="__main__":
    if len(argv)==3 or len(argv)==4:
        input_dir = argv[1]
        input_dir = os.path.normpath(input_dir)
        output_dim = int(argv[2])

        if len(argv)==3:
            output_dir = input_dir + '_unlabelled_formatted'
        else:
        	output_dir = os.path.normpath(argv[3])
    else:
        print('Please enter a dataset directory and an output dimension. Usage: `python3 format_unseen.py path/to/dataset output_dim`')
        exit()
    
    domain = input("Domain? 'image', 'video', 'series', 'text' or 'tabular' [Default='image'] ")
    if domain == '':
        domain = 'image'
    
    num_samples=0

    if domain in ['image', 'video', 'series']:
        data_csv_files = [file for file in glob.glob(os.path.join(input_dir, '*data*.csv'))]

        if len(data_csv_files) > 1:
            raise ValueError("Ambiguous data file! Several of them found: {}".format(data_csv_files))
        elif len(data_csv_files) < 1:
            raise ValueError("No label file found! The name of this file should follow the glob pattern `*data*.csv` (e.g. monkeys_data_file_format.csv).")
        else:
            data_csv_file = data_csv_files[0]
        
        fake_labels_file = os.path.join(input_dir, 'fake_labels.csv')

        with open(fake_labels_file, 'w') as fl_file:
            fl_file.write('FileName,Labels\n')

            with open(data_csv_file, 'r') as dc_file:
                for l in dc_file:
                    num_samples += 1

                    line = l.strip()+',0\n'
                    fl_file.write(line)

        label_name = None
        label_file = os.path.join(input_dir, 'label.name')
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()

    if domain=='tabular':
        data_csv_files = [file for file in glob.glob(os.path.join(input_dir, '*data*.csv'))]

        if len(data_csv_files) > 1:
            raise ValueError("Ambiguous data file! Several of them found: {}".format(data_csv_files))
        elif len(data_csv_files) < 1:
            raise ValueError("No label file found! The name of this file should follow the glob pattern `*data*.csv` (e.g. monkeys_data_file_format.csv).")
        else:
            data_csv_file = data_csv_files[0]
        
        fake_labels_file = os.path.join(input_dir, 'unlabelled_test.solution')
        
        shutil.copyfile(data_csv_file, os.path.join(input_dir,"unlabelled_test.data"))
        
        n_features = 0
        first_line = ""

        with open(fake_labels_file, 'w') as fl_file:
            with open(data_csv_file, 'r') as dc_file:
                for l in dc_file:
                    num_samples += 1
                    
                    line = ('1 '+('0 ')*(output_dim-1)).strip()+'\n'
                    fl_file.write(line)

                    if num_samples==1:
                    	n_features = len(l.split())
                    	first_line = l

        with open(os.path.join(input_dir, 'unlabelled_train.solution'), 'w') as train_sol_file:
        	line = ('1 '+('0 ')*(output_dim-1)).strip()+'\n'
        	train_sol_file.write(line)

        with open(os.path.join(input_dir, 'unlabelled_train.data'), 'w') as train_data_file:
            line = ('0 '*(n_features)).strip()+'\n'
            train_data_file.write(line)

        open(os.path.join(input_dir, 'unlabelled_valid.data'), 'a').close()
        open(os.path.join(input_dir, 'unlabelled_valid.solution'), 'a').close()

        label_name = None
        label_file = os.path.join(input_dir, 'label.name')
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()

    if domain=='text':
        fake_labels_file = os.path.join(input_dir, 'test.solution')
        data_file = os.path.join(input_dir, 'unlabelled.data')
        shutil.copyfile(data_file, os.path.join(input_dir, 'test.data'))

        with open(fake_labels_file, 'w') as fl_file:
            with open(data_file, 'r') as dc_file:
                for l in dc_file:
                    num_samples += 1
                    
                    line = ('1 '+('0 ')*(output_dim-1)).strip()+'\n'
                    fl_file.write(line)

        label_name = None
        label_file = os.path.join(input_dir, 'label.name')
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()

    # num channels
    num_channels = input('Number of channels? [Default=3] ')
    if num_channels == '':
        num_channels = 3
    try:
        num_channels = int(num_channels)
    except Exception as e:
        print('Number of channels must be an Integer:', e)
        exit()

    # format data in TFRecords
    print('Label list:')
    if label_name is None:
        flat_label_list = None
    else:
        label_list = label_name.values.tolist()
        flat_label_list = [item for sublist in label_list for item in sublist]

    print(flat_label_list)
    effective_sample_num = num_samples
    fake_name = "unseen_formatted"

    print(effective_sample_num)

    if domain in ['image', 'video', 'series', 'text']:
        format_data(input_dir, output_dir, fake_name, effective_sample_num, num_channels=num_channels, classes_list=flat_label_list, domain=domain, output_dim=output_dim, input_name="unlabelled")

        os.remove(os.path.join(output_dir, fake_name, fake_name+".solution"))
        if os.path.exists(os.path.join(output_dir, "unlabelled")):
            shutil.rmtree(os.path.join(output_dir, "unlabelled"))

        os.mkdir(os.path.join(output_dir, "unlabelled"))

        for file in glob.glob(os.path.join(output_dir, fake_name, fake_name+".data", "test", "*")):
            shutil.move(file, os.path.join(output_dir, "unlabelled"))

        shutil.rmtree(os.path.join(output_dir, fake_name))
        os.rename(os.path.join(output_dir, "unlabelled", "sample-"+fake_name+"-test.tfrecord"), os.path.join(output_dir, "unlabelled", "sample-unlabelled.tfrecord"))
        
        if domain != 'text':
            os.remove(os.path.join(input_dir, 'fake_labels.csv'))

    elif domain == 'tabular':
        format_data(input_dir, output_dir, fake_name, effective_sample_num, num_channels=num_channels, classes_list=flat_label_list, domain=domain, output_dim=output_dim, input_name="unlabelled")
        os.remove(os.path.join(output_dir,"unlabelled","unlabelled.solution"))
        files_to_remove = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if re.search(r'(valid|train|test)', f)]

        for file in files_to_remove:
        	if os.path.exists(file):
        		os.remove(file)
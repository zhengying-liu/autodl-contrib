# Authors: Isabelle Guyon, Adrien Pavao and Zhengying Liu
# Date: Feb 6 2019

# Usage: `python3 check_n_format path/to/dataset`

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

import pandas as pd
import format_image
import format_video
import format_series
import format_automl_new as format_tabular
import nlp_to_tfrecords

import run_local_test
import data_browser

def read_metadata(input_dir):
    """ Read private.info with pyyaml
    """
    #filename = os.path.join(input_dir, 'private.info')
    filename = find_file(input_dir, 'private.info')
    print(filename)
    return yaml.load(open(filename, 'r'))


def count_labels(series):
    """ Count the number of unique labels.
        We need to do some splits for the multi-label case.
    """
    s = set()
    for line in series:
        if isinstance(line, str):
            for e in line.split(' '):
                s.add(int(e))
        else:
            s.add(line)
    s = {x for x in s if x==x} # remove NaN (example without label)
    return len(s)


def compute_stats(labels_df, label_name=None):
    """ Compute simple statistics (sample num, label num)
    """
    res = {}
    res['sample_num'] = labels_df.shape[0]
    if 'Labels' in list(labels_df):
        res['label_num'] = count_labels(labels_df['Labels'])
    elif 'LabelConfidencePairs' in list(labels_df):
        res['label_num'] = len(labels_df['LabelConfidencePairs'].unique())
    else:
        raise Exception('No labels found, please check labels.csv file.')
    if label_name is not None:
        if(label_name.shape[0] != res['label_num']):
            raise Exception('Number of labels found in label.name and computed manually is not the same: {} != {}'.format(label_name.shape[0], res['label_num']))
    res['domain'] = 'image'
    return res

def compute_stats_tabular_or_text(samples_num, label_name=None):
    """ Compute simple statistics (sample num, label num) for tabular datasets
    """
    res = {}
    res['sample_num'] = samples_num

    res['domain'] = 'tabular'
    return res

def write_info(info_file, res):
    """ Write info file from dictionary res
    """
    file = open(info_file, 'w')
    for e in res:
        file.write('{} : {}\n'.format(e, res[e]))
    file.close()


def find_file(input_dir, name):
    """ Find filename containing 'name'
    """
    filename = [file for file in glob.glob(os.path.join(input_dir, '*{}*'.format(name)))]
    return filename[0]


# This are the 3 main functions: format, baseline and check

def format_data(input_dir, output_dir, fake_name, effective_sample_num,
                train_size=0.65,
                num_channels=3,
                classes_list=None,
                domain='image'):
    """ Transform data into TFRecords
    """
    print('format_data: Formatting... {} samples'.format(effective_sample_num))
    if effective_sample_num != 0:
        if domain == 'image':
            format_image.format_data(input_dir, output_dir, fake_name,
                                     train_size=train_size,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list)
        elif domain == 'video':
            format_video.format_data(input_dir, output_dir, fake_name,
                                     train_size=train_size,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list)
        elif domain == 'series':
            format_series.format_data(input_dir, output_dir, fake_name,
                                     train_size=train_size,
                                     max_num_examples=effective_sample_num,
                                     num_channels=num_channels,
                                     classes_list=classes_list)
        elif domain == 'tabular':
            max_num_examples_train = int(effective_sample_num*train_size)
            max_num_examples_test = effective_sample_num-max_num_examples_train
            num_shards_train = 1
            num_shards_test = 1
            print(fake_name)
            format_tabular.press_a_button_and_give_me_an_AutoDL_dataset(
                            input_dir, 
                            fake_name, 
                            output_dir,
                            None,
                            None,
                            num_shards_train,
                            num_shards_test,
                            new_dataset_name=fake_name)

        elif domain == 'text':
            name=fake_name

            language = nlp_to_tfrecords.get_language(os.path.join(input_dir, name+'.data', 'meta.json'))
            train_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.data', 'train.data'))
            train_solution = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.data', 'train.solution'))
            test_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.data', 'test.data'))
            test_solution = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.solution'))

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
        else:
            raise Exception('Unknown domain: {}'.format(domain))
    print('format_data: done.')


def run_baseline(data_dir, code_dir):
    print('run_baseline: Running baseline...')
    print('Saving results in {}.'.format(LOG_FILE))
    run_local_test.run_baseline(data_dir, code_dir)
    print('run_baseline: done.')


def manual_check(browser, num_examples=5):
    print('manual_check: Checking manually...')
    print('Samples of the dataset are going to be displayed. Please check that the display is correct. Click on the cross after looking at the images.')
    browser.show_examples(num_examples=num_examples)
    print('manual_check: done.')
    # TODO: ask for check

def is_formatted(output_dir):
    """ Check if data are already formatted """
    return os.path.exists(output_dir)


if __name__=="__main__":

    if len(argv)==2:
        input_dir = argv[1]
        input_dir = os.path.normpath(input_dir)
        output_dir = input_dir + '_formatted'
    else:
        print('Please enter a dataset directory. Usage: `python3 check_n_format path/to/dataset`')
        exit()

    # Read the meta-data in private.info.
    print(input_dir)
    metadata = read_metadata(input_dir)
    fake_name = metadata['name']
    print('\nDataset fake name: {}\n'.format(fake_name))

    # domain (image, video, text, etc.)
    domain = input("Domain? 'image', 'video', 'series', 'text' or 'tabular' [Default='image'] ")
    if domain == '':
        domain = 'image'

    effective_sample_num=0

    if domain in ['image', 'video', 'series']:
        print("Domain: {}. File format required.".format(domain))

        labels_df = format_image.get_labels_df(input_dir) # same function in format_video
        print('First rows of labels file:')
        print(labels_df.head())
        print()

        label_name = None
        label_file = os.path.join(input_dir, 'label.name')
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()
        # Compute simple statistics about the data (file number, etc.) and check consistency with the CSV file containing the labels.

        print('Some statistics:')
        res = compute_stats(labels_df, label_name=label_name)
        print(res)
        print()

        # Ask user what he wants to be done
        effective_sample_num = res['sample_num'] # if quick check, it'll be the number of examples to format for each class

    elif domain == 'tabular':
        print("Domain: tabular. AutoML format required.")
        label_name = None
        label_file = [file for file in glob.glob(os.path.join(input_dir, '*label.name'))][0]
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()

        labels_train_file = [file for file in glob.glob(os.path.join(input_dir, '*train.solution'))][0]
        labels_test_file = [file for file in glob.glob(os.path.join(input_dir, '*test.solution'))][0]
        s=0
        with open(labels_train_file, 'r') as f:
            for line in f:
                s+=1

        with open(labels_test_file, 'r') as f:
            for line in f:
                s+=1
    
        effective_sample_num = s
        res = compute_stats_tabular_or_text(s, label_name=label_name)
        print(res)
        print()

    elif domain == 'text':
        print("Domain: text. AutoNLP format required.")
        name = fake_name

        train_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.data', 'train.data'))
        test_data = nlp_to_tfrecords.read_file(os.path.join(input_dir, name+'.data', 'test.data'))
        s=len(train_data)+len(test_data)

        label_name = None
        label_file = [file for file in glob.glob(os.path.join(input_dir, '*label.name'))][0]
        if os.path.exists(label_file):
            print('First rows of label names:')

            label_name = pd.read_csv(label_file, header=None)
            print(label_name.head())
            print()

        effective_sample_num=s
        res = compute_stats_tabular_or_text(s, label_name=label_name)
        print(res)
        print()
    else:
        raise Exception('Unknown domain: {}'.format(domain))

    quick_check = 1 # just for display purpose
    if not input('Quick check? [Y/n] (creating a mini-dataset, only available for image, video and speech/time series) ') in ['n', 'N']:
        # quick check
        print('Quick check enabled: running script on a small subset of data to check if everything works as it should.')
        output_dir = output_dir + '_mini'
        effective_sample_num = min(effective_sample_num, 1)
        #quick_check = res['label_num'] # just for display purpose

    if is_formatted(output_dir):
        # Already exists
        if not input('Overwrite existing formatted data? [Y/n] ') in ['n', 'N']:
            # Overwrite
            #if input('Re-format all {} files? [Y/n] '.format(effective_sample_num * quick_check)) in ['n', 'N']: # Confirmation
            if input('Re-format all files? [Y/n] ') in ['n', 'N']: # Confirmation
                # Do nothing
                exit()
        else:
            effective_sample_num = 0

    # Init output_dir
    else:
        print('No formatted version found, creating {} folder.'.format(output_dir))
        os.mkdir(output_dir)

    """# domain (image, video, text, etc.)
    domain = input("Domain? 'image', 'video', 'series' or 'tabular' [Default='image'] ")
    if domain == '':
        domain = 'image'
    """
    # num channels
    num_channels = input('Number of channels? [Default=3] ')
    if num_channels == '':
        num_channels = 3
    try:
        num_channels = int(num_channels)
    except Exception as e:
        print('Number of channels must be an Integer:', e)
        exit()

    # booleans
    do_run_baseline = not input('Run baseline on formatted data? [Y/n] ') in ['n', 'N']
    do_manual_check = not input('Do manual check? [Y/n] ') in ['n', 'N']

    # format data in TFRecords
    print('Label list:')
    if label_name is None:
        flat_label_list = None
    else:
        label_list = label_name.values.tolist()
        flat_label_list = [item for sublist in label_list for item in sublist]
    print(flat_label_list)
    format_data(input_dir, output_dir, fake_name, effective_sample_num, num_channels=num_channels, classes_list=flat_label_list, domain=domain)
    formatted_dataset_path = os.path.join(output_dir, fake_name)

    # run baseline
    if do_run_baseline:
        code_dir = os.path.join(STARTING_KIT_DIR, 'AutoDL_sample_code_submission')
        run_baseline(formatted_dataset_path, code_dir)
        # TODO: save results in log file

    # manual check
    browser = data_browser.DataBrowser(formatted_dataset_path)
    if do_manual_check:
        manual_check(browser, num_examples=10)

    # Write metadata
    res['tensor_shape'] = browser.get_tensor_shape()
    public_info_file = os.path.join(output_dir, fake_name, 'public.info')
    write_info(public_info_file, res)
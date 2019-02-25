# Authors: Isabelle Guyon, Adrien Pavao and Zhengying Liu
# Date: Feb 6 2019

# Usage: `python3 check_n_format path/to/dataset`

from sys import argv, path
import glob, os, yaml
import tensorflow as tf
path.append('utils')
path.append('utils/image')
STARTING_KIT_DIR = '../autodl/codalab_competition_bundle/AutoDL_starting_kit'
LOG_FILE = 'baseline_log.txt'
path.append(STARTING_KIT_DIR)
path.append(os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program'))
import dataset_manager
import pandas as pd
import format_image
import run_local_test
import data_browser


def read_metadata(input_dir):
    """ Read private.info with pyyaml
    """
    #filename = os.path.join(input_dir, 'private.info')
    filename = find_file(input_dir, 'private.info')
    return yaml.load(open(filename, 'r'))


def compute_stats(labels_df, label_name=None):
    """ Compute simple statistics (sample num, label num)
    """
    res = {}
    res['sample_num'] = labels_df.shape[0]
    if 'Labels' in list(labels_df):
        res['label_num'] = len(labels_df['Labels'].unique())
    elif 'LabelConfidencePairs' in list(labels_df):
        res['label_num'] = len(labels_df['LabelConfidencePairs'].unique())
    else:
        raise Exception('No labels found, please check labels.csv file.')
    if label_name is not None:
        assert(label_name.shape[0] == res['label_num'])
    res['domain'] = 'image'
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

def format_data(input_dir, output_dir, fake_name, effective_sample_num, train_size=0.8):
    """ Transform data into TFRecords
    """
    print('format_data: Formatting... {} samples'.format(effective_sample_num))
    if effective_sample_num != 0:
        format_image.format_data(input_dir, output_dir, fake_name, train_size=train_size, max_num_examples=effective_sample_num)
    print('format_data: done.')


def run_baseline(data_dir, code_dir):
    print('run_baseline: Running baseline...')
    print('Saving results in {}.'.format(LOG_FILE))
    run_local_test.run_baseline(data_dir, code_dir)
    print('run_baseline: done.')


def manual_check(data_dir):
    print('manual_check: Checking manually...')
    print('Samples of the dataset are going to be displayed. Please check that the display is correct. Click on the cross after looking at the images.')
    data_browser.show_examples(data_dir)
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
    metadata = read_metadata(input_dir)

    fake_name = metadata['name']
    print('\nDataset fake name: {}\n'.format(fake_name))
    labels_df = format_image.get_labels_df(input_dir)

    print('First rows of labels file:')
    print(labels_df.head())
    print()

    label_name = None
    label_file = os.path.join(input_dir, 'label.name')
    if os.path.exists(label_file):
        label_name = pd.read_csv(label_file, header=None)
        print('First rows of label names:')
        print(label_name.head())
        print()

    # Compute simple statistics about the data (file number, etc.) and check consistency with the CSV file containing the labels.
    res = compute_stats(labels_df, label_name=label_name)
    print('Some statistics:')
    print(res)
    print()

    # Ask user what he wants to be done
    effective_sample_num = res['sample_num'] # if quick check, it'll be the number of examples to format for each class
    if is_formatted(output_dir):
        # already exists
        if not input('Overwrite existing formatted data? [Y/n] ') in ['n', 'N']:
            # Overwrite
            if not input('Quick check? [Y/n] ') in ['n', 'N']:
                # quick check
                print('Quick check enabled: running script on a small subset of data to check if everything works as it should.')
                output_dir = output_dir + '_mini'
                effective_sample_num = min(effective_sample_num, 1)

            elif input('Re-format all {} files? [Y/n] '.format(effective_sample_num)) in ['n', 'N']:
                # quick check
                effective_sample_num = min(effective_sample_num, 1)
        else:
            effective_sample_num = 0

    # Init output_dir
    else:
        if not input('Quick check? [Y/n] ') in ['n', 'N']:
            # quick check
            print('Quick check enabled: running script on a small subset of data to check if everything works as it should.')
            output_dir = output_dir + '_mini'
            effective_sample_num = min(effective_sample_num, 1)

        print('No formatted version found, creating {} folder.'.format(output_dir))
        os.mkdir(output_dir)


    # Write metadata
    public_info_file = os.path.join(output_dir, 'public.info')
    write_info(public_info_file, res)

    # booleans
    do_run_baseline = not input('Run baseline on formatted data? [Y/n] ') in ['n', 'N']
    do_manual_check = not input('Do manual check? [Y/n] ') in ['n', 'N']

    # format data in TFRecords
    format_data(input_dir, output_dir, fake_name, effective_sample_num)
    formatted_dataset_path = os.path.join(output_dir, fake_name)

    # run baseline
    if do_run_baseline:
        code_dir = os.path.join(STARTING_KIT_DIR, 'AutoDL_sample_code_submission')
        run_baseline(formatted_dataset_path, code_dir)
        # TODO: save results in log file

    # manual check
    if do_manual_check:
        manual_check(formatted_dataset_path)

# Authors: Isabelle Guyon, Adrien Pavao and Zhengying Liu
# Date: Feb 6 2019

from sys import argv, path
import os
import yaml
path.append('utils')
path.append('utils/image')
import dataset_manager
import pandas as pd
import format_image

def read_metadata(data_dir):
    filename = os.path.join(data_dir, 'private.info')
    return yaml.load(open(filename, 'r'))


def compute_stats(labels_df, label_name):
    res = {}
    res['sample_num'] = labels_df.shape[0]
    res['label_num'] = label_name.shape[0]
    assert(len(labels_df['Labels'].unique()) == res['label_num'])
    return res


def write_info(info_file, res):
    file = open(info_file, 'w')
    for e in res:
        file.write('{} : {}\n'.format(e, res[e]))
    file.close()


def format_data(effective_sample_num):
    print('Formatting... {} samples'.format(effective_sample_num))
    pass


def run_baseline():
    print('Running baseline...')
    pass


def manual_check():
    print('Checking manually...')
    pass


def is_formatted(output_dir):
    """ Check if data are already formatted """
    return os.path.exists(output_dir)

if __name__=="__main__":

    if len(argv)==2:
        data_dir = argv[1]
        output_dir = data_dir + '_formatted' # TODO clean up the path
    else:
        print('Please enter a dataset directory')
        exit()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read the meta-data in private.info.
    metadata = read_metadata(data_dir)
    print(metadata['name'])
    labels_df = format_image.get_labels_df(data_dir)
    print(labels_df.head())

    label_file = os.path.join(data_dir, 'label.name')
    if os.path.exists(label_file):
        label_name = pd.read_csv(label_file, header=None)
    print(label_name.head())

    # Compute simple statistics about the data (file number, etc.) and check consistency with the CSV file containing the labels.
    res = compute_stats(labels_df, label_name)
    print(res)

    public_info_file = os.path.join(output_dir, 'public.info')
    write_info(public_info_file, res)


    # Ask user

    effective_sample_num = res['sample_num']

    if is_formatted(output_dir):
        # already exists
        if not input('Overwrite existing formatted data? [Y/N] ') in ['n', 'N']:
            # Overwrite
            if not input('Quick check? [Y/N] ') in ['n', 'N']:
                # quick check
                effective_sample_num = min(effective_sample_num, 10)

            elif input('Re-format all {} files? [Y/N] '.format(effective_sample_num)) in ['n', 'N']:
                # quick check
                effective_sample_num = min(effective_sample_num, 10)

        else:
            effective_sample_num = 0

    # booleans
    do_run_baseline = not input('Run baseline on formatted data? [Y/N] ') in ['n', 'N']
    do_manual_check = not input('Do manual check? [Y/N] ') in ['n', 'N']

    # format data in TFRecords
    format_data(effective_sample_num)

    # run baseline
    if do_run_baseline:
        run_baseline()
        # TODO: save results in log file

    # manual check
    if do_manual_check:
        manual_check()

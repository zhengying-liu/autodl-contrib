#!/usr/bin/python
# -*- coding: utf-8 -*-

# DATASET INVENTORY SCRIPT
# Author: Adrien

# This script reads info files of every dataset and write a CSV table
# summarizing the data for AutoDL Challenge 2018

# WARNING : This script requires to be placed in a folder with a specific file tree!

### Table columns ###
# Name fake/real : title / name
# Contact name : contact_name
# Domain : domain
# Train num : train_num
# Output DIM : label_num
# Zip size : TODO

# Dim TFRecords si formaté : TODO
# Link to google cloud : resource_url TODO
# Remarks : remarks

# Status (No data, raw, formatted) /formatted, unformatted, promised/
### ### ### ### ###

import os, os.path
import re

# The output table
output_file = 'datasets_summary.csv'
domains = ['text', 'time', 'image', 'video']

# Let's separate tabular data to make it clearer
tabular_output_file = 'tabular_datasets_summary.csv'
tabular_domains = ['tabular']

folders = ['raw_datasets', 'formatted_datasets', 'promised_datasets']

def read_info_file(public_info_path, private_info_path):
    """ Return dictionnary with info file information """

    dic = dict()

    # read info files
    file1 = open(public_info_path, 'r')
    file2 = open(private_info_path, 'r')
    tab = file1.read().split('\n')
    tab = file2.read().split('\n') + tab
    tab = [x for x in tab if x != ''] # remove ''
    file1.close()
    file2.close()

    for row in tab:
        # checking for each row because there is 2 files
        if '=' in row: # info format
            regexp = '(\s)+=(\s)+'
        else: # yaml format
            regexp = '(\s)+:(\s)+'

        t = re.compile(regexp).split(row) # split ' = '
        for i in range(len(t) - 2): # remove ' '
            while ' ' in t:
                t.remove(' ')

        dic[t[0]] = t[1]

    return dic

def find_info_files(dataset_dir):
    """ Find path to public and private info files from dataset directory path """
    dataset_files = os.listdir(dataset_dir)

    try:
        public_info_file = [x for x in dataset_files if ('_public.info' in x)][0]
        private_info_file = [x for x in dataset_files if ('_private.info' in x)][0]

        public_info_path = dataset_dir + '/' + public_info_file
        private_info_path = dataset_dir + '/' + private_info_file

        return public_info_path, private_info_path

    except Exception as e:
        print('No info files in {}'.format(dataset_dir))
        return None, None


def init_info_file(dataset_dir, domain, public=True, templates_dir='info_file_templates/'):
    """ Initiate .info file according to a template """

    dataset_name = dataset_dir.split('/')[-1]

    if public:
        filename = dataset_dir + '/' + dataset_name + '_public.info'
        template = open(templates_dir + 'template_public.info', 'r')
        content = template.read().format(domain)
        template.close()

    else:
        filename = dataset_dir + '/' + dataset_name + '_private.info'
        template = open(templates_dir + 'template_private.info', 'r')
        content = template.read().format(dataset_name.capitalize())
        template.close()

    print('Creating {}'.format(filename))
    f = open(filename, 'w')
    f.write(content)
    f.close()

    return filename


def init_csv(csv_path):
    """ Initiate CSV file with column names """
    print('Creating {}'.format(csv_path))
    f = open(csv_path, 'w')
    f.write('Name,Fake name,Status,Contact name,Purpose,Domain,Train num,Output DIM,Zip size,TFDim,Link,Remarks\n')
    f.close()


def add_entry_csv(csv_path, dic, status=None):
    """ Add entry to a CSV file by reading information in dictionary """
    f = open(csv_path, 'a')

    row = ''
    infos = ['title', 'name', 'status', 'contact_name', 'purpose', 'domain', 'train_num', 'label_num', 'zip_size', 'TFDim', 'resource_url', 'remarks']

    for i, info in enumerate(infos):
        if info in dic:
            row = row + dic[info].replace("'", '').replace(',', ';')

        elif info == 'status' and status is not None:
            # Status: Raw dataset, Formatted dataset, Promised dataset
            row = row + status.capitalize().replace('_', ' ')

        else:
            row = row + 'Unknown'

        if i < (len(infos) -1):
            row = row + ','

    f.write(row+'\n')
    f.close()


def write_information_table(output_file, domains, folders):
    """ Create information table in CSV format
        This is the main function of the script
    """
    init_csv(output_file)

    for domain in domains:
        for folder in folders:
            DIR = domain + '/' + folder

            # Read dataset names
            if os.path.exists(DIR):
                dataset_names = os.listdir(DIR) # read the input folder
                dataset_names = [x for x in dataset_names if not (x.startswith('.') or x.startswith('__') or any(s in x for s in ('zip', 'tar', 'tgz')))] # remove hidden files and archives

                for dataset in dataset_names:

                    dataset_dir = DIR + '/' + dataset
                    public_info_path, private_info_path = find_info_files(dataset_dir)

                    # create info files if needed
                    if public_info_path is None:
                        public_info_path = init_info_file(dataset_dir, domain)
                    if private_info_path is None:
                        private_info_path = init_info_file(dataset_dir, domain, public=False)

                    # parse information
                    dic = read_info_file(public_info_path, private_info_path)

                    # Write information
                    add_entry_csv(output_file, dic, status=folder)


write_information_table(output_file, domains, folders)
write_information_table(tabular_output_file, tabular_domains, folders)



### OLD SCRIPT ###

# Display directory tree
#os.system('tree -d')
#print('\n')

# Count datasets
#for domain in domains:
#    print(domain.capitalize())
#    for folder in folders:
#        DIR = domain+'/'+folder
#        count = len([name for name in os.listdir(DIR)])
#        print('{} : {}'.format(folder.capitalize().replace('_', ' '), count))
#    print('\n')

### ### ### ### ###

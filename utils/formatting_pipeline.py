# Author: Adrien PAVAO
# Creation date: 2 Oct 2018
# Description: Script for formatting many datasets from AutoML to AutoDL format

### Imports ###
import os
import sys
import tensorflow as tf
import re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

sys.path.append('automl_format')
sys.path.append('automl_format/ingestion_program')
sys.path.append('dataset_test') 
sys.path.append('../tfrecord_format/autodl_format_definition')

# Clear flags to import several scripts with the same flags definition
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

from format_automl import press_a_button_and_give_me_an_AutoDL_dataset
del_all_flags(tf.flags.FLAGS) # clear flags
from inspect_dataset import check_integrity
#from test_with_baseline

### Parameters  ###
input_dir = '../raw_datasets/automl/'
output_dir = '../formatted_datasets/'
max_num_examples_train = None
max_num_examples_test = None
num_shards_train = 1
num_shards_test = 1
# Read dataset names
dataset_names = os.listdir(input_dir) # read the input folder
dataset_names = [x for x in dataset_names if not (x.startswith('.') or x.startswith('__'))] # remove hidden files
# Or specify values
dataset_names = ['HTRU2']

def parse_info(tab, info):
    """ Return value associated to an information in info file 
        e.g. 'task =  multilabel.classification'
        return 'multilabel'
    """
    for row in tab:
        t = re.compile('(\s)+=(\s)+').split(row) # split ' =  '
        for i in range(len(t) - 2): # remove ' '
            while ' ' in t:
                t.remove(' ')

        if t[0] == info:
            return t[1]
    return 'Undefined'

def init_tabular(filename='tabular.tex'):
    """ Initialize LaTeX tabular file """
    f = open(filename, 'w')
    f.write('\\begin{tabular}{|l |l |r |l |l |l |c |r |r |r |c |}\n')
    f.close()
    f = open(filename, 'a')
    f.write('\\hline\n')
    f.write('DATASET & Task & C & Sparse & Miss & Cat & Pte & Ptr & N & Ptr/N & Baseline Score \\\\\n') # Cbal ?
    f.write('\\hline\n')
    f.close()
    
def close_tabular(filename='tabular.tex'):
    """ Conclude LaTeX tabular file """ 
    f = open(filename, 'a')
    f.write('\\hline\n')
    f.write('\\end{tabular}')
    f.close()

def add_entry_tabular(input_dir, dataset_name, score, filename='tabular.tex'):
    """ Add dataset information in LaTeX tabular file """
    
    public_info_file = open(os.path.join(input_dir, dataset_name, dataset_name+'_public.info'), 'r')
    public_info = public_info_file.read().split('\n')
    private_info_file = open(os.path.join(input_dir, dataset_name, dataset_name+'_private.info'), 'r')
    private_info = private_info_file.read().split('\n')
    
    f = open(filename, 'a')
    
    f.write(dataset_name + ' & ')
    for info in ['task', 'target_num', 'is_sparse', 'has_missing', 'has_categorical', 'test_num', 'train_num']:
        value = parse_info(public_info + private_info, info) 
        if info == 'task':
            value = value.replace("'", "").split('.')[0]
        f.write(value + ' & ')
        if info == 'test_num':
            Pte = int(value)
        if info == 'train_num':
            Ptr = int(value)        
      
    N = Pte + Ptr
    f.write(str(N) + ' & ')
    f.write(str(round(float(Ptr / N), 3)) + ' & ')
    f.write(str(score) + ' \\\\\n')
    f.close()
    
def init_doc(filename='doc.tex'):
    """ Initialize LaTeX documentation file """
    f = open(filename, 'w')
    f.write('')
    f.close()

def add_entry_doc(input_dir, dataset_name, score, filename='doc.tex'):
    """ Add dataset information in LaTeX documentation file """

    public_info_file = open(os.path.join(input_dir, dataset_name, dataset_name+'_public.info'), 'r')
    public_info = public_info_file.read().split('\n')
    private_info_file = open(os.path.join(input_dir, dataset_name, dataset_name+'_private.info'), 'r')
    private_info = private_info_file.read().split('\n')

    f = open(filename, 'a')
    
    f.write('\\begin{center}\n')
    f.write('{\\bf SET 1.2: ARCENE} \\\\\n')
    #f.write('{\\footnotesize')
    f.write('\\vspace{5mm}\n')
    f.write('\\begin{tabular}{|l |c |r |l |l |l |c |l |r |r |r |r |c |}\n')
    f.write('\\hline\n')
    f.write('DATASET & Task & C & Sparse & Miss & Cat & Pte & Ptr & N & Ptr/N & Baseline Score \\\\\n') # Cbal ?
    f.write('\\hline\n')

    f.write(dataset_name + ' & ')
    for info in ['task', 'target_num', 'is_sparse', 'has_missing', 'has_categorical', 'test_num', 'train_num']:
        value = parse_info(public_info + private_info, info) 
        if info == 'task':
            value = value.replace("'", "").split('.')[0]
        f.write(value + ' & ')
        if info == 'test_num':
            Pte = int(value)
        if info == 'train_num':
            Ptr = int(value)        
      
    N = Pte + Ptr
    f.write(str(N) + ' & ')
    f.write(str(round(float(Ptr / N), 3)) + ' & ')
    f.write(str(score) + ' \\\\\n')

    f.write('\\hline\n')
    f.write('\\end{tabular}\n') # }
    f.write('\\end{center}\n')

    # Another description here?
    
    f.write('{\\bf Past Usage:}\n')
    value = parse_info(public_info + private_info, 'past_usage')
    f.write(value + '\n')

    f.write('{\\bf Description:}\n')
    value = parse_info(public_info + private_info, 'description')
    f.write(value + '\n')
    
    f.write('{\\bf Preparation:}\n')
    value = parse_info(public_info + private_info, 'preparation')
    f.write(value + '\n')

    f.write('{\\bf Representation:}\n')
    value = parse_info(public_info + private_info, 'representation')
    f.write(value + '\n')
    
    f.close()
    
def run_baseline_automl(input_dir, dataset_name):
    """ Run baseline model on AutoML format dataset """
    
    X_train = pd.read_csv(os.path.join(input_dir, dataset_name, dataset_name+'_train.data'), sep=' ', header=None).fillna(0)
    X_test = pd.read_csv(os.path.join(input_dir, dataset_name, dataset_name+'_test.data'), sep=' ', header=None).fillna(0)
    y_train = pd.read_csv(os.path.join(input_dir, dataset_name, dataset_name+'_train.solution'), sep=' ', header=None).fillna(0)
    y_test = pd.read_csv(os.path.join(input_dir, dataset_name, dataset_name+'_test.solution'), sep=' ', header=None).fillna(0)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    return round(clf.score(X_test, y_test), 3)

init_tabular()
init_doc()

### Run ###
for dataset_name in dataset_names:

    print(dataset_name + ' dataset')

    # Format dataset
    print('Formatting...')
    dataset_dir, new_dataset_name = press_a_button_and_give_me_an_AutoDL_dataset(
                                         input_dir,
                                         dataset_name,
                                         output_dir,
                                         max_num_examples_train,
                                         max_num_examples_test,
                                         num_shards_train,
                                         num_shards_test)
    print("Congratulations! You pressed a button and you created an AutoDL " +
            "dataset `{}` ".format(new_dataset_name) +
            "with {} maximum training examples".format(max_num_examples_train) +
            "and {} maximum test examples".format(max_num_examples_test) +
            "in the directory `{}`.".format(dataset_dir))

    # Inspection: check integrity
    print('Checking integrity...')
    check_integrity(output_dir, new_dataset_name, check_first_rows=True)

    # Test with baseline
    print('Running baseline...')
    # os.system('python dataset_test/test_with_baseline.py -dataset_name='+dataset_name)
    
    # LOGREG SKLEARN
    #score = run_baseline_automl(input_dir, dataset_name)
    score = 0
    
    # Write LaTeX tabular
    print('Writing LaTeX statistics and documentation...')
    try:
        add_entry_tabular(input_dir, dataset_name, score)
        add_entry_doc(input_dir, dataset_name, score)
    except:
        print('Unable to read dataset information\n')
        f = open('log.txt', 'a')
        f.write('{} dataset: unable to read info files\n'.format(dataset_name))
        f.close()
    
close_tabular()

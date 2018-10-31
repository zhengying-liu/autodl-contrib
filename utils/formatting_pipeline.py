# Author: Adrien PAVAO
# Creation date: 2 Oct 2018
# Description: Script for formatting many datasets from AutoML to AutoDL format

### Imports ###
import os
import sys
import tensorflow as tf

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
#dataset_names = ['adult']


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
    #print('Running baseline...')
    #os.system('python dataset_test/test_with_baseline.py -dataset_name='+dataset_name)

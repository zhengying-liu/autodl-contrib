# Author: Zhengying LIU
# Creation date: 20 Sep 2018
# Description: for a give AutoDL dataset, test it with baseline method in
#   the starting kit of the AutoDL competition bundle
"""To test an AutoDL dataset with the baseline method in starting kit, you need
to first have the starting kit at hand. If not, please git clone the GitHub repo
`zhengying-liu/autodl` at the SAME LEVEL of this repo (`autodl-contrib`):
  `cd <path_to_autodl-contrib>/..`
  `git clone https://github.com/zhengying-liu/autodl.git`
Then you can use the default `starting_kit_dir` value. But you still need to
configure your `dataset_dir` which is the dataset directory. Then you should be
ready to run the command line:
  `cd <dir_of_this_script>`
  `python test_with_baseline.py -dataset_dir='../../formatted_datasets/adult_600_100/' -starting_kit_dir='../../../autodl/codalab_competition_bundle/AutoDL_starting_kit/'`
Remember to change the value of `dataset_dir`.
"""

import tensorflow as tf
import os
import time
import webbrowser
from multiprocessing import Process

tf.flags.DEFINE_string('dataset_dir', '../../formatted_datasets/adult_600_100/',
                       "Directory containing the formatted AutoDL dataset.")

tf.flags.DEFINE_string('starting_kit_dir', '../../../autodl/codalab_competition_bundle/AutoDL_starting_kit/',
                       "Directory containing ingestion program "
                       "`AutoDL_ingestion_program/`, "
                       "scoring program `AutoDL_scoring_program`, "
                       "and the starting kit code "
                       "`AutoDL_sample_code_submission/`.")

FLAGS = tf.flags.FLAGS

def get_path_to_ingestion_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_ingestion_program', 'ingestion.py')

def get_path_to_scoring_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_scoring_program', 'score.py')

if __name__ == '__main__':
  dataset_dir = FLAGS.dataset_dir
  starting_kit_dir = FLAGS.starting_kit_dir
  path_ingestion = get_path_to_ingestion_program(starting_kit_dir)
  path_scoring = get_path_to_scoring_program(starting_kit_dir)
  dataset_dir = '../../formatted_datasets/cifar100/'

  # Run ingestion and scoring at the same time
  command_ingestion = 'python {} {}'.format(path_ingestion, dataset_dir)
  command_scoring = 'python {} {}'.format(path_scoring, dataset_dir)
  def run_ingestion():
    os.system(command_ingestion)
  def run_scoring():
    os.system(command_scoring)
  ingestion_process = Process(name='ingestion', target=run_ingestion)
  scoring_process = Process(name='scoring', target=run_scoring)
  ingestion_process.start()
  scoring_process.start()
  detailed_results_page = os.path.join(starting_kit_dir,
                                       'AutoDL_scoring_output',
                                       'detailed_results.html')
  detailed_results_page = os.path.abspath(detailed_results_page)
  print("detailed_results_page:", detailed_results_page)

  # Open detailed results page in a browser
  time.sleep(2)
  for i in range(30):
    if os.path.isfile(detailed_results_page):
      webbrowser.open('file://'+detailed_results_page, new=2)
      break
    time.sleep(1)

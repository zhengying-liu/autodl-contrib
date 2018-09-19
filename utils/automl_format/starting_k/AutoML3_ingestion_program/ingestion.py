#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# The input directory input_dir (e.g. sample_data/) contains the dataset(s), including:
#   dataname/
# dataname_feat.type
# dataname_public.info
# dataname_train1.data
# dataname_train1.solution
# dataname_train2.data
# dataname_train2.solution
# dataname_test1.data
# dataname_test1.solution
# dataname_test2.data
# dataname_test2.solution
# dataname_test3.data
# dataname_test3.solution
# etc.
#
# Truth values of training and test data are available to your program at some point.
# Cheating prevention: During the final phase, the labels will be made available only
# after predictions are made.
#
# The output directory output_dir (e.g. AutoML3_sample_predictions/)
# will receive the predicted values (no subdirectories):
# 	dataname_test1.predict
# 	dataname_test2.predict
# 	dataname_test3.predict
#	etc.
#
# The code directory submission_program_dir (e.g. AutoML3_sample_code_submission/) should contain your
# code submission model.py (an possibly other functions it depends upon).
#
# We implemented several classes:
# 1) DATA LOADING:
#    ------------
# DataManager
#
# 2) LEARNING MACHINE:
#    ----------------
# Model
#
# The class Model is what the participants need to modify.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# UNIVERSITE PARIS SUD, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL UNIVERSITE PARIS SUD AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon, Zhengying Liu, Hugo Jair Escalante

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 600

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the
# number of points on your learning curve (this is on a log scale, so each
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators
# (base learners).
max_cycle = 1
max_estimators = 1000
max_samples = float('Inf')

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
from os import getcwd as pwd
from os.path import join



root_dir = pwd()     # e.g. '../' or pwd()
default_input_dir = join(root_dir, "AutoML3_sample_data")
default_output_dir = join(root_dir, "AutoML3_sample_predictions")
default_program_dir = join(root_dir, "AutoML3_ingestion_program")
default_submission_dir = join(root_dir, "AutoML3_sample_code_submission")
default_hidden_dir = join(root_dir, "AutoML3_sample_ref")

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================


# Version of the sample code
version = 1

# General purpose functions
import time
import numpy as np
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
from glob import glob as ls

os.system('pip install category_encoders')

# =========================== BEGIN PROGRAM ================================\

if __name__=="__main__" and debug_mode<4:
    #### Check whether everything went well (no time exceeded)
    execution_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
	hidden_dir=default_hidden_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
	hidden_dir = os.path.abspath(argv[3])
        program_dir = os.path.abspath(argv[4])
        submission_dir = os.path.abspath(argv[5])
    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        print("Using hidden_dir: " + hidden_dir)

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)
    path.append (submission_dir + '/AutoML3_sample_code_submission') #IG: to allow submitting the starting kit as sample submission
    import data_io
    from data_io import vprint
    from model import Model
    from data_manager import DataManager
    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)

    data_io.mkdir(output_dir+'/res')

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order

   
    #### Delete zip files and metadata file, if present
    datanames = [x for x in datanames
      if x!='metadata' and not x.endswith('.zip')]

    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = [] # Do not proceed with learning and testing

    #### MAIN LOOP OVER DATASETS:
    overall_time_budget = 0
    time_left_over = 0
    for basename in datanames: # Loop over datasets

        vprint( verbose,  "\n========== Ingestion program version " + str(version) + " ==========\n")
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()

        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "========= Reading and converting data ==========")
        D = DataManager(basename, input_dir, replace_missing=True, max_samples=max_samples, verbose=verbose,testdata=0)	
        vprint( verbose,  "[+] Size of uploaded data  %5.2f bytes" % data_io.total_size(D))
         
        # ======== Keeping track of time
        if debug_mode<1:
            time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        else: 
            time_budget = max_time
        
        overall_time_budget = overall_time_budget + time_budget
        vprint( verbose,  "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint( verbose,  "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        # ========= Creating a model
        vprint( verbose,  "======== Creating model ==========")

        M = Model()

        # Iterating over datasets
        # =======================
        vprint( verbose,  "***************************************************")
        vprint( verbose,  "****** Training model and making predictions ******")
        vprint( verbose,  "***************************************************")
        modelname = os.path.join(submission_dir,basename)

	
        start = time.time()
	# Train the model in the available training batches
	for k in range(D.n_trainbatches):		
		M.fit(D.data[k], D.label[k]) 
	

	vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
	# predict and re train the model
	n_testbatches=len(ls((os.path.join(os.path.join (input_dir , basename ), '*test*data'))))
	print n_testbatches
	for k in range(n_testbatches):
		i=k+1            
		#try:
		E = DataManager(basename, input_dir, replace_missing=True, max_samples=max_samples, verbose=verbose,testdata=i,ltl=0)                
		Y_pred = M.predict(E.data[0])

		
		vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
			
		filename_predict = basename + '_test' + str(i) + '.predict'

		vprint( verbose, "======== Saving results to: " + output_dir)
		data_io.write(os.path.join(output_dir,filename_predict), Y_pred)
		vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
		time_spent = time.time() - start
		time_left_over = time_budget - time_spent
		vprint( verbose,  "[+] End cycle, time left %5.2f sec" % time_left_over)
		if time_left_over<=0: break
		## Do not fit the last batch		
		if i<n_testbatches:
			uE=E
			E = DataManager(basename, hidden_dir, replace_missing=True, max_samples=max_samples, verbose=verbose,testdata=i,ltl=1)	
			M.fit(uE.data[0], E.label[0]) 
			vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
		
		del E

		#except:
		#	test_num = i	
	
	
	overall_time_spent = time.time() - overall_start


    text_file = open(os.path.join(output_dir,'res', 'metadata'), "w")
    text_file.write("elapsedTime: %0.6f" % overall_time_spent)
    text_file.close()


    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)

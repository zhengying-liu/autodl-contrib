# Functions performing various input/output operations for the ChaLearn AutoML challenge

# Main contributor: Arthur Pesah, August 2014
# Edits: Isabelle Guyon, October 2014, May 2018
#        Hugo Jair Escalante, July 2018

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import data_converter
import data_io
from data_io import vprint
import numpy as np
from glob import glob as ls
try:
	import cPickle as pickle
except:
	import pickle
import os
import time

import pandas as pd
from category_encoders import *
import random
from shutil import copyfile


class DataManager:
	''' This class aims at loading and saving data easily with a cache and at generating a dictionary (self.info) in which each key is a feature (e.g. : name, format, feat_num,...).
	Methods defined here are :
	__init__ (...)
		x.__init__([(feature, value)]) -> void		
		Initialize the info dictionary with the tuples (feature, value) given as argument. It recognizes the type of value (int, string) and assign value to info[feature]. An unlimited number of tuple can be sent.
	
	getInfo (...)
		x.getInfo (filename) -> void		
		Fill the dictionary with an info file. Each line of the info file must have this format 'feature' : value
		The information is obtained from the public.info file if it exists, or inferred from the data files        

	getInfoFromFile (...)
		x.getInfoFromFile (filename) -> void		
		Fill the dictionary with an info file. Each line of the info file must have this format 'feature' : value
		
	getFormatData (...)
		x.getFormatData (filename) -> str		
		Get the format of the file ('dense', 'sparse' or 'sparse_binary') either using the 'is_sparse' feature if it exists (for example after a call of getInfoFromFile function) and then determing if it's binary or not, or determining it alone.
		
	getNbrFeatures (...)
		x.getNbrFeatures (*filenames) -> int		
		Get the number of features, using the data files given. It first checks the format of the data. If it's a matrix, the number of features is trivial. If it's a sparse file, it gets the max feature index given in every files.
		
	getTypeProblem (...)
		x.getTypeProblem (filename) -> str		
		Get the kind of problem ('binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'), using the solution file given.
	'''
	
	def __init__(self, basename="", input_dir="", verbose=False, replace_missing=True, filter_features=False, max_samples=float('inf'), testdata=0, ltl=0):
		'''Constructor'''
		self.use_pickle = False # Turn this to true to save data as pickle (inefficient)
		self.basename = basename
   		if basename in input_dir or os.path.isfile(os.path.join(input_dir, basename + '_train.data')) :
			self.input_dir = input_dir 
   		else:
			self.input_dir =  os.path.join (input_dir , basename )
		if self.use_pickle:
			if os.path.exists ("tmp"):
				self.tmp_dir = "tmp"
			elif os.path.exists ("../tmp"):
				self.tmp_dir = "../tmp" 
			else:
				os.makedirs("tmp")
				self.tmp_dir = "tmp"
		info_file = os.path.join (self.input_dir, basename + '_public.info')
		self.info = {}
		self.getInfo (info_file)
         	self.feat_type = self.loadType (os.path.join(self.input_dir, basename + '_feat.type'), verbose=verbose)
		self.data = {}  
		self.label = {}
		#Load training data
		if testdata==0:
			for k in range(len(ls((os.path.join(self.input_dir, '*train*data'))))):
				i=k+1 
				try:
					self.feat_type = self.loadType (os.path.join(self.input_dir, basename + '_feat.type'), verbose=verbose)
					self.data[k] = self.loadDataMV (os.path.join(self.input_dir, basename + '_train' + str(i) + '.data'), verbose=verbose, replace_missing=replace_missing)
					self.label[k] = self.loadData (os.path.join(self.input_dir, basename + '_train' + str(i) + '.solution'), verbose=verbose)
				except:
					self.train_num = i	
		#Load test data
			self.n_trainbatches=i
		else:
			i=testdata                       
			try:
				
				if ltl==0:
					self.label[0]=[]
					self.feat_type = self.loadType (os.path.join(self.input_dir, basename + '_feat.type'), verbose=verbose)
					self.data[0] = self.loadDataMV (os.path.join(self.input_dir, basename + '_test' + str(i) + '.data'), verbose=verbose, replace_missing=replace_missing)
				
				else:
								
					self.label[0] = self.loadData (os.path.join(self.input_dir, basename + '_test' + str(i) + '.solution'), verbose=verbose)	
				
			except:
				self.test_num = i		
		# Normally, feature selection should be done as part of a pipeline.
		# However, here we do it as a preprocessing for efficiency reason
		idx=range(self.feat_type.shape[0])
		#if filter_features: # add hoc feature selection, for the example...
		#	fn = min(Xtr.shape[1], 1000)       
		#	idx = data_converter.tp_filter(Xtr, Ytr, feat_num=fn, verbose=verbose)
		#	Xtr = Xtr[:,idx]
		#	Xva = Xva[:,idx]
		#	Xte = Xte[:,idx]  
		#self.feat_idx = np.array(idx).ravel()
		self.feat_idx =  np.array(idx) 

	def __repr__(self):
		return "DataManager : " + self.basename

	def __str__(self):
		val = "DataManager : " + self.basename + "\ninfo:\n"
		for item in self.info:
			val = val + "\t" + item + " = " + str(self.info[item]) + "\n"
  		val = val + "data:\n"
  		for i in range(len(self.data)):
			val = val + "\tX_" + str(i) + " = array"  + str(self.data[i].shape) + "\n"
			val = val + "\tY_" + str(i) + " = array"  + str(self.label[i].shape) + "\n"
		val = val + "feat_type:\tarray" + str(self.feat_type.shape) + "\n"
		val = val + "feat_idx:\tarray" + str(self.feat_idx.shape) + "\n"
		return val
	## outdated implementation for reading data				
	def loadData (self, filename, verbose=True, replace_missing=True):
		''' Get the data from a text file in one of 3 formats: matrix, sparse, sparse_binary'''
		if verbose:  print("========= Reading " + filename)
		start = time.time()
		if self.use_pickle and os.path.exists (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle")):
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "r") as pickle_file:
				vprint (verbose, "Loading pickle file : " + os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"))
				return pickle.load(pickle_file)
		if 'format' not in self.info.keys():
			self.getFormatData(filename)
                        print "not in self"
		if 'feat_num' not in self.info.keys():
			print "not in self2"
			self.getNbrFeatures(filename)

		data_func = {'dense':data_io.data, 'sparse':data_io.data_sparse, 'sparse_binary':data_io.data_binary_sparse}
		data = data_func[self.info['format']](filename, self.info['feat_num'])	
  
		# IMPORTANT: when we replace missing values we double the number of variables
  
		if self.info['format']=='dense' and replace_missing and np.any(map(np.isnan,data)):
			vprint (verbose, "Replace missing values by 0 (slow, sorry)")
			data = data_converter.replace_missing(data)
		if self.use_pickle:
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "wb") as pickle_file:
				vprint (verbose, "Saving pickle file : " + os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"))
				p = pickle.Pickler(pickle_file) 
				p.fast = True 
				p.dump(data)
		end = time.time()
		if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
		return data
	## updated data reading function, main difference, variables are read per their types
	def loadDataMV (self, filename, verbose=True, replace_missing=True):
		''' Get the data from a text file in one of 3 formats: matrix, sparse, sparse_binary'''
		if verbose:  print("========= Reading " + filename)
		ntime = nnum = ncat = nmvc = 0				
		start = time.time()
		# find the type of features for the data set
                dictfeats=self.feat_type
		usetime = np.array(np.where(self.feat_type=='Time'))[0]
		usenum = np.array(np.where(self.feat_type=='Numerical'))[0]
		usecat = np.array(np.where(self.feat_type=='Categorical'))[0]
		usemulticat = np.array(np.where(self.feat_type=='Multi-value'))[0]		
		if verbose:
			print("=== Detected %d Numerical Features" % len(usenum))
			print("=== Detected %d Categorical Features" % len(usecat))
			print("=== Detected %d Multi-valued Categorical Features" % len(usemulticat))
			print("=== Detected %d Time Features" % len(usetime))
		# artificial headers for features
		for i in range(len(dictfeats)):
			dictfeats[i]=str(i)	
		# read the first column to identify the total number of features 	
                df = pd.read_csv(filename,header=None, names=dictfeats,delim_whitespace=True, usecols=[0], parse_dates=True, na_values='NaN')

		n_samples=len(df.index)
		
		if verbose: print("=== %d Samples will be loaded " % len(df.index))
		concadat= np.zeros((n_samples, 1))
		del df
		
		# Check the available types of features
		if verbose: print("========================")		
		if usetime != []:
			if verbose: print("=== Processing %d Time features " % len(usetime))
		        try:	
				dftime = pd.read_csv(filename,	header=None, names=self.feat_type[usetime], usecols=usetime, delim_whitespace=True,parse_dates=True, na_values='NaN')			
				ddt=np.array(dftime)
				ntime=ddt.shape[1]
				concadat= np.concatenate((concadat,ddt),axis=1)
				del dftime
				del ddt
			except:	print "Failed to load time variables"
		
		if usenum != []:
			if verbose: print("=== Processing %d Numerical features " % len(usenum))
			try:		
				dfnum = pd.read_csv(filename, 	header=None, names=self.feat_type[usenum], usecols=usenum, delim_whitespace=True, na_values='NaN')
				dd=np.array(dfnum)
				nnum=dd.shape[1]
				concadat= np.concatenate((concadat,dd),axis=1)	
				del dfnum
				del dd						
			except:	print "Failed to load numerical variables"
		
		if usecat != []: # categorical features will be loaded as numbers for efficiency
			if verbose: print("=== Processing %d Categorical features " % len(usecat))
			try:	
				dfcat = pd.read_csv(filename, 	header=None, names=self.feat_type[usecat], usecols=usecat,dtype=object, delim_whitespace=True, na_values='NaN')
				# Treat categorical variables as integers or perform hash encoding (one hot encoding is far more expensive)				
				#catnumeric_dataset=np.array(dfcat)
				
				enca = OrdinalEncoder().fit(dfcat)
				catnumeric_dataset = enca.transform(dfcat)				
				catnumeric_dataset = np.array(catnumeric_dataset)
				ncat = catnumeric_dataset.shape[1]
				concadat= np.concatenate((concadat,catnumeric_dataset),axis=1)	
				del dfcat 
				del catnumeric_dataset				
			except:	print "Failed to load Categorical variables"
		
		if len(usemulticat) > 0:
			if verbose: print("=== Processing %d Multi Valued Categorical features " % len(usemulticat))	
			try:	
				dfmvc = pd.read_csv(filename, 	header=None, names=self.feat_type[usemulticat], usecols=usemulticat,  dtype=object, delim_whitespace=True, na_values='NaN')
				enc =  OrdinalEncoder().fit(dfmvc)
				numeric_dataset = enc.transform(dfmvc)
				nude= np.array(numeric_dataset)	
				nmvc = nude.shape[1]
				concadat= np.concatenate((concadat,nude),axis=1)							
			except:	print "Failed to load Multi-Valued Categorical variables"
		
		concadat=np.delete(concadat, 0, 1)
		self.feat_types_up= [ntime, nnum, ncat, nmvc]


		if self.use_pickle and os.path.exists (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle")):
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "r") as pickle_file:
				vprint (verbose, "Loading pickle file : " + os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"))
				return pickle.load(pickle_file)
		if 'format' not in self.info.keys():
			self.getFormatData(filename)
                        print "not in self"
		if 'feat_num' not in self.info.keys():
			self.getNbrFeatures(filename)
		
		data = concadat.astype(np.float64).copy(order='C')

		# IMPORTANT: when we replace missing values we double the number of variables
  
		if self.info['format']=='dense' and replace_missing and np.any(map(np.isnan,data)):
			vprint (verbose, "Replace missing values by 0 (slow, sorry)")
			data = data_converter.replace_missing(data)
		if self.use_pickle:
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "wb") as pickle_file:
				vprint (verbose, "Saving pickle file : " + os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"))
				p = pickle.Pickler(pickle_file) 
				p.fast = True 
				p.dump(data)
		end = time.time()
		if verbose:  
			print( "Loaded %d Samples and %d Features" % (data.shape[0], data.shape[1]))
			print( "[+] Success in %5.2f sec" % (end - start))						
		return data
	

	def loadLabel (self, filename, verbose=True):
		''' Get the solution/truth values'''
		if verbose:  print("========= Reading " + filename)
		start = time.time()
		if self.use_pickle and os.path.exists (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle")):
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "r") as pickle_file:
				vprint (verbose, "Loading pickle file : " + os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"))
				return pickle.load(pickle_file)
		if 'task' not in self.info.keys():
			self.getTypeProblem(filename)
	
           # IG: Here change to accommodate the new multiclass label format
		if self.info['task'] == 'multilabel.classification':
			label = data_io.data(filename)
		elif self.info['task'] == 'multiclass.classification':
			label = data_io.data(filename)
            # IG: I changed that because it was too confusing.
            #label = data_converter.convert_to_num(data_io.data(filename))              
		else:
                        #HJ: I am not sure why this should be used?
			#label = np.ravel(data_io.data(filename)) # get a column vector
			label = np.array([np.ravel(data_io.data(filename))]).transpose() # get a column vector
                        #print label
		if self.use_pickle:
			with open (os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"), "wb") as pickle_file:
				vprint (verbose, "Saving pickle file : " + os.path.join (self.tmp_dir, os.path.basename(filename) + ".pickle"))
				p = pickle.Pickler(pickle_file) 
				p.fast = True 
				p.dump(label)
		end = time.time()
		if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
		return label

	def loadType (self, filename, verbose=True):
		''' Get the variable types'''
		if verbose:  print("========= Reading " + filename)
		start = time.time()
		type_list = []
		if os.path.isfile(filename):
			type_list = data_converter.file_to_array (filename, verbose=False)
		else:
			n=self.info['feat_num']
			type_list = [self.info['feat_type']]*n
		type_list = np.array(type_list).ravel()
		end = time.time()
		if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
		return type_list
  
 	def getInfo (self, filename, verbose=True):
		''' Get all information {attribute = value} pairs from the filename (public.info file), 
              if it exists, otherwise, output default values''' 
		if filename==None:
			basename = self.basename
			input_dir = self.input_dir
		else:   
			
			basename = os.path.basename(filename).split('_')[0]			
			input_dir = os.path.dirname(filename)
		if os.path.exists(filename):
			self.getInfoFromFile (filename)
			vprint (verbose, "Info file found : " + os.path.abspath(filename))
			# Finds the data format ('dense', 'sparse', or 'sparse_binary')   
			self.getFormatData(os.path.join(input_dir, basename + '_train1.data'))
		else:    
			vprint (verbose, "Info file NOT found : " + os.path.abspath(filename))            
			# Hopefully this never happens because this is done in a very inefficient way
			# reading the data multiple times...              
			self.info['usage'] = 'No Info File'
			self.info['name'] = basename
			# Get the data format and sparsity
			self.getFormatData(os.path.join(input_dir, basename + '_train1.data'))
			# Assume no categorical variable and no missing value (we'll deal with that later)
			self.info['has_categorical'] = 0
			self.info['has_missing'] = 0              
			# Get the target number, label number, target type and task               
			self.getTypeProblem(os.path.join(input_dir, basename + '_train1.solution'))
			if self.info['task']=='regression':
				self.info['metric'] = 'r2_metric'
			else:
				self.info['metric'] = 'auc_metric'
			# Feature type: Numerical, Categorical, or Binary
			# Can also be determined from [filename].type        
			self.info['feat_type'] = 'Mixed'  
			# Get the number of features and patterns
			self.getNbrFeatures(os.path.join(input_dir, basename + '_train.data'), os.path.join(input_dir, basename + '_test.data'), os.path.join(input_dir, basename + '_valid.data'))
			self.getNbrPatterns(basename, input_dir, 'train')
			self.getNbrPatterns(basename, input_dir, 'valid')
			self.getNbrPatterns(basename, input_dir, 'test')
			# Set default time budget
			self.info['time_budget'] = 600
		return self.info
                  
	def getInfoFromFile (self, filename):
		''' Get all information {attribute = value} pairs from the public.info file'''
		with open (filename, "r") as info_file:
			lines = info_file.readlines()
			features_list = list(map(lambda x: tuple(x.strip("\'").split(" = ")), lines))
			
			for (key, value) in features_list:
				self.info[key] = value.rstrip().strip("'").strip(' ')
				if self.info[key].isdigit(): # if we have a number, we want it to be an integer
					self.info[key] = int(self.info[key])
		return self.info     

	def getFormatData(self,filename):
		''' Get the data format directly from the data file (in case we do not have an info file)'''
		if 'format' in self.info.keys():
			return self.info['format']
		if 'is_sparse' in self.info.keys():
			if self.info['is_sparse'] == 0:
				self.info['format'] = 'dense'
			else:
				data = data_converter.read_first_line (filename)
				if ':' in data[0]:
					self.info['format'] = 'sparse'
				else:
					self.info['format'] = 'sparse_binary'
		else:
			data = data_converter.file_to_array (filename)
			if ':' in data[0][0]:
				self.info['is_sparse'] = 1
				self.info['format'] = 'sparse'
			else:
				nbr_columns = len(data[0])
				for row in range (len(data)):
					if len(data[row]) != nbr_columns:
						self.info['format'] = 'sparse_binary'
				if 'format' not in self.info.keys():
					self.info['format'] = 'dense'
					self.info['is_sparse'] = 0			
		return self.info['format']
			
	def getNbrFeatures (self, *filenames):
		''' Get the number of features directly from the data file (in case we do not have an info file)'''
		if 'feat_num' not in self.info.keys():
			self.getFormatData(filenames[0])
			if self.info['format'] == 'dense':
				data = data_converter.file_to_array(filenames[0])
				self.info['feat_num'] = len(data[0])
			elif self.info['format'] == 'sparse':
				self.info['feat_num'] = 0
				for filename in filenames:
					sparse_list = data_converter.sparse_file_to_sparse_list (filename)
					last_column = [sparse_list[i][-1] for i in range(len(sparse_list))]
					last_column_feature = [a for (a,b) in last_column]
					self.info['feat_num'] = max(self.info['feat_num'], max(last_column_feature))				
			elif self.info['format'] == 'sparse_binary':
				self.info['feat_num'] = 0
				for filename in filenames:
					data = data_converter.file_to_array (filename)
					last_column = [int(data[i][-1]) for i in range(len(data))]
					self.info['feat_num'] = max(self.info['feat_num'], max(last_column))			
		return self.info['feat_num']
  
  	def getNbrPatterns (self, basename, info_dir, datatype):
		''' Get the number of patterns directly from the data file (in case we do not have an info file)'''
        	line_num = data_converter.num_lines(os.path.join(info_dir, basename + '_' + datatype + '.data'))
        	self.info[datatype+'_num'] =  line_num
		return line_num
		
	def getTypeProblem (self, solution_filename):
     		''' Get the type of problem directly from the solution file (in case we do not have an info file)'''
		if 'task' not in self.info.keys():
			solution = np.array(data_converter.file_to_array(solution_filename))
			target_num = solution.shape[1]
			self.info['target_num']=target_num
			if target_num == 1: # if we have only one column
				solution = np.ravel(solution) # flatten
				nbr_unique_values = len(np.unique(solution))
				if nbr_unique_values < len(solution)/8:
					# Classification
					self.info['label_num'] = nbr_unique_values
					if nbr_unique_values == 2:
						self.info['task'] = 'binary.classification'
						self.info['target_type'] = 'Binary'
					else:
						self.info['task'] = 'multiclass.classification'
						self.info['target_type'] = 'Categorical'
				else:
					# Regression
					self.info['label_num'] = 0
					self.info['task'] = 'regression'
					self.info['target_type'] = 'Numerical'     
			else:
				# Multilabel or multiclass       
				self.info['label_num'] = target_num
				self.info['target_type'] = 'Binary' 
				if any(item > 1 for item in map(np.sum,solution.astype(int))):
					self.info['task'] = 'multilabel.classification'     
				else:
					self.info['task'] = 'multiclass.classification'        
		return self.info['task']
		
		

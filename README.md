# How to Contribute Datasets to AutoDL Challenge
This repo provides instructions and examples to contribute datasets to a repository we are building, in conjuction with the preparation of the [AutoDL challenge](http://autodl.chalearn.org).

## Table of Contents
- [Quick start](#quick-start)
- [What is needed?](#what-is-needed)
- [Formats](#formats)
	- [File Format](#file-format)
	- [TFRecord Format](#tfrecord-format)
- [Carefully Name Your Files](#carefully-name-your-files)
- [Check the Integrity of a Dataset](#check-the-integrity-of-a-dataset)
- [Write info files](#write-info-files)
- [Credits](#credits)

## Quick start

To run the example type the following commands:

```
git clone http://github.com/zhengying-liu/autodl-contrib
cd autodl-contrib
python3 dataset_manager.py /file_format/monkeys
```

## What is needed?

* **multi-label (or multi-class) classification tasks.** 
* **Video, image, text, speech or time series datasets.**
* **No size limit** but if your dataset exceed 10 GB, please [Contact us](mailto:autodl@chalearn.org).


## Where to submit

[Email us](mailto:autodl@chalearn.org) a URL to an on-line storage place (e.g. dropbox or Google drive) when we can pull your data from.


## Formats

* Each example is an independent file.
* Labels are contained in a separate CSV file.
* Meta-data in `private.info`.

Examples are provided in [file_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format) folder.


### TFRecord Format

TFRecord format is the final format we'll actually use in this AutoDL challenge (so when you provide your data under matrix format or file format, thereafter we'll do the conversion job to have a dataset in TFRecord format). Data from all domains will be uniformly formatted to TFRecords following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L92) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview)).

More details and [an example dataset](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format/mini-mnist) in TFRecord format can be found in the sub-directory [`tfrecord_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format).

**Isabelle: the info file with metadata is still needed **

## Carefully Name Your Files

Please name your files carefully such that dataset info can be inferred correctly (by `dataset_manager.py`). Some simple rules apply:
- **metadata** files should follow the glob pattern `*metadata*`;
- **training data** files should follow the glob pattern `*train*`;
- **test data** files should follow the glob pattern `*test*`;
- **examples (or samples)** files should follow the glob pattern `*example*` or `*features*` or `*.data`;
- **labels** files should follow the glob pattern `*label*` or `*.solution`;
- Try to use extension names to make the file type explicit (`*.csv`, `*.tfrecord`, `*.avi`, `*.jpg`, `*.txt`, etc);
- If no `*example*` or `*label*` is specified in the file names of data, these files contain both examples and labels.

**Isabelle: I do not understand these guidelines**

In addition, it's recommended that the name of all files belonging to a given dataset begin with the dataset name.

The following directory `mnist/` gives a valid example:
```
mnist/
├── metadata.textproto
├── mnist-test-examples-00000-of-00002.tfrecord
├── mnist-test-examples-00001-of-00002.tfrecord
├── mnist-test-labels.tfrecord
├── mnist-train-00000-of-00012.tfrecord
├── mnist-train-00001-of-00012.tfrecord
├── mnist-train-00002-of-00012.tfrecord
├── mnist-train-00003-of-00012.tfrecord
├── mnist-train-00004-of-00012.tfrecord
├── mnist-train-00005-of-00012.tfrecord
├── mnist-train-00006-of-00012.tfrecord
├── mnist-train-00007-of-00012.tfrecord
├── mnist-train-00008-of-00012.tfrecord
├── mnist-train-00009-of-00012.tfrecord
├── mnist-train-00010-of-00012.tfrecord
└── mnist-train-00011-of-00012.tfrecord
```

We can see that above is a dataset in **TFRecord format**. Note that TFRecord format is the final format we will really use for the challenge. Data donors don't have to format their data in this form.

A simpler example in **AutoML format** could be
```
iris/
├── iris.data
└── iris.solution
```

Note that in this case, no **metadata** is provided and no train/test split is done yet. However, these two files suffice to construct a valid dataset for AutoDL challenge.

**Isabelle: why do you have inconsistent instructions: sometimes there is a train/test split and sometimes not**

WARNING: as no informative extension name is provided, the dataset will considered to be in `.csv` (!!!) format, as is the case for all datasets in AutoML format.

**Isabelle: this should not be allowed.**

## Check the Integrity of a Dataset
We provide a Python script `dataset_manager.py` that can automatically
- infer the dataset format (e.g. matrix format, file format or TFRecord format)
- infer different components of a dataset (e.g. training data, test data, metadata, etc)
- extract some basic metadata (e.g. `num_examples`, `num_features`) and other info on the dataset

Data donors can follow next steps to check the integrity of their datasets to make sure that these datasets are valid for the challenge:
1. Prepare and format the dataset in one of the possible formats (matrix format, file format, TFRecord format, etc) and put all files into a single directory called `<dataset_name>/`, for example `mnist/`
2. Clone this GitHub repo
    ```
    git clone https://github.com/zhengying-liu/autodl-contrib.git
    cd autodl-contrib
    ```
    and use dataset manager to check dataset integrity and consistency
    ```
    python dataset_manager.py /path/to/your/dataset/
    ```
    (*For now, this script only works for datasets under file format.*)

    After running the script, messages on whether the dataset is valid will be generated.

3. The script will create a YAML file `dataset_info.yaml` in the dataset directory. You can read this file and check if all inferred informations on the dataset are correct as expected;

4. If some info isn't correct, make sure there is no bug in your dataset directory. In some rare cases, you are allowed to modify the file `dataset_info.yaml` manually.

## Write info files

Provide meta-information through `public.info` and `private.info` files. Please follow the structure and syntax of the example below.
If you don't know what to write in a section you can keep it empty with `''` or `0`. Don't worry; we are always going to check the files after your contribution.

##### Example


`digits_public.info` contains meta-information that participants are allowed to access:

```
usage = 'AutoML challenge 2014'
name = 'digits'
domain = 'image'
task = 'multiclass.classification'
target_type = 'Categorical'
feat_type = 'Numerical'
metric = 'bac_metric'
time_budget = 300
feat_num =  1568
target_num =    10
label_num =    10
train_num = 15000
valid_num = 20000
test_num = 35000
has_categorical =     0
has_missing =     0
is_sparse =     0
```


`digits_private.info` contains meta-information that participants are NOT allowed to access:

```
title = 'MNIST handwritten digit dataset'
name = 'Digits'
keywords = 'handwriting.recognition,digit.recognition,OCR'
authors = 'Yann LeCun, Corinna Cortes, and Chris Burges'
resource_url = 'http://yann.lecun.com/exdb/mnist/'
contact_name = 'Isabelle Guyon'
contact_url = 'http://clopinet.com/isabelle/'
license = 'http://creativecommons.org/about/cc0'
date_created = '30-Sep-2014'
past_usage = 'Many methods have been tried on the MNIST database, in its original data split (60,000 training examples, 10,000 test examples, 10 classes). See http://yann.lecun.com/exdb/mnist/. This dataset was used in the NIPS 2003 Feature Selection Challenge under the name GISETTE and in the WCCI 2006 Performance Prediction Challenge and the IJCNN 2007 Agnostic Learning vs. Prior Knowledge Challenge under the name GINA, and in the ICML 2011 Unsupervised and Transfer Learning Challenge under the name ULE.'
description = 'This is a dataset of handwritten digits. It is a subset of a larger set available from NIST. The digits in pixel representation have been size-normalized and centered in a fixed-size image by the authors. The data are quantized on 256 gray level values.'
preparation = 'For the purpose of the AutoML challenge, all samples were merged and the data were freshly randomly split in three sets: training, validation, and test. The order of the features (pixels) was also randomize, after adding a few distractor features (probes) that are permuted versions of real features.'
representation = 'pixels'
remarks = 'This dataset is very famous!'
zip_size = 'Unkown'
real_feat_num =   784
feat_type = { 'Numerical' 'Categorical' 'Binary' }
label_names = { 'Two' 'Seven' 'Six' 'Nine' 'Zero' 'Three' 'Five' 'One' 'Four' 'Eight' }
```

_Note that `name` section represents the fake name of the dataset._

## Credits
AutoDL is a project in collaboration with Google, ChaLearn and Inria
Please contact us via email: autodl@chalearn.org.

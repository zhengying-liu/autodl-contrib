# How to Contribute Datasets to AutoDL Challenge
This repo is intended for giving instructions and examples on how to contribute datasets to AutoDL, a new data science challenge on Automatic Deep Learning we are organizing with Google, ChaLearn and Inria.


## Table of Contents
- [Two Words on AutoDL](#two-words-on-autodl)
- [Reward for Data Donors](#reward-for-data-donors)
- [Rights of Data](#rights-of-data)
- [What is needed?](#what-is-needed)
- [3 Possible Formats](#3-possible-formats)
	- [Matrix Format](#matrix-format)
	- [File Format](#file-format)
	- [TFRecord Format](#tfrecord-format)
- [Carefully Name Your Files](#carefully-name-your-files)
- [Check the Integrity of a Dataset](#check-the-integrity-of-a-dataset)
- [Contact us](#contact-us)

## Two Words on AutoDL
The AutoDL challenge (planned for 2019, hopefully) is going to be the next big thing in the field of automatic machine learning (AutoML). It challenges participants to find fully automatic solution for designing deep (machine) learning models. This means that participants' (one single) algorithm should be able to construct machine learning models for all tasks (i.e. dataset + metric) in this challenge.

## Reward for Data Donors
By contributing datasets to this challenge, data donors will benefit from:
1. Having their datasets published to our repository [LINK TBA]
2. Being invited to write a joint paper on the repository [to be submitted to the JMLR MLOSS track or elsewhere]
3. Being invited to contribute to the paper on analyses of the challenge [to be published in a book in the Springer CiML series or elsewhere]
4. Being acknowledged in presentations and publications on the challenge.

## Rights of Data
Data donors are are responsible for properly anonimizing their data and should release it under an open data license [SUGGESTION LIST]. The data will remain available in the repository beyond the challenge termination.

## What is needed?
As the tasks in this first edition of AutoDL challenge will all be **multi-label classification** tasks, all you need to provide is **examples** (a.k.a samples or features) with corresponding **labels**, and optionally some **metadata**. Datasets used in AutoDL cover all kinds of different domains such as video, image, text, speech, etc. You are thus welcome to contribute any datasets of any kinds.

All datasets will ultimately be formatted into a uniform format (TFRecords) then provided to participants of the challenge. However, to facilitate the work of data donors, we accept [3 possible formats](https://github.com/zhengying-liu/autodl-contrib#3-possible-formats). Quite many existing datasets are already in one of these 3 formats or require very few modification.

For a given dataset, all its files should be under the **same directory**. And note that **no train/test split is required**. The organizers can carry out train/test split themselves.

There is **no size limit** for datasets in this challenge (not as in previous AutoML challenges). (*@Isabelle: but computing resource is very limited. Good thing is that using metric such as area under learning curve doesn't require complete convergence of participants' algorithm.*) All data will be stored on Google Cloud Platform and we accept and even welcome large datasets. On the other hand, this means that participants are challenged to write algorithms that are able to deal with tasks of very different scales.

## Three Possible Formats
We accept dataset contributions under 3 possible formats:
- Matrix format
- File format
- TFRecord format

### Matrix Format
Under matrix format, each example is represented as a *feature vector*, as is the case in many classic machine learning settings. If you are familiar with `scikit-learn`, you should be familiar with this matrix representation of datasets (e.g. `X`, `y`). So if you want to contribute data in this format, the minimum kit would be **two CSV files**: `examples.csv` and `labels.csv`, containing respectively a matrix (`X`) with feature vectors and a matrix (`Y`) with labels.

Note that, this is the standard [AutoML format](https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data) used in prior AutoML challenges, from [2015](https://competitions.codalab.org/competitions/2321) to [2018](http://prada-research.net/pakdd18/index.php/call-for-competition/).

More details and [an example dataset](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format/iris-AutoML) in matrix format can be found in the sub-directory [`matrix_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format).

### File Format
Under file format, each example is an independent file (this is usually the case for large examples such as videos) and all labels are contained in another file.

More details and [2 example datasets](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/monkeys) in file format can be found in the sub-directory [`file_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format).

### TFRecord Format
TFRecord format is the final format we'll actually use in this AutoDL challenge (so when you provide your data under matrix format or file format, thereafter we'll do the conversion job to have a dataset in TFRecord format). Data from all domains will be uniformly formatted to TFRecords following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L92) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview)).

More details and [an example dataset](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format/mnist) in TFRecord format can be found in the sub-directory [`tfrecord_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format).


## Carefully Name Your Files

Please name your files carefully such that dataset info can be inferred correctly (by `dataset_manager.py`). Some simple rules apply:
- **metadata** files should follow the glob pattern `*metadata*`;
- **training data** files should follow the glob pattern `*train*`;
- **test data** files should follow the glob pattern `*test*`;
- **examples (or samples)** files should follow the glob pattern `*example*` or `*features*` or `*.data`;
- **labels** files should follow the glob pattern `*label*` or `*.solution`;
- Try to use extension names to make the file type explicit (`*.csv`, `*.tfrecord`, `*.avi`, `*.jpg`, `*.txt`, etc);
- If no `*example*` or `*label*` is specified in the file names of data, these files contain both examples and labels.

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

WARNING: as no informative extension name is provided, the dataset will considered to be in `.csv` (!!!) format, as is the case for all datasets in AutoML format.

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

## Contact us
Please contact us via email:  
Zhengying Liu   <zhengying.liu@inria.fr>  
Isabelle Guyon  <guyon@chalearn.org>

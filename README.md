# How to Contribute Datasets to AutoDL Challenge
This repo is intended for giving instructions and examples on how to contribute datasets to AutoDL, a new data science challenge on Automatic Deep Learning we are organizing with Google, ChaLearn and Inria.

We strongly encourage different entities to contribute their own data to this challenge, enriching the database of datasets and making the challenge's results more robust and convincing. In return, data donors can benefit from a direct machine learning solution for their own problems, after a competitive challenge of the state of the art. Lastly, being credited for a cool challenge like this one tends to be a pretty nice thing to do. :)

## Two Words on AutoDL
AutoDL challenge is going to be the next big thing in the field of automatic machine learning (AutoML). It challenges participants to find fully automatic solution for designing deep (machine) learning models. This means that participants' (one single) algorithm should be able to construct machine learning models for all tasks (i.e. dataset + metric) in this challenge.

## What is needed?
As the tasks in this first edition of AutoDL challenge will all be **multi-label classification** tasks, all you need to provide is **examples** with corresponding **labels**, and optionally some **metadata**. Datasets used in AutoDL cover all kinds of different domains such as video, image, text, speech, etc. You are thus welcome to contribute any datasets of any kinds.

Please note that **no train/test split is required**. The organizers carry out train/test split themselves.

## 3 Possible Formats
We accept dataset contributions under 3 possible formats:
- Matrix format
- File format
- TFRecord format

### Matrix Format
Under matrix format, each example is represented as a *feature vector*, as is the case in many classic machine learning settings. If you are familiar with `scikit-learn`, you should be familiar with this matrix representation of datasets (e.g. `X`, `y`). So if you want to contribute data in this format, the minimum kit would be two CSV files: `examples.csv` and `labels.csv`, containing respectively a matrix (`X`) with feature vectors and a matrix (`Y`) with labels.

Note that, this is the standard [AutoML format](https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data) used in prior AutoML challenges, from [2015](https://competitions.codalab.org/competitions/2321) to [2018](http://prada-research.net/pakdd18/index.php/call-for-competition/).

More details and an example dataset in matrix format can be found in the sub-directory [`matrix_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format).

### File Format
Under file format, each example is an independent file (this is usually the case for large examples such as videos) and all labels are contained in another file.
*TODO (Find some dataset examples in file format)*

More details and an example dataset in file format can be found in the sub-directory [`file_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format).

### TFRecord Format
TFRecord format is the final format we'll actually use in this AutoDL challenge (so when you provide your data under matrix format or file format, thereafter we'll do the conversion job to have a dataset in TFRecord format). Data from all domains will be uniformly formatted to TFRecords following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L292) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview).

More details and an example dataset in TFRecord format can be found in the sub-directory [`tfrecord_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format).


## Check the Integrity of a Dataset
We provide a Python script `dataset_manager.py` that can automatically
- infer the dataset format (e.g. matrix format, file format or TFRecord format)
- infer different components of a dataset (e.g. training data, test data, metadata, etc)
- extract some basic metadata (e.g. `num_examples`, `num_features`) and other info on the dataset

Donors of data can follow these steps to check the integrity of his/her datasets to make sure that these datasets are valid for the challenge:
1. Prepare and format the dataset in one of the possible formats (matrix format, file format, TFRecord format, etc) and put all files into a single directory called `<dataset_name>/`, for example `mnist/`
2. Clone this GitHub repo
```
git clone https://github.com/zhengying-liu/autodl-contrib.git
cd autodl-contrib
```
and use dataset manager to check dataset integrity and consistency
```
python dataset_manager.py /path/to/your/dataset
```
3. This will create a YAML file `dataset_info.yaml` in the dataset directory. You can read this file and check if all inferred info on the dataset are correct

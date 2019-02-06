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


## Credits
AutoDL is a project in collaboration with Google, ChaLearn and Inria
Please contact us via email: autodl@chalearn.org.

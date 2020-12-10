# Format Datasets for AutoDL Challenges
We provide instructions and examples to format your own datasets for the [AutoDL](http://autodl.chalearn.org) challenges in the generic TF records format being used.


## Quickstart

In the directory containing your personal projects (e.g. `~/projects/`), clone the two GitHub repos of AutoDL:
```bash
git clone http://github.com/zhengying-liu/autodl-contrib
git clone http://github.com/zhengying-liu/autodl
```
Add the directory `autodl-contrib` to `PYTHONPATH`
```bash
cd autodl-contrib
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
and test using
```bash
python -c "import utils.dataset_manager"
```
If no error message is produced, you are good to go.

Then you can choose to
1. work on your local machine (and make local changes); or
2. work in a Docker container (in the same environment as AutoCV/AutoDL challenge).

### Option 1: Work on Your Local Machine
Install [Python 3](https://www.anaconda.com/distribution/) (Anaconda 3) and install **Tensorflow >= 1.12 and < 2** by:
```
conda install tensorflow
```
If you have any doubt about your version of Tensorflow:
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
make sure the version >= 1.12.0 and < 2. Then install necessary packages
```
pip3 install -r requirements.txt
```
Then you should have a work environment ready for formatting datasets and preparing baseline methods (and make submissions).

### Option 2: Work in Docker
Instead of installing packages and make local changes, you can also choose to work in the Docker image that is used for AutoCV/AutoDL challenge: `evariste/autodl:cpu`. Thus, by using in this Docker image, you are working in the exact same environment in which participants' submissions are handled.

To do this, you first need to [setup Docker](https://www.docker.com/products/docker-desktop). Then go to `~/projects/` and run
```bash
docker run --memory=4g -it -u root -v $(pwd):/app/codalab evariste/autodl:cpu bash
```
In the Docker container session, you can check the version of tensorflow by running
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
and you should get `1.12.0`.

### Now Let's Format a Dataset!

After installing the work environment (on your local machine or in Docker), begin a quick dataset formatting by entering the following commands:
```
cd autodl-contrib
python3 check_n_format.py file_format/mini-cifar
```
To answer [Y] to all questions, just keep hitting "return": this should be good enough to check that everything is running smoothly.
When you see images pop-up, check that they are displayed properly and the labels correspond well. You may also see an HTML file with a learning curve pop-up in your browser. Your formatted data (in the AutoDL format) ends up in `file_format/mini-cifar_formatted`.

To create your own dataset, create a new directory at the same level than mini-cifar a fill it with your own data, then re-run the same script.

NB: The script `check_n_format.py` also accepts other arguments to customize dataset parameters. For example, say if you have an image dataset of 4 channels instead of the default 3 RGB channels, you can run:
```
python3 check_n_format.py -raw_dataset_dir=path/to/your/dataset -num_channels=4
```
This will generate a dataset containing 4-D tensors of shape
```
(sequence_size, row_count, col_count, num_channels) = (1, ?, ?, 4)
```
If you look at `metadata.textproto` files of the formatted datasets, you should notice the difference.

## What is needed?

* **Multi-label (or multi-class) classification tasks.**
* **Video, image, text, speech or time series datasets.**
* **No size limit**

If your dataset exceed 10 GB or if you have a regression task please [Contact us](mailto:autodl@chalearn.org).

To convert tabular datasets from a csv format to the AutoDL format, see the [utils/automl_format folder](https://github.com/zhengying-liu/autodl-contrib/tree/master/utils/automl_format).

## Where to submit

[Email us](mailto:autodl@chalearn.org) a URL to an on-line storage place (e.g. dropbox or Google drive) where we can pull your data from.


## Formats

There are three formats from which you can transform your data into AutoDL's format.

#### 1. File format
* Each image is an independent file (jpg, png).
* Labels are contained in a separate `labels.csv` file.
* Meta-data in `private.info`. Please edit by hand to supply information needed.

Examples are provided in [file_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format) folder.

_Works for images, videos, sounds (time series)._

#### 2. AutoML format

Examples are provided in [matrix_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format) folder.

_Creates a tabular dataset._

#### 3. Raw text/speech format

\[To describe\]

It's the format used in AutoSpeech and AutoNLP challenge. We have script that converts the data from this format to the AutoDL (TFRecords) format.

_Works for text and time series._


### Understanding check_n_format.py

This script does the following:

* Read the meta-data in `private.info`.
* Compute simple statistics about the data (file number, etc.) and check consistency with the CSV file containing the labels.
* Train/test split data.
* Format the data to TFRecord format.
* Run baseline.
* Ask the user to check a few samples manually.


TFRecord format is the final format of the AutoDL challenge, following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L92) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview)).

More details and [an example dataset](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format/mini-mnist) in TFRecord format can be found in the sub-directory [`tfrecord_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/tfrecord_format).


## Credits
AutoDL is a project in collaboration with Google, ChaLearn and Inria.

Please contact us via [email](mailto:autodl@chalearn.org).

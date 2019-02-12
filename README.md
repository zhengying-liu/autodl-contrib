# Format Datasets for AutoCV/AutoDL Challenges
We provide instructions and examples to format your own datasets for the AutoCV and [AutoDL](http://autodl.chalearn.org) challenges in the generic TF records format being used.


## Quickstart

In the directory containing your personal projects (e.g. `~/projects/`), clone the two GitHub repos of AutoDL:
```bash
git clone http://github.com/zhengying-liu/autodl-contrib
git clone http://github.com/zhengying-liu/autodl
```
Then you can choose to 
1. work on your local machine (and make local changes); or
2. work in a Docker container (in the same environment as AutoCV/AutoDL challenge).

### Option 1: Work on Your Local Machine
Install [Python 3](https://www.anaconda.com/distribution/) (Anaconda 3) and install Tensorflow 1.12 by:
```
conda install tensorflow 
```
If you have any doubt about your version of Tensorflow:
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
make sure the version >= 1.12.0. Then install necessary packages 
```
pip3 install -r requirements.txt
```
Then you should have a work environment ready for formatting datasets and preparing baseline methods (and make submissions).

### Option 2: Work in Docker
Instead of installing packages and make local changes, you can also choose to work in the Docker image that is used for AutoCV/AutoDL challenge: `evariste/autodl`. Thus, by using in this Docker image, you are working in the exact same environment in which participants' submissions are handled.

To do this, you first need to [setup Docker](https://www.docker.com/products/docker-desktop). Then go to `~/projects/` and run
```bash
docker run --memory=4g -it -u root -v $(pwd):/app/codalab evariste/autodl bash
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

## What is needed?

* **Multi-label (or multi-class) classification tasks.**
* **Video, image, text, speech or time series datasets.**
* **No size limit**

If your dataset exceed 10 GB or if you have a regression task please [Contact us](mailto:autodl@chalearn.org).


## Where to submit

[Email us](mailto:autodl@chalearn.org) a URL to an on-line storage place (e.g. dropbox or Google drive) where we can pull your data from.


## Formats

* Each image is an independent file (jpg, png).
* Labels are contained in a separate `labels.csv` file.
* Meta-data in `private.info`. Please edit by hand to supply information needed.

Examples are provided in [file_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format) folder.


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

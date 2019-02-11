# Format Datasets for AutoCV/AutoDL Challenges
We provide instructions and examples to format your own datasets for the AutoCV and [AutoDL](http://autodl.chalearn.org) challenges in the generic TF records format being used.


## Quick start

Install [Python 3.7](https://www.anaconda.com/distribution/) (Anaconda 3) and install Tensorflow 1.12 by:
```
conda install tensorflow 
```
If you have any doubt about your version of Tensorflow:
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
then enter the following commands:

```
git clone http://github.com/zhengying-liu/autodl-contrib
git clone http://github.com/zhengying-liu/autodl
cd autodl-contrib
pip3 install -r requirements.txt
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

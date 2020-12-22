# Format Datasets for AutoDL Challenges
We provide instructions and examples to format your own datasets in the generic TFRecords format used in [AutoDL](http://autodl.chalearn.org) challenge series.


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
Instead of installing packages and make local changes, you can also choose to work in the Docker image that is used for AutoDL challenge: `evariste/autodl:cpu-latest`. Thus, by using in this Docker image, you are working in the exact same environment in which participants' submissions are handled.

To do this, you first need to [setup Docker](https://www.docker.com/products/docker-desktop). Then go to `~/projects/` and run
```bash
docker run --memory=4g -it -u root -v $(pwd):/app/codalab evariste/autodl:cpu-latest bash
```
Make sure not to run it directly in the `autodl-contrib/` folder, because you will need to access the `autodl/` folder to run the scripts.

In the Docker container session, you can check the version of tensorflow by running
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```
and you should get `1.13.1`.

This option may not work at the moment when it comes to formatting video datasets (if so, option 1 is recommended).

### Now Let's Format a Dataset!

After installing the work environment (on your local machine or in Docker), begin a quick dataset formatting by entering the following commands:
```
cd autodl-contrib
python3 check_n_format.py file_format/mini-cifar
```
To answer [Y] to all questions, just keep hitting "return": this should be good enough to check that everything is running smoothly.
When you see images pop-up, check that they are displayed properly and the labels correspond well. You may also see an HTML file with a learning curve pop-up in your browser. Your formatted data (in the AutoDL format) ends up in `file_format/mini-cifar_formatted`.

To create your own dataset, create a new directory at the same level than mini-cifar a fill it with your own data, then re-run the same script:

```bash
python3 check_n_format.py path/to/your/dataset
```
This will generate a dataset containing 4-D tensors of shape `(sequence_size, row_count, col_count, num_channels)`.


## What is needed?

* **Multi-label (or multi-class) classification tasks**
* **Video, image, text, speech (or time series) or tabular datasets**
* **No size limit**
* **Prepare your dataset in one of the 3 formats presented below**

If your dataset exceed 10 GB or if you have a regression task please [Contact us](mailto:autodl@chalearn.org).


## Formats

There are three formats from which you can transform your data into AutoDL's format.

#### 1. File format
* Each data is an independent file (jpg, png, bmp, gif or wav, mp3 or avi, mp4).
* Labels are contained in a separate `labels.csv` file.
* Meta-data in `private.info`. Please edit by hand to supply information needed.

Examples and documentation are provided in [file_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format) folder.

_Works for images, videos, sounds (time series)._

#### 2. AutoML format
* All the data is provided in a csv format, with space as a delimiter and with no header.
* The dataset is already divided in train, valid (can be empty) and test sets.
* Labels for each set are provided in separated `.solution` files.

Examples and documentation are provided in [matrix_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format) folder.

_Works for tabular dataset._

#### 3. AutoNLP format
* All the data is provided in a txt format : each line of text is an example in the dataset.
* The dataset is already divided in train and test sets with separated `.data` files corresponding to each set.
* Labels for each set are provided in separated `.solution` files.
* A `meta.json` metadata file is necessary in order to specify the number of train and test samples, the number of classes and the language (english or chinese) 

Examples and documentation are provided in [nlp_format](https://github.com/zhengying-liu/autodl-contrib/tree/master/nlp_format) folder.

It's the format used in AutoNLP challenge. We have a script that converts the data from this format to the AutoDL (TFRecords) format.

_Works for text data._


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

## Format unlabelled data
You may want to format unseen data, i.e. data for which you do not have labels yet. These missing labels could then be predicted by the [AutoDL Self-Service](https://competitions.codalab.org/competitions/27082). If your data has no label files, you can use the `format_unseen.py` script to turn it into TFRecords and then be able use it in the AutoDL Self-Service for making predictions :

```
cd autodl-contrib
python3 format_unseen.py path/to/your/data output_dim path/to/output/directory
```

You must provide a `output_dim` parameter which corresponds to the number of classes of the dataset (we still work with multilabel classification task). The `path/to/output/directory` is optionnal, but it may be judicious to put the path to your dataset formatted with `check_n_format.py`. It will create a `unlabelled` directory in the output folder, containing two files :

* A `.tfrecords` file with your data
* A `metadata.textproto` file

Two questions will be asked while the script is running : the domain of your dataset and the number of channels (especially for images and videos, for other domains there is only one channel).

The script can convert unlabelled data from the five domains mentionned above : images, videos, series (or speech), tabular and text. The data must be formatted in one of the three formats mentionned above, with little variation due to absence of labels.

### Under File format (image, video, speech/series)
The file format for unlabelled data is very simple : it's the same as the one required by `check_n_format.py`, except that there is no `labels.csv` file, but you must provide in your input directory a file `data.csv` which will list all the files which form your unseen data, e.g.

```
n0159.jpg
n0165.jpg
n0167.jpg
...
```

This will allow the script to make the arrangement you want for your unseen data. Note that there is no header, while there should be one in a `labels.csv`.

### Under AutoML format (tabular)
If you are working with tabular datasets, your input directory should contain only one file : your unlabelled data in a single csv file named `data.csv`. Your features must be separated with a space and no header must be provided. Thus, it should look like :

```
1 2 3 4 
5 6 7 8
9 10 11 12
...
```
if you had four features.


### Under AutoNLP format (text) 
For text datasets, your unlabelled data must be in the form of a text file listing examples as text lines : the name of the file should be `unlabelled.data`. You must also provide the same `train.data` and `train.solution` files you used for converting your train and test sets with `check_n_format.py`. It will allow the script to know the vocabulary of your NLP dataset. You must also provide a `meta.json` file similar to the one you provided to `check_n_format.py`. You will fill it with :

```
{
    "class_num": <number of classes>,
    "train_num": <number of train samples>,
    "test_num": <number of unlabelled samples>,
    "language": <"EN" or "CH">,
    "time_budget": 2400
}
```
And all this files must remain in one single directory, there is no need for a `dataset.data` subdirectory here.

```
unlabelled_data/
├── train.data/
├── meta.json
├── unlabelled.data
└── train.solution
```

## Where to submit

If you wish to share your dataset with us, simply [Email us](mailto:autodl@chalearn.org) a URL to an on-line storage place (e.g. dropbox or Google drive) where we can pull your data from.

## Credits
AutoDL is a project in collaboration with Google, ChaLearn and Inria.

Please contact us via [email](mailto:autodl@chalearn.org).

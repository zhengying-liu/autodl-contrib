# File Format

We accept dataset contributions under **file format**.

Under file format, each example is an independent file (this is usually the case for large examples such as videos) and all labels are contained in another file.

## Supported File Types
File format is extremely convenient when examples in the dataset are indeed *files*. This is the case when you already have a database of files (e.g. images, videos, audios) with corresponding labels. If you want to contribute data in file format, you can simply provide these files plus the labels contained in a CSV file. We'll further format these files into TFRecords using `dataset_manager.py`. Some preferred file types are:
- images: `.jpg`, `.png`, `.gif`
- videos: `.avi`, `.mp4`
- audios: `.mp3`, `.wav`

If you want to contribute data in `.txt` type, you need to provide **tokenized data**, after some typical pre-processing in NLP (stemming, lemmatization, stopwords removing, argumentation, etc). Then each examples is represented by a **series of integers** which are indexes of words. [An example](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/randomtext) of such datasets can be found in this directory too.

## Carefully Name Your Files
Remember, you need to be careful about naming your files in a dataset and follow our [file naming rules](https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files).

## How Should These Files Look Like?

The directory of a valid dataset in file format may look like
```
monkeys/
├── monkeys_labels_file_format.csv
├── n0159.jpg
├── n0165.jpg
├── n0167.jpg
├── n1034.jpg
├── n1110.jpg
├── n1128.jpg
├── n2019.jpg
├── n2020.jpg
├── n2048.jpg
├── n3021.jpg
├── n3029.jpg
├── n3043.jpg
├── n4016.jpg
├── n41559.jpg
├── n41562.jpg
├── n5046.jpg
├── n5122.jpg
├── n5143.jpg
├── n6128.jpg
├── n6134.jpg
├── n6146.jpg
├── n7031.jpg
├── n7049.jpg
├── n7145.jpg
├── n8061.jpg
├── n8120.jpg
├── n8136.jpg
├── n9158.jpg
├── n9159.jpg
└── n9162.jpg
```

Well yes, the files don't really follow our naming rules (with no `*example*` pattern etc). You are allowed to do this only if **you precise that the dataset is in file format in the file name of lables**, as in `monkeys_labels_file_format.csv`.

**WARNING**: in this CSV file, there should be a column called `FileName` to indicate the corresponding files.

Note that besides having dense representation (e.g. 0-1 vectors), the labels can also have **sparse representation**. This means that we can have **label-confidence pairs**. For example,
```
FileName,LabelConfidencePairs
n0159.jpg,2 0.488458 9 0.48776 0 0.486832
n7031.jpg,1 0.544249 0 0.53526 7 0.490038 3 0.48924 2 0.485278 6 0.481826
...
```

## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory.

Additional files can contain all sorts of metadata (number of examples, column names, column type, donor name, etc).

## We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).

For now, we don't consider regression problems.

## Check the Integrity of Your Dataset
Remember to [check the integrity of your dataset](https://github.com/zhengying-liu/autodl-contrib#check-the-integrity-of-a-dataset) before donating.

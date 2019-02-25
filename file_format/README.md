# File Format

We accept dataset contributions under **file format**.

Under file format, each example is an independent file (this is usually the case for large examples such as videos) and all labels are contained in another file.

## Supported File Types
File format is the best choice when examples are indeed *files*; if you already has a database of files (e.g. images, videos, audios) with corresponding labels. If you want to contribute data in file format, you can simply provide these files plus the labels contained in a CSV file. We can further format these files into TFRecords using `check_n_format.py` (currently images only) or `data_manager.py`. Some preferred file types are:
- images: `.jpg`, `.png`, `.gif`
- videos: `.avi`, `.mp4`
- audios: `.mp3`, `.wav`

Examples can have a **fixed size** (height and width) or be **iregular** (each example has its own dimensions).

If you want to contribute data in `.txt` type, you need to provide **tokenized data**, after some typical pre-processing in NLP (stemming, lemmatization, stopwords removing, argumentation, etc). Then each examples is represented by a **series of integers** which are indexes of words. [An example](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/randomtext) of such text datasets can be found in this directory too.


## How Should These Files Look Like?

The directory of a valid dataset in file format may look like
```
monkeys/
├── labels.csv
├── labels.name
├── private.info
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

The `labels.csv` file should have two comma separated columns: `FileName` and `Labels`.
```
FileName, Labels
n0159.jpg, 0 9
n7031.jpg, 0 1 7
n8136.jpg, 5
...
```
The second column is a space separated list of numerical labels (since all problems are multi-label problems). Labels are numbered 0 to c-1, where c is the total number of categories. 

The `labels.name` file should contain the names of the labels, one per line, in the order of the numerica labels: 0 to c-1, 0 corresponding to the first line. 

## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory.

Additional files can contain all sorts of metadata (number of examples, column names, column type, donor name, etc).

## We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).

If you have a regression task, contact us and we'll probably turn it into a classification task (categorical regression).

## Label confidence pairs
You can also provide labels as **label confidence pairs** as in the following example [NOT RECOMMENDED]:
```
FileName, LabelConfidencePairs
n0159.jpg, 2 0.488458 9 0.48776 0 0.486832
n7031.jpg, 1 0.544249 0 0.53526 7 0.490038 3 0.48924 2 0.485278 6 0.481826
...
```

# File Format

Raw data must be supplied in **file format**.

Under the file format, each example is an independent file and all labels are contained in a separate CSV file.

## Supported File Types
For images and videos, preferred file types are:
- images: `.jpg`, `.png`, `.gif`
- videos: `.avi`, `.mp4`
- audios: `.mp3`, `.wav`

The images or videos may be of a **fixed size** (height and width) or **irregular size** (each example has its own dimensions).

For text data, you may contribute text files with a `.txt` extension. We prefer **tokenized data**, after some typical pre-processing in NLP (stemming, lemmatization, stopwords removing, argumentation, etc). Each examples is represented by a **series of integers** which are indexes of words. We provide [an example](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/randomtext) of such text datasets.


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
├──  
etc.
```

The `labels.csv` file should have two comma separated columns: `FileName` and `Labels`.
```
FileName, Labels
n0159.jpg, 0 9
n0165.jpg, 0 1 7
n0167.jpg, 5
etc.
```
The second column is a space separated list of numerical labels (since all problems are multi-label problems). Labels are numbered 0 to c-1, where c is the total number of categories. 

The `labels.name` file should contain the names of the labels, one per line, in the order of the numerica labels: 0 to c-1, 0 corresponding to the first line.

```
Baboon
Chimp
Gorilla
etc.
```

## What to do next?
You can convert these files into TFRecords using `check_n_format.py` (currently images only) or `data_manager.py`. See [INSTRUCTIONS](https://github.com/zhengying-liu/autodl-contrib).

## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory. Please add a private.info file.
```
title : 'Monkeys image example dataset'
name : 'Monkeys'
keywords : 'monkeys.image.recognition'
authors : 'Authors'
resource_url : ''
contact_name : 'Isabelle Guyon'
contact_url : 'http://clopinet.com/isabelle/'
license : 'Unknown'
date_created : 'date'
past_usage : ''
description : ''
preparation : ''
representation : 'pixels'
remarks : 'This is a toy dataset.'
```

## Important: We Only Accept Multi-label Classification Datasets
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

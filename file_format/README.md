# File Format

Raw data must be supplied in **file format**.

Under the file format, each example is an independent file and all labels are contained in a separate CSV file.

## Supported File Types
For images and videos, preferred file types are:
- images: `.jpg`, `.bmp`, `.png`, `.gif`
- videos: `.avi`, `.mp4`
- audios: `.mp3`, `.wav`

The images or videos may be of a **fixed size** (height and width) or **irregular size** (each example has its own dimensions).

For text data, you may contribute text files with a `.txt` extension. We prefer **tokenized data**, after some typical pre-processing in NLP (stemming, lemmatization, stopwords removing, argumentation, etc). Each examples is represented by a **series of integers** which are indexes of words. We provide [an example](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/randomtext) of such text datasets.


## How Should These Files Look Like?

The directory of a valid dataset in file format may look like
```
monkeys/
├── labels.csv
├── label.name
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
n0159.jpg,0 9
n0165.jpg,0 1 7
n0167.jpg,5
etc.
```
The second column is a space separated list of numerical labels (since all problems are multi-label problems). Labels are numbered 0 to c-1, where c is the total number of categories. 

The `label.name` file should contain the names of the labels, one per line, in the order of the numerica labels: 0 to c-1, 0 corresponding to the first line.

```
Baboon
Chimp
Gorilla
...
```

## What to do next?
You can convert these files into TFRecords using `check_n_format.py` (currently images and videos only) or `data_manager.py`. See [INSTRUCTIONS](https://github.com/zhengying-liu/autodl-contrib).

## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory. 

### Private Info File
Please add a private.info file.
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
### Feature Name and Label Name Files
To enable visualization (or possibly transfer learning), you are welcome to provide additional information on the names of features and labels (although it's optional). This is crucial for visualizing **image** datasets and **text** datasets because without this information we can only visualize integers instead of the real names of features/labels. More details are provided in the following.

#### Feature Name File
Each dataset in File Format can be attached with a file under the name `feat.name`, which is a CSV file with only *one* column. Each row of index `i` is a string indicating the name of feature `i` in the dataset. Take text datasets as example, we use one-hot encoding to represent each word in the vocabulary, say of size `V`. Then the `feature_name.csv` should contain `V` rows where each row is a word in the vocaubulary. Note that since we are dealing with example tensors of shape `(sequence_size, row_count, col_count, num_channels)` in this challenge, the number of feature names will eventually be equal to `col_count`. One example of feature name file can be found [here](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/randomtext).

#### Label Name File
Each dataset in File Format can be attached with a file under the name `label.name`, which is a CSV file with only *one* column. Each row of index `i` is a string indicating the name of label `i` in the dataset. Since we are dealing with label tensors of shape `(output_size,)` in this challenge, the number of label names should be equal to `output_size`. One example of label name file can be found [here](https://github.com/zhengying-liu/autodl-contrib/tree/master/file_format/monkeys).

## Important: We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).
If you have a regression task, contact us and we'll probably turn it into a classification task (categorical regression).

## Label confidence pairs [NOT RECOMMENDED]
You can also provide labels as **label confidence pairs** as in the following example:
```
FileName, LabelConfidencePairs
n0159.jpg, 2 0.488458 9 0.48776 0 0.486832
n7031.jpg, 1 0.544249 0 0.53526 7 0.490038 3 0.48924 2 0.485278 6 0.481826
...
```

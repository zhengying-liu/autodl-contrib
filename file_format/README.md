# File Format

We accept dataset contributions under **file format**.

Under file format, each example is an independent file (this is usually the case for large examples such as videos) and all labels are contained in another file.

## Carefully Name Your Files
Remember, you need to be careful about naming your files in a dataset and follow our [file naming rules](https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files).

## How Should These Files Look Like?

The directory of a valid dataset in file format looks like
```
monkeys/
├── monkey_labels.csv
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

Note that the labels contained in `monkey_labels.csv` can have **sparse representation**. This means that we can have **label-confidence pairs**. For example,
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

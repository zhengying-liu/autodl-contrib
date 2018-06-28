# Matrix Format (AutoML Format)

We accept dataset contributions under **matrix format**. More specifically, we accept the standard **AutoML format** used in prior AutoML challenges. There exist [detailed guidance]((https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data)) on how to format dataset into AutoML format.

Under matrix format, each example is represented as a *feature vector*, as is the case in many classic machine learning settings. If you want to contribute data in this format, the minimum package would be two CSV files: `examples.csv` and `labels.csv`, containing respectively a matrix (`X`) with feature vectors and a matrix (`Y`) with labels.

## Carefully Name Your Files
Remember, you need to be careful about naming your files in a dataset. According to our [file naming rules](https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files), you may name these `.csv` files differently. For example, this directory
```
iris/
├── iris.data
└── iris.solution
```

defines a valid dataset contribution. Here `iris.data` will be understood as `examples.csv` and `iris.solution` as `labels.csv`.

## How Should These Files Look Like?

For the dataset `iris`, each example in `iris.data` is characterized by 4 numeric values (i.e. a vector of dimension 4). So this CSV file looks like
```
5.1 3.5 1.4 0.2
4.9 3 1.4 0.2
4.7 3.2 1.3 0.2
4.6 3.1 1.5 0.2
...
```

Categorical variables and sparse representation are accepted too.

And `iris.solution` could be something like
```
1 0 0
0 0 1
0 1 0
1 0 0
...
```

We see that each label is a 0-1 vector indicating one of the three classes each example belongs to.

Actually the labels don't always need to be 0-1 vectors. They can be more generally (probability) **distributions** on all classes. Plus, each distribution is allowed NOT to be normalized, for example
```
1 1 0
1 0 1
0 1 0
1 0 0
...
```

is valid.

Once again, sparse representation is accepted. This means that in `labels.csv`, we can have **label-confidence pairs**. For example,
```
VideoId,LabelConfidencePairs
JVEF,2377 0.488458 2105 0.48776 3346 0.486832
JoEF,1 0.544249 0 0.53526 14 0.490038 36 0.48924 94 0.485278 153 0.481826
...
```

In this case, an additional column of `Id` is also contained.


## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory, for example,
```
iris-AutoML/
├── iris.data
├── iris.solution
├── iris_feat.name
├── iris_info.m
├── iris_label.name
└── iris_sample.name
```

Additional files can contain all sorts of metadata (number of examples, column names, column type, donor name, etc).

You can find above example in [this repo](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format/iris-AutoML).

## We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).

For now, we don't consider regression problems.

## Check the Integrity of Your Dataset
Remember to [check the integrity of your dataset](https://github.com/zhengying-liu/autodl-contrib#check-the-integrity-of-a-dataset) before donating.

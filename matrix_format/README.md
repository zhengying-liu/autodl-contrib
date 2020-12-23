# Matrix Format (AutoML Format)

In the matrix format, each example is represented as a *feature vector*. If you are familiar with `scikit-learn`, you should be familiar with this matrix representation of datasets (e.g. `X`, `y`). So if you want to convert data from this format, the minimum kit would be **text files**: to begin with, `dataset_train.data` and `dataset_train.solution`, containing respectively a matrix (`X`) with feature vectors and a matrix (`Y`) with labels, examples in lines and features in columns. You must also provide similar `dataset_test.data`, `dataset_test.solution`, `dataset_valid.data` and `dataset_valid.solution` files (the latters can be empty, because the validation set and the train set will be merged, but the files must exist).

This follows the standard [AutoML format](https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data) used in prior AutoML challenges, from [2015](https://competitions.codalab.org/competitions/2321) to [2018](http://prada-research.net/pakdd18/index.php/call-for-competition/). The format includes metadata that we encourage you to provide.

More details and [an example dataset](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format/iris-AutoML) in matrix format can be found in the sub-directory [`matrix_format`](https://github.com/zhengying-liu/autodl-contrib/tree/master/matrix_format).

<!-- Isabelle: I think this is confusing and error prone to have both AutoML and CSV formats. We should stick to the AutoML format. It is more general. There is also a sparse matrix version. We also need more metadata. This should NOT be optional -->

We accept dataset contributions under **matrix format**. More specifically, we accept the standard **AutoML format** used in prior AutoML challenges. There exist [detailed guidance](https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data) on how to format dataset into AutoML format.

<!-- Under matrix format, each example is represented as a *feature vector*, as is the case in many classic machine learning settings. If you want to contribute data in this format, the minimum package would be two CSV files: `examples.csv` and `labels.csv`, containing respectively a matrix with feature vectors and a matrix with labels. -->

## Carefully Name Your Files
Remember, you need to be careful about naming your files in a dataset. According to our [file naming rules](https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files), you may name these `.csv` files differently. For example, this directory
```
iris/
├── iris_train.data
├── iris_train.solution
├── iris_test.data
├── iris_test.solution
├── iris_valid.data
└── iris_valid.solution
```

defines a valid dataset contribution. Here `iris_train.data` will be understood as `examples.csv` and `iris_train.solution` as `labels.csv`.

## How Should These Files Look Like?

For the dataset `iris`, each example in `iris_train.data` is characterized by 4 numeric values (i.e. a vector of dimension 4). So this CSV file looks like
```
5.1 3.5 1.4 0.2
4.9 3 1.4 0.2
4.7 3.2 1.3 0.2
4.6 3.1 1.5 0.2
...
```

Categorical variables and sparse representation are accepted too.

And `iris_train.solution` could be something like
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

Sparse representation is accepted. This means that in `labels.csv`, we can have **label-confidence pairs**. For example,
```
Id,LabelConfidencePairs
JVEF,2377 0.488458 2105 0.48776 3346 0.486832
JoEF,1 0.544249 0 0.53526 14 0.490038 36 0.48924 94 0.485278 153 0.481826
...
```

In this case, an additional column of `Id` is also contained.


## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory, for example,
```
iris-AutoML/
├── iris_train.data
├── iris_train.solution
├── iris_test.data
├── iris_test.solution
├── iris_valid.data
├── iris_valid.solution
├── iris_feat.name
├── iris_public.info
├── iris_private.info
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

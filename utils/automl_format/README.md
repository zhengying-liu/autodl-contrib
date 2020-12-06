# Generate AutoDL datasets from datasets in AutoML format.

The `format_automl_new.py` script allows to generate a tabular dataset in the format used for AutoDL from a dataset already formatted in the AutoML format, i.e. essentially csv files.

Run a command line (in the current directory) with for example:

`python format_automl_new.py -input_dir='../../raw_datasets/automl/' -output_dir='../../formatted_datasets/' -dataset_name=adult`

Please change `input_dir` to the right directory on your disk containing the AutoML datasets. Under this directory, there should be a folder named `dataset_name`, say `adult/`. The files in this folder should be organized as
the following:

```
adult
├── adult_feat.name (optional)
├── adult_label.name (optional, but recommended)
├── adult_private.info (optional)
├── adult_public.info (optional)
├── adult_test.data
├── adult_test.solution
├── adult_train.data
├── adult_train.solution
├── adult_valid.data (required, but can be empty, will be merged to train)
└── adult_valid.solution (required, but can be empty)
```
The `.data` files are CSV files **with space as separator**. Each one of them represents a matrix of shape `(num_examples, num_features)` so each example is a vector of shape `(num_features,)`.

As in AutoDL challenge, each example is a tensor of shape `(sequence_size, row_count, col_count, num_channels) = (T, H, W, C)`, we should have `num_features = T * H * W * C`.

To get a vector of shape `(num_features,)` from a tensor of shape `(T, H, W, C)`, we can typically do a flattening (e.g. by calling `numpy.ravel`). And in order to be able to reconstruct the tensor during the challenge, the shape information (e.g. T, H, W, C) should be provided as arguments when calling this script. For
example:

```
python format_automl.py -input_dir='../../raw_datasets/automl/' -output_dir='../../formatted_datasets/' -dataset_name=adult -sequence_size=1 -row_count=1 -col_count=24 -num_channels=1
```

The `.solution` files, containing the labels, must be already **one-hot** encoded.

For more detailed guidance on AutoML format, please see [this page](https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data).

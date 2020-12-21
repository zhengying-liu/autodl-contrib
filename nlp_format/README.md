# AutoNLP format

Text data must be supplied in the **AutoNLP format**.

Under the AutoNLP format, each example is a line in a `.data` file and all corresponding labels are contained in separate files.

## How Should These Files Look Like?

The directory of a valid dataset in file format may look like

```
dataset/
├── dataset.data/
├────── meta.json
├────── test.data
├────── train.data
├────── train.solution
└── dataset.solution
```

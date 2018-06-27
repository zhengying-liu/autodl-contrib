# Matrix Format (AutoML Format)

We accept dataset contributions under Matrix Format.

## Carefully Naming Your Files

Please name your files carefully such that dataset info (contained in `dataset_info.yaml`) can be inferred correctly. Some simple rules apply:
- **metadata** files should follow the glob pattern `*metadata*`;
- **training data** files should follow the glob pattern `*train*`;
- **test data** files should follow the glob pattern `*test*`;
- **examples data** files should follow the glob pattern `*example*` or `*features*` or `*.data`;
- **training data** files should follow the glob pattern `*label*` or `*.solution`;

In addition, it's recommended that the name of all files belonging to a given dataset begin with dataset name.

As example, the following directory has a valid naming system:
```
mnist/
├── metadata.textproto
├── mnist-test-examples-00000-of-00002.tfrecord
├── mnist-test-examples-00001-of-00002.tfrecord
├── mnist-test-labels.tfrecord
├── mnist-train-00000-of-00012.tfrecord
├── mnist-train-00001-of-00012.tfrecord
├── mnist-train-00002-of-00012.tfrecord
├── mnist-train-00003-of-00012.tfrecord
├── mnist-train-00004-of-00012.tfrecord
├── mnist-train-00005-of-00012.tfrecord
├── mnist-train-00006-of-00012.tfrecord
├── mnist-train-00007-of-00012.tfrecord
├── mnist-train-00008-of-00012.tfrecord
├── mnist-train-00009-of-00012.tfrecord
├── mnist-train-00010-of-00012.tfrecord
└── mnist-train-00011-of-00012.tfrecord

```

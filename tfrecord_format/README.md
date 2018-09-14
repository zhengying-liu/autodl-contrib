# TFRecord Format

We accept dataset contributions under **TFRecord format**.

TFRecord format is the final format we'll actually use in this AutoDL challenge (so when you provide your data under matrix format or file format, thereafter we'll do the conversion job to have a dataset in TFRecord format). Data from all domains will be uniformly formatted to TFRecords following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L92) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview)).

**WARNING:** Formatting data into valid TFRecord format requires solid understanding of TFRecord and SequenceExample protocol buffers, as well as the design of dataset API we provide in this challenge. Thus providing data in this format is actually **NOT** recommended.

## Carefully Name Your Files
Remember, you need to be careful about naming your files in a dataset and follow our [file naming rules](https://github.com/zhengying-liu/autodl-contrib#carefully-name-your-files).

## How Should These Files Look Like?

The directory of a valid dataset in file format looks like
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

Note that this directory has already **train/test split** and **separated labels**, which are NOT required for data donors.

In the TFRecord file `mnist-train-00000-of-00012.tfrecord`, each example looks like
<pre><code>context {
  feature {
    key: "id"
    value {
      int64_list {
        value: 0
      }
    }
  }
  feature {
    key: "label_index"
    value {
      int64_list {
        value: 7
      }
    }
  }
  feature {
    key: "label_score"
    value {
      float_list {
        value: 1.0
      }
    }
  }
}
feature_lists {
  feature_list {
    key: "0_dense_input"
    value {
      feature {
        float_list {
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 255.0
          value: 141.0
          value: 0.0
          [<em>...More pixel-wise numerical values</em>]
        }
      }
    }
  }
}
</code></pre>

We provide a Python script to generate this example from scratch. You can do it by running
```
python convert_mnist_to_tfrecords.py
```
in the current directory. The script contains more technical details on how to format dataset in TFRecord format.

## Explain SequenceExample in natural language
If you haven't looked at what SequenceExample proto is in the official [code](https://www.tensorflow.org/code/tensorflow/core/example/example.proto),
you can look at this paragraph to have a general idea.

SequenceExample proto is a kind of protocol to store data. This protocol can be
followed to format dataset in TFRecord format (TensorFlow's official data format).
One such dataset consists of a list of **sequence examples**.
Each sequence example has two components: `context` and `feature_lists`.
In a Python-like jargon, `context` can be considered as a dict mapping
each feature name to feature (i.e. a list of numbers). `feature_list` is then
considered as a dict mapping each feature name to a list of features
(i.e. a list of lists of numbers).

To know more (very important) details about how to start hands-on experience
on formatting datasets in TFRecord format, please do readings in the following
section and read the codes in the directory `autodl-format-definition`. Some
important paragraphs would be:
- the function `_parse_function` in the class `AutoDLDataset` in the script `dataset.py`;
- the whole file `data.proto`, which defines the final challenge format.
But be aware: **we don't want to format datasets following this proto.** Instead,
we'll do this following SequenceExample proto and then do some parsing using
`_parse_function`;

Notice that the script `data_pb2.py` is automatically generated with the
compiler `protoc` from `data.proto`. It's not recommended to read, but you can
read it anyway if interested.

## Readings
In order to understand better what TFRecords are and how to work with them, we strongly recommend to read the following references if you really want to contribute data in TFRecord format:
- A [basic introduction](https://developers.google.com/protocol-buffers/docs/pythontutorial) on **Protocol Buffers** for Python programmers;
- After reading above introduction, you can find the definition of two important `proto`'s (short for Protocol Buffers) in the source code of TensorFlow:
  - [Feature](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto) proto;
  - [Example](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) proto, in which we find the extremely important definition of **SequenceExample** proto that we'll use in this challenge.
- The [Consuming TFRecord data](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) section of TensorFlow's official documentation;
- Other blog articles on this topics, for example [this article](https://planspace.org/20170323-tfrecords_for_humans/).

## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory.

Additional files can contain all sorts of metadata (number of examples, column names, column type, donor name, etc).

## We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).

For now, we don't consider regression problems.

## Check the Integrity of Your Dataset
Remember to [check the integrity of your dataset](https://github.com/zhengying-liu/autodl-contrib#check-the-integrity-of-a-dataset) before donating.

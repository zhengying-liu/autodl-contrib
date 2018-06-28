# TFRecord Format

We accept dataset contributions under **TFRecord format**.

TFRecord format is the final format we'll actually use in this AutoDL challenge (so when you provide your data under matrix format or file format, thereafter we'll do the conversion job to have a dataset in TFRecord format). Data from all domains will be uniformly formatted to TFRecords following [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L292) proto (see [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview)).

WARNING: Formatting data into valid TFRecord format requires solid understanding of TFRecord and SequenceExample protocol buffers, as well as the design of dataset API we provide in this challenge. Thus providing data in this format is actually NOT recommended.

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

Each example looks like
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


## Providing More Info Is Always Well-received
You are of course welcome to add more informations in your dataset directory.

Additional files can contain all sorts of metadata (number of examples, column names, column type, donor name, etc).

## We Only Accept Multi-label Classification Datasets
In multi-label classification tasks, each example can belong to several classes (i.e. have several labels).

For now, we don't consider regression problems.

## Check the Integrity of Your Dataset
Remember to [check the integrity of your dataset](https://github.com/zhengying-liu/autodl-contrib#check-the-integrity-of-a-dataset) before donating.

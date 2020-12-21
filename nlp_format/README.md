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

The train set and the test set must already be separated in two files. A `.data` file is a list of examples, each example being a line of text. It sould indeed look like (*extracted from the O1 dataset*)
```
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films 
apparently reassembled from the cutting-room floor of any given daytime soap . 
they presume their audience wo n't sit still for a sociology lesson , however entertainingly presented , so they trot out the conventional science-fiction elements of bug-eyed monsters and futuristic women in skimpy clothes . 
this is a visually stunning rumination on love , memory , history and the war between art and commerce . 
jonathan parker 's bartleby should have been the be-all-end-all of the modern-office anomie films . 
campanella gets the tone just right -- funny in the middle of sad in the middle of hopeful . 
a fan film that for the uninitiated plays better on video with the sound turned down . 
béart and berling are both superb , while huppert ... is magnificent . 
a little less extreme than in the past , with longer exposition sequences between them , and with fewer gags to break the tedium . 
the film is strictly routine .
...
```

The `.solution` files are specifying the labels corresponding to each line of the corresponding `.data` file (`train.solution` corresponds to the labels of the examples contained in `train.data` while `dataset.solution` corresponds to `test.data`).

A solution file must look like
```
0 1
1 0
1 0
0 1
0 1
0 1
1 0
0 1
1 0
1 0
0 1
1 0
1 0
0 1
0 1
0 1
```

In these files, the labels must be one-hot encoded : each column corresponds to a class and a 1 means that the example is in the corresponding class, while the 0 means that it's not.

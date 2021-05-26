
# A Structured Self-attentive Sentence Embedding

## TeamSmallisTheNewBig

![jupyter notebook x python](https://img.shields.io/badge/jupyter%20notebook-python-orange)![DeepLearning x DNN](https://img.shields.io/badge/DeepLearning-DNN-blue)![ML x Regressions](https://img.shields.io/badge/MachineLearning-Regressions-ff69b4)

We worked relentlessly as a united team to understand and improve the neural network described in the paper [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130) by Lin et al. We started by extracting the attention matrix relevant to the yelp data and then applying Deep Learning concepts we learned through out the course to improve its performance.
This project was done under the supervision of Professor Lelarge at Ecole polytechnique, during his Deep Learning course. Inès Multrier, Jean-Charles Layoun and Tom Sander are equal contributors.

## Path to original data

We expect you to put the **small** original data in the `small` folder of our directory.

## Notebooks explained

### Preprocessing

To generate the dataset, you will need to install [spacy](https://spacy.io/usage) and run:

```
python tokenizer-yelp.py --input [Yelp dataset] --output [output path, will be a json file] --dict [output dictionary path, will be a json file]
```

A small version of the tokenized dataset is available [here](https://www.di.ens.fr/~lelarge/small_yelp.zip).

In order to get the Glove vectors as PyTorch tensors, you can use [torchtext](https://github.com/pytorch/text), see [here](https://github.com/dataflowr/self-attentive-sentence-embedding/blob/main/glove_tensors.ipynb). For convenience, I did it for [glove.6B.200d.txt.pt](https://www.di.ens.fr/~lelarge/glove.6B.200d.txt.pt).

### Visualization

The notebook `Plot_train_logs.ipynb` plots the training accuracy and loss on validation set after each epoch. You can execute it to visualize the learning curves shown in our presentation. 

### Model Analysis and Feature Extraction

In `Model_Analysis.ipynb`, you will find the code used to extract the attention matrix and to count the parameters of the model.

### Heatmap

In `annotation_matrix.ipynb ` and `plot_annotation_matrix.py`, you will find the script that generated our heatmaps.

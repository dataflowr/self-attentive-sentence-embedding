# A Structured Self-attentive Sentence Embedding

Mini-project for the [deep learning course](http://dataflowr.com/) based on [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130) by Lin et al.

The code has been adapted from [the repo of Freda Shi](https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding).

# Preprocessing

To generate the dataset, you will need to install [spacy](https://spacy.io/usage) and run:
```
python tokenizer-yelp.py --input [Yelp dataset] --output [output path, will be a json file] --dict [output dictionary path, will be a json file]
```

A small version of the tokenized dataset is available [here](https://www.di.ens.fr/~lelarge/small_yelp.zip).

In order to get the Glove vectors as PyTorch tensors, you can use [torchtext](https://github.com/pytorch/text), see [here](https://github.com/dataflowr/self-attentive-sentence-embedding/blob/main/glove_tensors.ipynb). For convenience, I did it for [glove.6B.200d.txt.pt](https://www.di.ens.fr/~lelarge/glove.6B.200d.txt.pt).

# Running on Colab

Now, provided you downloaded everything on [Colab](https://github.com/dataflowr/self-attentive-sentence-embedding/blob/main/sase_colab.ipynb), the training can be done via:
```
python train.py data.train_data="/content/small/train_tok.json" data.val_data="/content/small/val_tok.json" data.test_data="/content/small/test_tok.json" data.dictionary="/content/small/dict_review_short.json" data.word_vector="content/glove.6B.200d.txt.pt" data.save="/content/self-attentive-sentence-embedding/models/model-small-6B.pt"
```




# A Structured Self-attentive Sentence Embedding

## TeamSmallisTheNewBig

![jupyter notebook x python](https://img.shields.io/badge/jupyter%20notebook-python-orange)![DeepLearning x DNN](https://img.shields.io/badge/DeepLearning-DNN-blue)![ML x Regressions](https://img.shields.io/badge/MachineLearning-Regressions-ff69b4)

We worked relentlessly as a united team to understand and improve the neural network described in the paper *A STRUCTURED SELF-ATTENTIVE
SENTENCE EMBEDDING*. We started by extracting the attention matrix relevant to the yelp data and then applying Deep Learning concepts we learned through out the course to improve its performance.

## Path to original data

We expect you to put the **small** original data in the `small` folder of our directory.

## Folder Hierarchy explained

### Visualization

The notebook `Plot_train_logs.ipynb` plots the training accuracy and loss on validation set after each epoch. You can execute it to visualize the learning curves shown in our presentation. 

### Model Analysis and Feature Extraction

In `Model_Analysis.ipynb`, you will find the code used to extract the attention matrix and to count the parameters of the model.

### Basic Regression

In the **Basic Regression** folder, you will find `Basic_Regression_Models.ipynb`. Execute all cells in order to train and evaluate all our basic regression models.

### Simon neural network model

In the folder named **Simon-NN**, you will find `Simon-NN.ipynb`. By executing this notebook, you will train your own version of **Simon**, our fist neural network model.

### DNN_BERT neural network model

In the folder named **DNN_BERT**, you will find `BERT_Tokenizer for Tweets.ipynb`  and `DNN_BERT.ipynb`. You will need to execute `BERT_Tokenizer for Tweets.ipynb` first to compute the tokens that will be saved in the folder `Tensors`. When the script terminates, run `DNN_BERT.ipynb`. By executing this notebook, you will train your own version of **DNN_BERT**, our second neural network model.




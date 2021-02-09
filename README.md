# A Structured Self-attentive Sentence Embedding

Mini-project for the [deep learning course](http://dataflowr.com/) based on [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130) by Lin et al.

The code has been adapted from [the repo of Freda Shi](https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding).

# Preprocessing

To generate the dataset, you will need to install [spacy](https://spacy.io/usage) and run:
```
python tokenizer-yelp.py --input [Yelp dataset] --output [output path, will be a json file] --dict [output dictionary path, will be a json file]
```

A small version of the tokenized dataset is available [here](https://www.di.ens.fr/~lelarge/small_yelp.zip).

In order to get the Glove vectors as PyTorch tensors, you can use [torchtext](https://github.com/pytorch/text). For convenience, I did it for [glove.6B.200d.txt.pt](https://www.di.ens.fr/~lelarge/glove.6B.200d.txt.pt).

# Running on Colab

Now, provided you downloaded everything on [Colab](https://github.com/dataflowr/self-attentive-sentence-embedding/sase_colab.ipynb), the training can be done via:
```
python train.py data.train_data="/content/small/train_tok.json" data.val_data="/content/small/val_tok.json" data.test_data="/content/small/test_tok.json" data.dictionary="/content/small/dict_review_short.json" data.word_vector="content/glove.6B.200d.txt.pt" data.save="/content/self-attentive-sentence-embedding/models/model-small-6B.pt"
```

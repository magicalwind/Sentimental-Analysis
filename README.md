# Sentimental-Analysis

The repo is designed for sentiment analysis based on [PyTorch](https://github.com/pytorch/pytorch) and [TorchText](https://github.com/pytorch/text). Two different architectures (Bi-LSTM and CNN) are built for this objective. 

## Getting Started

Pytorch can be easilyt installed on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torchtext
```

SpaCy is used for tokenization in the program. To install spaCy, follow the instructions [here](https://spacy.io/usage/) ##Note## install the English models:

``` bash
python -m spacy download en
```

- ## Bi-directional LSTM.ipynb
Load and create dataset (split test/train), create a vocab with built-in glove representation and customized max_size (or min_freq)  (Thanks to the powerful function of torchtext, those can be done with only a couple of sentences). Use BucketIterator to shuffle and bucket the input sequences into sequences with similar length for efficient batch padding. 
Build Naive Bi-LSTM with regularization for training and evaluation

- ## Training with CNN.ipynb
Follow the same basic workflow. Implement a CNN encoder to simulate n-grams tokenization, which refers to [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) and [Convolutional Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb).

- ## Word vector of Glove
Play with the word vectors in the Glove embedding space. Compare the similarity of different words and find their semantic analogy. 

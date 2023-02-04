# models.py

from calendar import EPOCH
from tokenize import Double
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from tqdm import tqdm

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        ### we need to set predict function, based on one sentence as ex_word is list[str] ###  
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class NeuralNet(nn.Module):

    def __init__(self, embeddings: nn.Embedding, hidden_dim: int):
        super().__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        # TODO: Your code here!
        # defining input shape, and output shape
        n_in = int(self.embeddings.get_embedding_length()) # 50
        n_out = 2 # class = 2

        # create a nn that has one ReLU function
        self.net = nn.Sequential(
            nn.Linear(n_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_out),
        )

        # define a log softmax function to convert nn result into log-probability distribution over classes
        self.log_softmax = nn.LogSoftmax(dim=-1)
                

    def forward(self, ex_words: List[str]) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.

        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """
        # TODO: Your code here!
        #   1. Embed each word using self.embeddings.
        ex_words_embedded = np.array([self.embeddings.get_embedding(ex_word) for ex_word in ex_words])

        #   2. Take the average embedding of each word.
        avg_ex_words_embedded = np.sum(ex_words_embedded, axis=0) / len(ex_words)

        #   3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
        embedding_nn_passed = self.net(torch.tensor(avg_ex_words_embedded).float())

        #   4. Return a tensor representing a log-probability distribution over classes. 
        return self.log_softmax(embedding_nn_passed)
        


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    # we simply inherit the model 
    def __init__(self, model):
        super().__init__()
        self.model = model

    # and add predict function
    def predict(self, ex_words: List[str]) -> int:
        # TODO: Your code here!
        probs = self.model.forward(ex_words)
        return torch.argmax(probs).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # TODO: Your code here!

    ### we need to call a model with given hidden size, and given word_embeddings (which is glove embeddings for now)
    net = NeuralNet(word_embeddings, args.hidden_size)
    
    # Set up our criterion - our loss function 
    criterion = nn.NLLLoss()
    
    # Set up our optimizer. We need to tell the optimizer which parameters it will optimize over (which parameters it is allowed to modify).
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ### epoch iteration
    for _ in tqdm(range(args.num_epochs)):
        # shuffle dataset
        random.shuffle(train_exs)

        # iterate for each rows and conduct forward/backward process
        for exs in train_exs:
            # we define input and target 
            input = exs.words
            target = torch.tensor(exs.label)

            # Always zero-out the gradients managed by your optimizer
            optimizer.zero_grad()
            
            # Put model into training mode. This does not do anything
            net.train()        

            # Compute the predicted log-probabilities
            y_hat = net.forward(input)

            # Compute the loss
            train_loss = criterion(y_hat, target)

            # Back-propagate the gradients to the parameters
            train_loss.backward()

            # Apply the gradient updates to the parameters
            optimizer.step()

    # return the classifier 
    return NeuralSentimentClassifier(net)
import numpy as np

from ngram_vanilla import NGramVanilla


class NGramBackoff(NGramVanilla):
    def __init__(self, n, vsize):
        self.n = n
        self.sub_models = [NGramVanilla(k, vsize) for k in range(1, n + 1)]

    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability with backoff.
        
        That is, if the n-gram count of size self.n is defined, return that.
        Otherwise, check the n-gram of size self.n - 1, self.n - 2, etc. until you find one that is defined.
        
        Hint: Refer to ngram_prob in ngrams_vanilla.py.
        """
        # TODO: Your code here!
        # as similar to interpolation, start from trigram by reversing sub_model index
        for sub_model in self.sub_models[::-1]:
            # calculate the probability 
            prob = sub_model.ngram_prob(ngram)        
            # if it is greater than 0, return that probability
            if prob > 0: return prob
            # else, parse the ngram and iterater for the next model
            else: ngram = ngram[1:]
        # after all, if the prob is 0 for all models, return 0
        return 0

        # End of your code.

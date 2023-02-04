import numpy as np
from collections import defaultdict

from ngram_vanilla import NGramVanilla


class NGramAdditive(NGramVanilla):
    def __init__(self, n, delta, vsize):
        self.n = n
        self.delta = delta
        self.count = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(float)
        self.vsize = vsize

    def ngram_prob(self, ngram):
        """Return the smoothed probability of an ngram with additive smoothing.
        
        Hint: Refer to ngram_prob in ngrams_vanilla.py.
        """
        # TODO: Your code here!
        # divide prefix as ngram (-) last word, and word as last ngram
        prefix = ngram[:-1]
        word = ngram[-1]
        
        # as per formula defined in the textbook, we add delta to numerator, and add delta*vsize to denominator
        return (self.count[prefix][word] + self.delta) / (self.total[prefix] + (self.delta*self.vsize))
        # End of your code.
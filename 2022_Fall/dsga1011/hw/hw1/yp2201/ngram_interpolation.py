import numpy as np

from ngram_vanilla import NGramVanilla, NEG_INFINITY

class NGramInterpolation(NGramVanilla):
    def __init__(self, lambdas, vsize):
        self.lambdas = lambdas
        self.vsize = vsize
        self.sub_models = [NGramVanilla(n, vsize) for n in range(1, len(lambdas) + 1)]
    
    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def sequence_logp(self, sequence):
        n = len(self.lambdas)
        padded_sequence = ['<bos>']*(n - 1) + sequence + ['<eos>']
        total_logp = 0
        for i in range(len(padded_sequence) - n + 1):
            ngram = tuple(padded_sequence[i:i+n])
            logp = np.log2(self.ngram_prob(ngram))
            total_logp += max(NEG_INFINITY, logp)
        return total_logp

    def ngram_prob(self, ngram):
        """Return the smoothed probability of an n-gram with interpolation smoothing.
        
        Hint: Call ngram_prob on each vanilla n-gram model in self.sub_models!
        """
        # TODO: Your code here
        # initialize probability as 0.0
        prob = 0.0

        # reverse sub_model index to start from trigram
        for i, sub_model in enumerate(self.sub_models[::-1]):
            # add model ngram_prob with appropriate ngram, we need to parse ngrams for given sub models 
            # and add up the probabilities multiplied by lambdas
            prob += self.lambdas[len(self.sub_models)-1-i] * sub_model.ngram_prob(ngram[i:])

        # return probability
        return prob
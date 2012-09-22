""" 
@author Zongheng Yang
@email  zhyang.sms@gmail.com
@date   Apr 14, 2012

    Goal: compute the score of a sentence. 

    score = logP(w1|<s>) + logP(w2|w1) + ...
    where
            P(w_i | w_(i-1)) = (c(w_i | w_(i-1)) + 1) / (c(w_(i-1)) + V)
            V = size of vocabulary

    Laplace Bigram Language Model: 
    correct: 64 total: 471 accuracy: 0.135881
"""

import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.v = 0                                    
    self.counts = collections.defaultdict(lambda: 0)    
    self.probs = collections.defaultdict(lambda: 0.0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  

    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            if not self.counts[token]:
                self.v += 1
            self.counts[token] += 1
            if last_token:
                self.counts[(last_token, token)] += 1
            last_token = token

    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            if last_token:
                self.probs[(last_token, token)] = 1.0 * (self.counts[(last_token, token)] + 1) / (self.counts[last_token] + self.v)
            last_token = token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    last_token = None
    for token in sentence:
        if not last_token:
            last_token = token
            continue
        if (last_token, token) in self.counts:
            score += math.log(self.probs[(last_token, token)])
        else:
            score += math.log(1.0 / (self.counts[last_token] + self.v))
        last_token = token
    return score

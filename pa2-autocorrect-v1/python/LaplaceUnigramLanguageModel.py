""" 
@author Zongheng Yang
@email  zhyang.sms@gmail.com
@date   Apr 13, 2012

    Goal: compute the score of a sentence. 

    score = logP(w1) + logP(w2) + ...
    where
            P(w) = (c(w) + 1) / (# of tokens * 2)
"""

import math, collections

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.ntokens = 0                                    
    self.counts = collections.defaultdict(lambda: 0)    
    self.probs = collections.defaultdict(lambda: 0.0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        for datum in sentence.data:
            token = datum.word
            self.counts[token] += 1
            self.ntokens += 1

    for sentence in corpus.corpus:
        for datum in sentence.data:
            token = datum.word
            # seems that the denominator should be 
            #   self.ntokens + self.V
            self.probs[token] = 1.0 * (self.counts[token] + 1) / (self.ntokens * 2)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for token in sentence:
        if token in self.counts:
            score += math.log(self.probs[token])
        else:
            # if out-of-vocab word, then treat it as 0-freq and smooth
            score += math.log(1.0 / (self.ntokens * 2))
    return score


""" 
@author Zongheng Yang
@email  zhyang.sms@gmail.com
@date   Apr 17, 2012

    Goal: compute the score of a sentence. 

    score = logS(w1|<s>) + logS(w2|w1) + ...
    where
            S(w_i | w_(i-1)) = c(w_i | w_(i-1)) / c(w_(i-1)) 
            or               = 0.4 * S(w_(i-1))
            where
                    S(w_i) = (c(w_i) + 1) / (# of tokens * 2)

Stupid Backoff Language Model: 
correct: 86 total: 471 accuracy: 0.182590
"""

import math, collections

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.ntokens = 0
    self.counts = collections.defaultdict(lambda: 0)
    self.s = collections.defaultdict(lambda: 0.0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            self.ntokens += 1
            self.counts[token] += 1
            if last_token:
                self.counts[(last_token, token)] += 1
            last_token = token

    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            if last_token:
                tup = (last_token, token)
                if self.counts[tup]:
                    self.s[tup] = math.log(1.0 * self.counts[tup] / self.counts[last_token])
                else: # backing off
                    if self.s[token] == 0:
                        self.s[token] = math.log(1.0 * (self.counts[token] + 1) / (self.ntokens * 2))
                    self.s[tup] = math.log(0.4 * self.s[token])
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
        tup = (last_token, token)
        if tup in self.counts:
            score += self.s[tup]
        else: # stupid backoff to add-one smoothed unigram
            if self.s[token]: score += self.s[token]
            else: score += math.log(1.0 * (self.counts[token] + 1) / (self.ntokens * 2))
        last_token = token
    return score

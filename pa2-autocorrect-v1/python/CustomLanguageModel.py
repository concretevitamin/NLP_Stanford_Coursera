""" 
@author Zongheng Yang
@email  zhyang.sms@gmail.com
@date   Apr 22, 2012

    Goal: compute the score of a sentence. 

    score = logP(w1|<s>) + logP(w2|w1) + ...

    where
            p(w_i | w_(i-1)) = max{c(w_i | w_(i-1)) - d, 0} / c(w_(i-1)) 
                               + l(w_(i-1))*p_cont(w_i)
            where

                p_cout(w_i) = |{w_(i - 1) | c(w_(i-1), w_i) > 0}| /
                                |{(w_(j-1), w_j) | c(w_(j-1), w_(j)) > 0}|

                l(w_(i-1))  = d * |{w | c(w_(i-1), w) > 0}| / c(w_(i-1))

                d           = discount constant

Model:
Interpolated Kneser-Ney Smoothing (bigram, absolute discounting)

Custom Language Model: (d = 0.5)
correct: 116 total: 471 accuracy: 0.246285
"""

import math, collections

class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.ntokens = 0
    self.d = 0.5
    self.counts = collections.defaultdict(lambda: 0)
    self.p = collections.defaultdict(lambda: 0.0)
    self.l = collections.defaultdict(lambda: 0.0)
    self.p_cont = collections.defaultdict(lambda: 0.0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    prev = collections.defaultdict(lambda: [])
    bigram = collections.defaultdict(lambda: 0)
    succ = collections.defaultdict(lambda: [])

    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            self.ntokens += 1
            self.counts[token] += 1
            if last_token:
                self.counts[(last_token, token)] += 1
                prev[token].append(last_token)
                bigram[(last_token, token)] = 1
                succ[last_token].append(token)
            last_token = token

    for sentence in corpus.corpus:
        last_token = None
        for datum in sentence.data:
            token = datum.word
            if not self.p_cont[token]:
                self.p_cont[token]= 1.0 * len(prev[token]) / len(bigram)
            if last_token:
                tup = (last_token, token)
                self.l[last_token] = 1.0 * self.d * len(succ[last_token]) / self.counts[last_token]
                self.p[tup] = 1.0 * max(self.counts[tup] - self.d, 0) / self.counts[last_token] + self.l[last_token] * self.p_cont[token]
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
        if tup in self.p:
            score += self.p[tup]
        else:
            if self.l[last_token] and self.p_cont[token]:
                score += self.l[last_token] * self.p_cont[token]
            else: # w_(i-1) or w_(i) does not appear in the training corpus
                score += math.log(1.0 / self.ntokens)
        last_token = token
    return score

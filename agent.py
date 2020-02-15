#!/usr/bin/env python3

from utils import *
from functools import lru_cache, partialmethod

class Agent:

    def __init__(self, level=0, error=1e-3):
        '''
        '''
        self.level = level
        self.error = error

    def speaker(self, target, lexicon, level=0):
        emission = []
        itoa = dict()
        for i, entry in enumerate(lexicon): # consider each possible lexical entry
            word = entry.word
            itoa[i] = word
            meaning = entry.meaning
            if target in meaning: # if the word for this lexical entry is the one we want
                emission += [log(1-self.error)] # if yes, it is likely to be used                
            else:
                emission[word] += [log(self.error)] # otherwise it is unlikely to be used 
        lnorm = normalize_logprobs(emission) # normalize these so they are true log probabilities  
        lnorm = {itoa[i]: lprob for i, lprob in enumerate(lnorm)}
        return logprobs_to_probs(lnorm)

    def listener(self):
        pass

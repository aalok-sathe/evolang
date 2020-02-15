#!/usr/bin/env python3

from math import log, log1p, exp
from scipy.special import logsumexp
import numpy as np
# from matplotlib import pyplot as plt


sortuple = lambda a: tuple(sorted(set(a)))


def log_subtract(x,y):
    '''
    subtracts the values corresponding to x, y, which are logarithmic values,
    and returns a log representation of the result
    '''
    return x + log1p(-exp(y-x))

def normalize_logprobs(logprobs):
    '''
    normalizes log probability(magnitude) values so their corresponding
    probabilities sum to ~1.0
    '''
    logtotal = logsumexp(logprobs) # calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) # normalise---subtracting in the log
                                           # domain is equivalent to dividing in
                                           # the normal domain
    return normedlogs


def logprobs_to_probs(logprobs):
    '''
    A little utility function that converts a list of log probabilities to normal
    probabilities 
    '''
    if type(logprobs) is list:
        return [exp(logp) for logp in logprobs]
    elif type(logprobs) is dict:
        return {word: exp(logp) for word, logp in logprobs.items()}
    else:
        raise TypeError('unknown type for logprobs', type(logprobs))



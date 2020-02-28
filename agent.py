#!/usr/bin/env python3

from utils import *
from language import LexicalEntry, Lexicon
from functools import lru_cache, partial
# import numpy as np
from random import choices, uniform

class Agent:
    agents = dict() # level -> instance

    def __init__(self, level=0, lexicon=Lexicon([LexicalEntry(None, 0)]),
                 error=1e-3, noise=lambda bound: uniform(0, 1)*bound):
        '''
        '''
        self.level = level
        self.error = error
        self.lexicon = lexicon
        self.noise = partial(noise, error if level == 0 else (error/level))

        if level > 0 and level-1 not in self.agents:
            self.agents[level-1] = self.__class__(level-1, lexicon, error)
        if level not in self.agents:
            self.agents[level] = self


    @lru_cache(maxsize=None)
    def speakdist(self, referent):
        '''
        takes in a target referent and (optionally) lexicon, and returns
        a probability distribution over the lexicon indicating what could
        be spoken
        '''
        lexicon = self.lexicon
        noise = self.noise

        emission = [] # we will store emission probabilities here
        itoa = dict() # assign an index to each word for later retrieval
        # consider each possible lexical entry
        for i, entry in enumerate(lexicon):
            itoa[i] = entry.word
            meaning = entry.meaning

            if self.level == 0:
                if referent in meaning: # if the word for this lexical entry is the one we want
                    emission += [log((1-self.error) * noise())] # if yes, it is likely to be used
                else:
                    emission += [log(self.error * noise())] # otherwise it is (very) unlikely to be used
            else:
                listener = self.agents[self.level-1]
                dist = listener.listendist(entry.word)
                emission += [log(dist[referent] * noise())]

        lnorm = normalize_logprobs(emission) # normalize these so they are true log probabilities
        lnorm = {itoa[i]: lprob for i, lprob in enumerate(lnorm)}

        return logprobs_to_probs(lnorm)

    @lru_cache(maxsize=None)
    def listendist(self, word):
        '''
        takes in a word (hears it), and returns a probability distribution
        over referents based on what it thinks was likely meant by the
        speaker
        '''
        lexicon = self.lexicon

        itor = {i: r for i, r in enumerate(lexicon.possible_referents())}
        refprobs = []
        # consider each possible referent that they could have meant
        for i, r in sorted(itor.items()):
            entry = lexicon[word]
            if self.level == 0: # literal listener
                refprobs += [log((1-self.error)) if r in entry.meaning else log(self.error)]
            else:
                speaker = self.agents[self.level]
                dist = speaker.speakdist(r)
                refprobs += [log(dist[word])]

        lnorm = normalize_logprobs(refprobs)
        lnorm = {itor[i]: lprob for i, lprob in enumerate(lnorm)}

        return logprobs_to_probs(lnorm)

    def speak(self, ref):
        '''
        infers a referent upon hearing a word based on the distribution
        (non deterministic method)
        '''
        dist = self.speakdist(ref)
        itow = {i: w for i, w in enumerate(dist)}
        probs = [dist[w] for i, w in itow.items()]
        [i] = choices([*itow.keys()], weights=probs)
        return itow[i]

    def listen(self, word):
        '''
        infers a referent upon hearing a word based on the distribution
        (non deterministic method)
        '''
        dist = self.listendist(word)
        itor = {i: r for i, r in enumerate(dist)}
        probs = [dist[r] for i, r in itor.items()]
        [i] = choices([*itor.keys()], weights=probs)
        return itor[i]

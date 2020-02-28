#!/usr/bin/env python3

from utils import *
from language import LexicalEntry, Lexicon
from functools import lru_cache, partial

class Agent:
    agents = dict() # level -> instance

    def __init__(self, level=0, lexicon=Lexicon([LexicalEntry(None, 0)]),
                 error=1e-3):
        '''
        '''
        self.level = level
        self.error = error
        self.lexicon = lexicon

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
        emission = [] # we will store emission probabilities here
        itoa = dict() # assign an index to each word for later retrieval
        # consider each possible lexical entry
        for i, entry in enumerate(lexicon):
            itoa[i] = entry.word
            meaning = entry.meaning

            if self.level == 0:
                if referent in meaning: # if the word for this lexical entry is the one we want
                    emission += [log(1-self.error)] # if yes, it is likely to be used
                else:
                    emission += [log(self.error)] # otherwise it is (very) unlikely to be used
            else:
                listener = self.agents[self.level-1]
                print('DEBUG listener level', self.level-1, listener.level)
                dist = listener.listendist(entry.word)
                emission += [log(dist[referent])]

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
                refprobs += [log(1-self.error) if r in entry.meaning else log(self.error)]
            else:
                print('DEBUG I\'m pragmatic (listener)!', word)
                speaker = self.agents[self.level]
                dist = speaker.speakdist(r)
                refprobs += [log(dist[word])]

        lnorm = normalize_logprobs(refprobs)
        lnorm = {itor[i]: lprob for i, lprob in enumerate(lnorm)}

        return logprobs_to_probs(lnorm)

#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict

from utils import *
from language import Lexicon, LexicalEntry
from agent import Agent


def main():
    '''
    '''
    speakers = []
    listeners = []

    lexdict = [({0}, 'word1'), ({0, 1, 2}, 'word2')]

    entries = []
    for meanings, word in lexdict:
        e = LexicalEntry(word, meanings)
        entries += [e]
    lexicon = Lexicon(entries)

    print('Lexicon:')
    print([item for item in lexicon])
    print('-'*64)
    print()

    a0 = Agent(0, lexicon)
    tgt = 0
    print('Emission probabilities of s0 to speak target={}:'.format(tgt))
    print(a0.speakdist(tgt))

    # l0 = Agent(0, lexicon=lexicon)
    word = 'word2'
    print('Listener probabilities of l0 on hearing word={}:'.format(word))
    print(a0.listendist(word))

    print()
    print('Now, we define a level 1 agent')
    a1 = Agent(1, lexicon)

    print('Suppose pragmatic speaker 1 wants to convey target={}'.format(tgt))
    print(a1.speakdist(tgt), '\n')

    word = 'word1'
    print('And suppose pragmatic listener 1 heard word={}'.format(word))
    print(a1.listendist(word), '\n')

    outcomes = defaultdict(list)
    for trials in range(1000):
        for i in lexicon.possible_referents():
            outcomes[i] += ['meant: {}, inferred: {}'.format(i, a1.listen(a1.speak(i)))]
    for i in lexicon.possible_referents():
        print(Counter(outcomes[i]))

if __name__ == '__main__':
    parser = ArgumentParser()
    main()

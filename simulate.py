#!/usr/bin/env python3

from argparse import ArgumentParser

from utils import *
from language import *
from agent import *


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

    s0 = Agent(0)
    print('Emission probabilities of s0:')
    print(s0.speaker(0, lexicon))

if __name__ == '__main__':
    parser = ArgumentParser()
    main()

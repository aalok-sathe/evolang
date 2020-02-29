#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict

from utils import *
from language import Lexicon, LexicalEntry
from agent import Agent


def main():
    '''
    '''
    print('='*64)

    lexdict = [({0,1}, 'zeroone'), ({0, 2}, 'zerotwo'), ({1, 2}, 'onetwo')]

    entries = []
    for meanings, word in lexdict:
        e = LexicalEntry(word, meanings)
        entries += [e]
    lexicon = Lexicon(entries)

    print('Lexicon:')
    print([item for item in lexicon])
    print('='*32)
    print()

    for level in range(100):
        print('-'*64)
        print('Speaker level:\t{}'.format(level))
        s = Agent(level, lexicon)
        print('Listener level:\t{}'.format(level))
        l = Agent(level, lexicon)

        for tgt in lexicon.possible_referents():
            print('Speaker {} goal:\t{}'.format(level, tgt))
            print('\t', s.speakdist(tgt))

        print()
        for word in lexicon.possible_words():
            print('Listener {} hears:\t{}'.format(level, word))
            print('\t', l.listendist(word))

        # if level < 90: continue
        print()
        trials = 1000
        print('Some monte-carlo over {} runs'.format(trials))
        outcomes = defaultdict(list)
        for trial in range(trials):
            for r in lexicon.possible_referents():
                outcomes[r] += ['tgt: {}, act: {}'.format(r, l.listen(s.speak(r)))]
        for r in lexicon.possible_referents():
            print('\t', Counter(outcomes[r]))

    print('='*64)


if __name__ == '__main__':
    parser = ArgumentParser()
    main()

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

    lexdict = [({0, 1}, 's01'), ({0, 1, 2}, 's012'), ({1, 2, 3}, 's123'),
               ({1, 2}, 's12')]

    entries = []
    for meanings, word in lexdict:
        e = LexicalEntry(word, meanings)
        entries += [e]
    lexicon = Lexicon(entries)

    print('Lexicon:')
    print([item for item in lexicon])
    print('='*32)
    print()

    generations = 500
    for level in range(generations):
        print('-'*64)
        print('Speaker level:\t{}'.format(level))
        # s = Agent(level, lexicon, noise=lambda _: 1)
        s = Agent(level, lexicon)
        print('Listener level:\t{}'.format(level))
        l = Agent(level, lexicon)

        def fmtfloatdict(d):
            return '  '.join(['"{}" : {: >3.4f}'.format(k,v) for k,v in d.items()])

        for tgt in lexicon.possible_referents():
            print('Speaker {} goal:\t{}'.format(level, tgt))
            print('\t', fmtfloatdict(s.speakdist(tgt)))

        print()
        for word in lexicon.possible_words():
            print('Listener {} hears:\t{}'.format(level, word))
            print('\t', fmtfloatdict(l.listendist(word)))

        if level < generations-2: continue
        print()
        trials = 1000
        print('Monte-carlo over {} runs'.format(trials))
        outcomes = defaultdict(lambda: defaultdict(lambda: 0))
        for trial in range(trials):
            for r in lexicon.possible_referents():
                infr = l.listen(s.speak(r))
                outcomes[r][infr] += 1

        fmt = '{: >5} '*4
        print(' ', fmt.format(*lexicon.possible_referents()), sep='\t')
        print()
        for r in lexicon.possible_referents():
            for r_ in lexicon.possible_referents(): outcomes[r][r_]
            print(r, fmt.format(*[v for k, v in sorted(outcomes[r].items())]), sep='\t')

    print('='*64)


if __name__ == '__main__':
    parser = ArgumentParser()
    main()

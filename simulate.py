#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm
import random

from utils import *
from language import Lexicon, LexicalEntry
from agent import Agent


def main(args):
    '''
    '''
    print(args)
    np.random.seed(args.random_state)
    random.seed(args.random_state)

    print('='*64)

    # lexdict = [({0, 1}, 's01'), ({0, 1, 2}, 's012'), ({1, 2, 3}, 's123'),
    #            ({1, 2}, 's12')]
    lexdict = [({0, 1, 2}, 'A'),
               ({0, 1, 2}, 'B'),
               ({1, 2}, 'C')]

    entries = []
    for meanings, word in lexdict:
        e = LexicalEntry(word, meanings)
        entries += [e]
    lexicon = Lexicon(entries)

    print('Lexicon:')
    print([item for item in lexicon])
    print('='*32)
    print()

    generations = 1000
    for level in tqdm(range(generations)):
        s = Agent(level, lexicon)
        l = Agent(level, lexicon)

        if level < 996: continue

        print('-'*64)
        print('Speaker level:\t{}'.format(level))
        # s = Agent(level, lexicon, noise=lambda _: 1)
        print('Listener level:\t{}'.format(level))


        def fmtfloatdict(d):
            return '  '.join(['"{}" : {: >3.4f}'.format(k,v) for k,v in d.items()])


        for tgt in lexicon.possible_referents():
            print('Speaker {} goal:\t{}'.format(level, tgt))
            print('\t', fmtfloatdict(s.speakdist(tgt)))

        print()
        for word in lexicon.possible_words():
            print('Listener {} hears:\t{}'.format(level, word))
            print('\t', fmtfloatdict(l.listendist(word)))

        print()
        trials = 1000
        print('Monte-carlo over {} runs'.format(trials))
        outcomes = defaultdict(lambda: defaultdict(lambda: 0))
        for trial in range(trials):
            for r in lexicon.possible_referents():
                infr = l.listen(s.speak(r))
                outcomes[r][infr] += 1

        fmt = '{: >5} '*len([*lexicon.possible_referents()])
        print(' ', fmt.format(*lexicon.possible_referents()), sep='\t')
        print()
        for r in lexicon.possible_referents():
            for r_ in lexicon.possible_referents(): outcomes[r][r_]
            print(r, fmt.format(*[v for k, v in sorted(outcomes[r].items())]), sep='\t')

    print('='*64)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--random_state', default=None, help='seed', type=int)
    args = parser.parse_args()
    main(args)

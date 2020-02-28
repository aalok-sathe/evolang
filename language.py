#!/usr/bin/env python3

from typing import *
from utils import *
from collections import defaultdict

class Word(str): pass # = NewType('Word', str)
class Referent(int): pass # = NewType('Referent', int)
class Meaning(tuple): pass # = NewType('Meaning', Tuple[Referent])


class LexicalEntry:
    word = None
    meaning = None

    def __init__(self, w: Word, *meanings):
        ''' l = LexicalEntry('dax', 3,2,4,5,21)
            OR
            l = LexicalEntry('dax', (3,2,4,5,21))
        '''
        self.word = w
        if type(meanings[0]) in (Referent, int):
            self.meaning = sortuple(meanings)
        else:
            self.meaning = sortuple(meanings[0])

    def __str__(self):
        ''' str repr
        '''
        fmt = '{}: ({})'
        return fmt.format(self.word, ', '.join(map(str, self.meaning)))
    __repr__ = __str__

    def __hash__(self):
        ''' implement the hash function for a LexicalEntry object
        '''
        return hash((self.word, self.meaning))


class Lexicon:
    # lexicon = defaultdict(set)
    lexicon = None

    def __init__(self, lexical_entries):
        ''' initialize a lexicon optionally from a dictionary mapping
            word->meaning
        '''
        self.lexicon = dict()
        for entry in lexical_entries:
            # self.lexicon[entry.word].add(entry)
            self.lexicon[entry.word] = entry

    def add(self, entry):
        ''' add a new entry to the lexicon
        '''
        assert type(entry) is LexicalEntry
        # self.lexicon[entry.word].add(entry)
        self.lexicon[entry.word] = entry

    def remove(self, entry):
        ''' remove an entry from the lexicon
        '''
        assert type(entry) is LexicalEntry
        # self.lexicon[entry.word].discard(entry)
        self.lexicon[entry.word] = entry

    def __getitem__(self, word):
        '''
        '''
        try:
            return self.lexicon[word]
        except KeyError:
            return LexicalEntry(None, 0)

    def __iter__(self):
        ''' iterator over LexicalEntry objects
        '''
        for word in self.lexicon:
            yield self.lexicon[word]

    def __len__(self):
        ''''''
        return len(self.lexicon)

    def __str__(self):
        ''' string representation of the lexicon
        '''
        return '\n'.join(str(e) for e in self)

    def possible_words(self):
        ''' return a generator over all the words we know of
        '''
        seen = set()
        for word in self.lexicon:
            if word not in seen:
                seen.add(word)
                yield word
            else:
                continue

    def possible_referents(self):
        ''' returns a generator over all the meanings (referents) we know of
        '''
        seen = set()
        for word in self.lexicon:
            for m in self.lexicon[word].meaning:
                if m not in seen:
                    seen.add(m)
                    yield m
                else:
                    continue

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project

"""
Utilities for lexical analysis of text e.g. split a text in a list of tokens and
recognize each token type or meaning.

See https://en.wikipedia.org/wiki/Lexical_analysis

Tokens are kept in a lightweight Token class with a label. A "token label" is an
uppercase string that specifies some property of a string, such as its part of
speech or whether it is a keyword, literal or variable.
"""

# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
#
# The Natural Language Toolkit (NLTK) is an open source Python library
# for Natural Language Processing.  A free online book is available.
# (If you use the library for academic research, please cite the book.)
#
# Steven Bird, Ewan Klein, and Edward Loper (2009).
# Natural Language Processing with Python.  O'Reilly Media Inc.
# http://nltk.org/book

import re

from pygmars import Token


class Lexer:
    """
    Regular Expression Lexer

    The Lexer assigns a token lebel to tokens by comparing their strings
    to a series of regular expressions. For example, the following lexer uses
    word suffixes to make guesses about the part of speech tag to use as a token
    label:

    >>> from pygmars.lex import Lexer
    >>> words = '''The Fulton County Grand Jury said Friday an investigation
    ... of Atlanta's recent primary election produced `` no evidence '' that
    ... any irregularities took place .'''.split()
    >>> regexp_lexer = Lexer(
    ...     [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    ...      (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    ...      (r'.*able$', 'JJ'),                # adjectives
    ...      (r'.*ness$', 'NN'),                # nouns formed from adjectives
    ...      (r'.*ly$', 'RB'),                  # adverbs
    ...      (r'.*s$', 'NNS'),                  # plural nouns
    ...      (r'.*ing$', 'VBG'),                # gerunds
    ...      (r'.*ed$', 'VBD'),                 # past tense verbs
    ...      (r'.*', 'NN')                      # nouns (default)
    ... ])
    >>> regexp_lexer
    <Lexer: size=9>
    >>> results = regexp_lexer.lex_strings(words)
    >>> expected = [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'),
    ... ('Grand', 'NN'), ('Jury', 'NN'), ('said', 'NN'), ('Friday', 'NN'),
    ... ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
    ... ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'),
    ... ('election', 'NN'), ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'),
    ... ('evidence', 'NN'), ("''", 'NN'), ('that', 'NN'), ('any', 'NN'),
    ... ('irregularities', 'NNS'), ('took', 'NN'), ('place', 'NN'), ('.', 'NN')]
    >>> results = [(t.value, t.label) for t in results]
    >>> assert results == expected
    """

    def __init__(self, regexps):
        """
        Initialize a Lexer from a list of ``(regexp, label)`` tuples that
        indicates that a Token with a value matching ``regexp`` should be
        assigned a label of ``label``.  The regexp are evaluated in sequence and
        the first match is returned.
        """
        try:
            self._regexps = [
                (re.compile(regexp).match, label, regexp)
                for regexp, label in regexps
            ]

        except Exception as e:
            raise Exception(
                'Invalid Lexer regexp:', str(e),
                'regexp:', repr(regexp), 'label:', repr(label)
            ) from e

    def tokenize(self, string, splitter=str.split):
        """
        Return an iterable of pygmars.Tokens given a ``string`` split with the
        ``splitter`` function.
        """
        for ln, line in enumerate(string.splitlines(False), 1):
            for pos, value in enumerate(splitter(line)):
                yield Token(value, pos=pos, start_line=ln)

    def lex_string(self, string):
        """
        Return an iterable of pygmars.Tokens given a ``string``. Assign a
        "label" to every token whose value is matched by one of rules of this
        lexer.
        """
        return self.lex_tokens(self.tokenize(string))

    def lex_strings(self, strings):
        """
        Return an iterable of pygmars.Tokens given a ``strings`` iterable of
        strings. Assign a "label" to every token whose value is matched by one
        of rules of this lexer.
        """
        tokens = (Token(val, pos=pos) for pos, val in enumerate(strings))
        return self.lex_tokens(tokens)

    def lex_tokens(self, tokens):
        """
        Return an iterable of pygmars.Token given a ``tokens`` Token iterable.
        Assign a "label" to every token whose value is matched by one of regexp
        rules of this lexer.
        """
        regexps = self._regexps
        for token in tokens:
            for regexp, label, _pattern in regexps:
                if regexp(token.value):
                    # print(f'matched {token} with {_pattern!r}, {label}')
                    token.label = label
                    break
            yield token

    def __repr__(self):
        return f"<Lexer: size={len(self._regexps)}>"
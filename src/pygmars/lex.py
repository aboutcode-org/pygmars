# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>
#
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
#

"""
Utilities for lexing list of tokens by assigning a token type to a string.

A "token type" is a case-sensitive string that specifies some property of a
string, such as its part of speech or whether it is a keyword, literal or
variable. Tokens are encoded as tuples ``(string, token type)``.  For example,
the following combines the word ``'class'`` with token type (``'KEYWORD'``):

    >>> token = ('class', 'KEYWORD')
"""

import re


class RegexpLexer:
    """
    Regular Expression Lexer

    The RegexpLexer assigns a token type to tokens by comparing their
    strings to a series of regular expressions.  The following lexer
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag as a token type:

    >>> from pygmars.lex import RegexpLexer
    >>> sent = '''The Fulton County Grand Jury said Friday an investigation
    ... of Atlanta's recent primary election produced `` no evidence '' that
    ... any irregularities took place .'''.split()
    >>> regexp_lexer = RegexpLexer(
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
    <Regexp Lexer: size=9>
    >>> result = regexp_lexer.lex(sent)
    >>> expected = [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'),
    ... ('Grand', 'NN'), ('Jury', 'NN'), ('said', 'NN'), ('Friday', 'NN'),
    ... ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
    ... ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'),
    ... ('election', 'NN'), ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'),
    ... ('evidence', 'NN'), ("''", 'NN'), ('that', 'NN'), ('any', 'NN'),
    ... ('irregularities', 'NNS'), ('took', 'NN'), ('place', 'NN'), ('.', 'NN')]
    >>> assert result == expected

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, token_type)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be assigned a ``token_type``.  The pairs will be evalutated in
        order.
    """

    def __init__(self, regexps):
        try:
            self._regexps = [
                (re.compile(regexp).match, token_type,)
                for regexp, token_type in regexps
            ]

        except Exception as e:
            raise Exception(
                'Invalid RegexpLexer regexp:', str(e),
                'regexp:', regexp, 'token_type:', token_type
            ) from e

    def lex(self, tokens):
        """
        Given a ``tokens`` sequence of string, return a sequence of
        (token, token_type) tuples, assigning a token_type to every token
        that is matched. If a token is not recognized, it is  assigned a None
        token type.

        :rtype: list(tuple(str, str))
        """
        lexed_tokens = []
        lexed_tokens_append = lexed_tokens.append
        regexps = self._regexps
        for token in tokens:
            recognized = False
            for regexp, token_type in regexps:
                if regexp(token):
                    lexed_tokens_append((token, token_type,))
                    recognized = True
                    break
            if not recognized:
                lexed_tokens_append((token, None,))
        return lexed_tokens

    def __repr__(self):
        return f"<Regexp Lexer: size={len(self._regexps)}>"


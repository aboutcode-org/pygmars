# Natural Language Toolkit: Sequential Backoff Taggers
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes for tagging sentences sequentially, left to right.
"""

import re


class RegexpTagger:
    """
    Regular Expression Tagger

    The RegexpTagger assigns tags to tokens by comparing their
    word strings to a series of regular expressions.  The following tagger
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag:

    >>> from nltk.tag import RegexpTagger
    >>> sent = '''The Fulton County Grand Jury said Friday an investigation
    ... of Atlanta's recent primary election produced `` no evidence '' that
    ... any irregularities took place .'''.split()
    >>> regexp_tagger = RegexpTagger(
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
    >>> regexp_tagger
    <Regexp Tagger: size=9>
    >>> result = regexp_tagger.tag(sent)
    >>> expected = [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'), ('Grand', 'NN'), ('Jury', 'NN'),
    ... ('said', 'NN'), ('Friday', 'NN'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
    ... ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'), ('election', 'NN'),
    ... ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'), ('evidence', 'NN'), ("''", 'NN'),
    ... ('that', 'NN'), ('any', 'NN'), ('irregularities', 'NNS'), ('took', 'NN'),
    ... ('place', 'NN'), ('.', 'NN')]
    >>> assert result == expected

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, tag)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be tagged with ``tag``.  The pairs will be evalutated in
        order.  If none of the regexps match a word, then the
        optional backoff tagger is invoked, else it is
        assigned the tag None.
    """

    def __init__(self, regexps):
        try:
            self._regexps = [
                (re.compile(regexp).match, tag,) 
                for regexp, tag in regexps
            ]

        except Exception as e:
            raise Exception(
                'Invalid RegexpTagger regexp:', str(e), 'regexp:', regexp, 'tag:', tag
            ) from e

    def tag(self, tokens):
        """
        Determine the most appropriate tag sequence for the given
        token sequence, and return a corresponding list of tagged
        tokens.  A tagged token is encoded as a tuple ``(token, tag)``.

        :rtype: list(tuple(str, str))
        """
        tags = []
        for token in tokens:
            for regexp, tag in self._regexps:
                if regexp(token):
                    tags.append((token, tag,))
                    break
        return tags

    def __repr__(self):
        return f"<Regexp Tagger: size={len(self._regexps)}>"


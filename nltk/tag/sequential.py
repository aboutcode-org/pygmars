# Natural Language Toolkit: Sequential Backoff Taggers
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes for tagging sentences sequentially, left to right.  The
abstract base class SequentialBackoffTagger serves as the base
class for all the taggers in this module.  Tagging of individual words
is performed by the method ``choose_tag()``, which is defined by
subclasses of SequentialBackoffTagger.  If a tagger is unable to
determine a tag for the specified token, then its backoff tagger is
consulted instead.  Any SequentialBackoffTagger may serve as a
backoff tagger for any other SequentialBackoffTagger.
"""
from abc import abstractmethod

import re



######################################################################
# Abstract Base Classes
######################################################################
class SequentialBackoffTagger:
    """
    An abstract base class for taggers that tags words sequentially,
    left to right.  Tagging of individual words is performed by the
    ``choose_tag()`` method, which should be defined by subclasses.  If
    a tagger is unable to determine a tag for the specified token,
    then its backoff tagger is consulted.

    :ivar _taggers: A list of all the taggers that should be tried to
        tag a token (i.e., self and its backoff taggers).
    """

    def __init__(self, backoff=None):
        if backoff is None:
            self._taggers = [self]
        else:
            self._taggers = [self] + backoff._taggers

    @property
    def backoff(self):
        """The backoff tagger for this tagger."""
        return self._taggers[1] if len(self._taggers) > 1 else None

    def tag(self, tokens):
        # docs inherited from TaggerI
        tags = []
        for i in range(len(tokens)):
            tags.append(self.tag_one(tokens, i, tags))
        return list(zip(tokens, tags))

    def tag_one(self, tokens, index, history):
        """
        Determine an appropriate tag for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, then its backoff tagger is consulted.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """
        tag = None
        for tagger in self._taggers:
            tag = tagger.choose_tag(tokens, index, history)
            if tag is not None:
                break
        return tag

    @abstractmethod
    def choose_tag(self, tokens, index, history):
        """
        Decide which tag should be used for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, return None -- do not consult
        the backoff tagger.  This method should be overridden by
        subclasses of SequentialBackoffTagger.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """

######################################################################
# Tagger Classes
######################################################################


class RegexpTagger(SequentialBackoffTagger):
    """
    Regular Expression Tagger

    The RegexpTagger assigns tags to tokens by comparing their
    word strings to a series of regular expressions.  The following tagger
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag:

        >>> from nltk.corpus import brown
        >>> from nltk.tag import RegexpTagger
        >>> test_sent = brown.sents(categories='news')[0]
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
        >>> regexp_tagger.tag(test_sent)
        [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'), ('Grand', 'NN'), ('Jury', 'NN'),
        ('said', 'NN'), ('Friday', 'NN'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
        ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'), ('election', 'NN'),
        ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'), ('evidence', 'NN'), ("''", 'NN'),
        ('that', 'NN'), ('any', 'NN'), ('irregularities', 'NNS'), ('took', 'NN'),
        ('place', 'NN'), ('.', 'NN')]

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, tag)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be tagged with ``tag``.  The pairs will be evalutated in
        order.  If none of the regexps match a word, then the
        optional backoff tagger is invoked, else it is
        assigned the tag None.
    """

    json_tag = "nltk.tag.sequential.RegexpTagger"

    def __init__(self, regexps, backoff=None):
        """
        """
        super().__init__(backoff)
        try:
            self._regexps = [(re.compile(regexp), tag,) for regexp, tag in regexps]
        except Exception as e:
            raise Exception(
                'Invalid RegexpTagger regexp:', str(e), 'regexp:', regexp, 'tag:', tag
            ) from e

    def encode_json_obj(self):
        return [(regexp.pattern, tag) for regexp, tag in self._regexps], self.backoff

    @classmethod
    def decode_json_obj(cls, obj):
        regexps, backoff = obj
        return cls(regexps, backoff)

    def choose_tag(self, tokens, index, history):
        for regexp, tag in self._regexps:
            if re.match(regexp, tokens[index]):
                return tag
        return None

    def __repr__(self):
        return "<Regexp Tagger: size={}>".format(len(self._regexps))


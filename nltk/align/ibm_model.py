# -*- coding: utf-8 -*-
# Natural Language Toolkit: IBM Model Core
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Common methods and classes for all IBM models. See ``IBMModel1``,
``IBMModel2``, and ``IBMModel3`` for specific implementations.

The IBM models are a series of generative models that learn lexical
translation probabilities, p(target language word|source language word),
given a sentence-aligned parallel corpus.

The models increase in sophistication from model 1 to 5. Typically, the
output of lower models is used to seed the higher models. All models
use the Expectation-Maximization (EM) algorithm to learn various
probability tables.

Words in a sentence are one-indexed. The first word of a sentence has
position 1, not 0. Index 0 is reserved in the source sentence for the
NULL token. The concept of position does not apply to NULL, but it is
indexed at 0 by convention.

Each target word is aligned to exactly one source word or the NULL
token.

References:
Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.

Peter E Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and
Robert L. Mercer. 1993. The Mathematics of Statistical Machine
Translation: Parameter Estimation. Computational Linguistics, 19 (2),
263-311.
"""

from collections import defaultdict


class IBMModel(object):
    """
    Abstract base class for all IBM models
    """

    # Avoid division by zero and precision errors by imposing a minimum
    # value for probabilities. Note that this approach is theoretically
    # incorrect, since it may create probabilities that sum to more
    # than 1. In practice, the contribution of probabilities with MIN_PROB
    # is tiny enough that the value of MIN_PROB can be treated as zero.
    MIN_PROB = 1.0e-12 # GIZA++ is more liberal and uses 1.0e-7

    def __init__(self, sentence_aligned_corpus):
        self.init_vocab(sentence_aligned_corpus)

        self.translation_table = defaultdict(
            lambda: defaultdict(lambda: IBMModel.MIN_PROB))
        """
        dict[str][str]: float. Probability(target word | source word).
        Values accessed as ``translation_table[target_word][source_word]``.
        """

        self.alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: IBMModel.MIN_PROB))))
        """
        dict[int][int][int][int]: float. Probability(i | j,l,m).
        Values accessed as ``alignment_table[i][j][l][m]``.
        Used in model 2.
        """

        self.distortion_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: self.MIN_PROB))))
        """
        dict[int][int][int][int]: float. Probability(j | i,l,m).
        Values accessed as ``distortion_table[j][i][l][m]``.
        Used in model 3 and higher.
        """

        self.fertility_table = defaultdict(
            lambda: defaultdict(lambda: self.MIN_PROB))
        """
        dict[int][str]: float. Probability(fertility | source word).
        Values accessed as ``fertility_table[fertility][source_word]``.
        Used in model 3 and higher.
        """

        # Initial probability of null insertion
        self.p1 = 0.5
        """
        Probability that a generated word requires another target word
        that is aligned to NULL.
        Used in model 3 and higher.
        """

    def init_vocab(self, sentence_aligned_corpus):
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in sentence_aligned_corpus:
            trg_vocab.update(aligned_sentence.words)
            src_vocab.update(aligned_sentence.mots)
        # Add the NULL token
        src_vocab.add(None)

        self.src_vocab = src_vocab
        """
        set(str): All source language words used in training
        """

        self.trg_vocab = trg_vocab
        """
        set(str): All target language words used in training
        """


class AlignmentInfo(object):
    """
    Helper data object for training IBM Model 3

    Read-only. For a source sentence and its counterpart in the target
    language, this class holds information about the sentence pair's
    alignment and fertility.
    """

    def __init__(self, alignment, src_sentence, trg_sentence, fertility_of_i):
        if not isinstance(alignment, tuple):
            raise TypeError("The alignment must be a tuple because it is used "
                            "to uniquely identify AlignmentInfo objects.")

        self.alignment = alignment
        """
        tuple(int): Alignment function. ``alignment[j]`` is the position
        in the source sentence that is aligned to the position j in the
        target sentence.
        """

        self.fertility_of_i = fertility_of_i
        """
        tuple(int): Fertility of source word. ``fertility_of_i[i]`` is
        the number of words in the target sentence that is aligned to
        the word in position i of the source sentence.
        """

        self.src_sentence = src_sentence
        """
        tuple(str): Source sentence referred to by this object.
        Should include NULL token (None) in index 0.
        """

        self.trg_sentence = trg_sentence
        """
        tuple(str): Target sentence referred to by this object.
        """

    def __eq__(self, other):
        return self.alignment == other.alignment

    def __hash__(self):
        return hash(self.alignment)

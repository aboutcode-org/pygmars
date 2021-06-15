# -*- coding: utf-8 -*-
# Natural Language Toolkit: Taggers
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""
NLTK Taggers

This package contains classes and interfaces for part-of-speech
tagging, or simply "tagging".

A "tag" is a case-sensitive string that specifies some property of a token,
such as its part of speech.  Tagged tokens are encoded as tuples
``(tag, token)``.  For example, the following tagged token combines
the word ``'fly'`` with a noun part of speech tag (``'NN'``):

    >>> tagged_tok = ('fly', 'NN')
"""

from nltk.tag.sequential import RegexpTagger

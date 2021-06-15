# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>

"""
Regression Tests
~~~~~~~~~~~~~~~~
 
 
Regression Testing for NLTK issue #1025
===========================================
 
We want to ensure that a RegexpLexer can be created with more than 100 patterns
and does not fail with:
 "AssertionError: sorry, but this version only supports 100 named groups"
 
    >>> from pygmars.lex import RegexpLexer
    >>> patterns = [(str(i), 'NNP',) for i in range(200)]
    >>> tagger = RegexpLexer(patterns)

"""

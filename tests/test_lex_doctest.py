# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/pygmars for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.

# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>

"""
Regression Tests
~~~~~~~~~~~~~~~~


Regression Testing for NLTK issue #1025
===========================================

We want to ensure that a Lexer can be created with more than 100 patterns
and does not fail with:
 "AssertionError: sorry, but this version only supports 100 named groups"

    >>> from pygmars.lex import Lexer
    >>> patterns = [(str(i), 'NNP',) for i in range(200)]
    >>> tagger = Lexer(patterns)

"""

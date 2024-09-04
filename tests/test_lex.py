# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/aboutcode-org/pygmars for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.

# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>

import unittest

from pygmars.lex import InvalidLexerMatcher
from pygmars.lex import Lexer


class TestLexer(unittest.TestCase):
    def test_Lexer_simple(self):
        def callable_matcher(s):
            return False

        matchers = [
            (r"^Copyright\.txt$", "NN"),
            (callable_matcher, "FOO"),
        ]
        Lexer(matchers)

    def test_Lexer_fails_on_regex_error(self):
        def callable_matcher(s):
            return False

        matchers = [
            (r"^C(opyright\.txt$[(", "NN"),
        ]
        try:
            Lexer(matchers)
        except InvalidLexerMatcher:
            pass

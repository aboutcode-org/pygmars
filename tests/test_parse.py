# -*- coding: utf-8 -*-
# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>

import unittest

from pygmars import Token
from pygmars.parse import Parser


class TestRule(unittest.TestCase):

    def test_can_use_label_patterns_with_quantifiers(self):
        """
        Test for bug https://github.com/nltk/nltk/issues/1597

        Ensures that curly bracket quantifiers as in {4,} can be used inside a
        rule. This type of quantifier has been used for the supplementary
        example in http://www.nltk.org/book/ch07.html#exploring-text-corpora.
        """
        value_labels = [
            ('The', 'AT'),
            ('September-October', 'NP'),
            ('term', 'NN'),
            ('jury', 'NN'),
            ('had', 'HVD'),
            ('been', 'BEN'),
            ('charged', 'VBN'),
            ('by', 'IN'),
            ('Fulton', 'NP-TL'),
            ('Superior', 'JJ-TL'),
            ('Court', 'NN-TL'),
            ('Judge', 'NN-TL'),
            ('Durwood', 'NP'),
            ('Pye', 'NP'),
            ('to', 'TO'),
            ('investigate', 'VB'),
            ('reports', 'NNS'),
            ('of', 'IN'),
            ('possible', 'JJ'),
            ('``', 'BACKTICK'),
            ('irregularities', 'NNS'),
            ("''", "QUOTE"),
            ('in', 'IN'),
            ('the', 'AT'),
            ('hard-fought', 'JJ'),
            ('primary', 'NN'),
            ('which', 'WDT'),
            ('was', 'BEDZ'),
            ('won', 'VBN'),
            ('by', 'IN'),
            ('Mayor-nominate', 'NN-TL'),
            ('Ivan', 'NP'),
            ('Allen', 'NP'),
            ('Jr.', 'NP'),
            ('.', 'DOT'),
        ]
        tokens = list(Token.from_value_label_tuples(value_labels))
        cp = Parser('GROUP: <N.*>{4,}')
        tree = cp.parse(tokens)
        expected = """(ROOT
  The/AT
  September-October/NP
  term/NN
  jury/NN
  had/HVD
  been/BEN
  charged/VBN
  by/IN
  Fulton/NP-TL
  Superior/JJ-TL
  (GROUP Court/NN-TL Judge/NN-TL Durwood/NP Pye/NP)
  to/TO
  investigate/VB
  reports/NNS
  of/IN
  possible/JJ
  ``/BACKTICK
  irregularities/NNS
  ''/QUOTE
  in/IN
  the/AT
  hard-fought/JJ
  primary/NN
  which/WDT
  was/BEDZ
  won/VBN
  by/IN
  (GROUP Mayor-nominate/NN-TL Ivan/NP Allen/NP Jr./NP)
  ./DOT)"""

        assert tree.pformat() == expected

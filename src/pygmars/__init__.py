# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others

import re
from collections import namedtuple


class Token:
    """
    Represent a word token with its label, line number and line position.
    """
    #  keep memory requirements low by preventing creation of instance dicts.
    __slots__ = (
        # the string value of a token
        'value',
        # a token label as a string.
        # this is converted to an UPPER-CASE, dash-seprated string on creation
        'label',
        # starting line number in the original text, one-based
        'start_line',
        # the positition of this token; typically a token pos in its line, zero-
        # based, but can be an absolute position or an offset too
        'pos',
    )

    def __init__(self, value, label=None, start_line=None, pos=None):
        self.value = value
        self.label = as_token_label(label) if label else None
        self.start_line = start_line
        self.pos = pos

    def __repr__(self, *args, **kwargs):
        return f'Token({self.value!r}, {self.label!r}, {self.start_line}, {self.pos})'

    @classmethod
    def from_numbered_lines(cls, numbered_lines, splitter=str.split):
        """
        Return an iterable of tokens built from a ``numbered_lines`` iterable of
        tuples of (line number, text). Use the ``splitter`` callable to split
        a line in words/tokens. The line numbers are expected to be one-based.
        """
        for start_line, line in numbered_lines:
            for line_pos, value in enumerate(splitter(line)):
                yield cls(value, label=None, start_line=start_line, pos=line_pos)

    @classmethod
    def from_lines(cls, lines, splitter=str.split):
        """
        Return an iterable of tokens built from a ``lines`` iterable of strings
        Use the ``splitter`` callable to split a line in words/tokens.
        The line numbers are one-based.
        """
        numbered_lines = enumerate(lines, 1)
        return cls.from_numbered_lines(numbered_lines, splitter)

    @classmethod
    def from_string(cls, s, splitter=str.split):
        """
        Return an iterable of tokens built from a ``s`` string. Use the
        ``splitter`` callable to split a line in words/tokens. The line numbers
        are one-based.
        """
        lines = s.splitlines(False)
        numbered_lines = enumerate(lines, 1)
        return cls.from_numbered_lines(numbered_lines, splitter)

    @classmethod
    def from_value_label_tuples(cls, value_labels):
        """
        Return an iterable of tokens built from a ``value_labels`` iterable of
        tuples of token (value, label).
        """
        for pos, (value, label) in enumerate(value_labels):
                yield cls(value, label=label, pos=pos)

    def serialized(self):
        return f"Token({self.value}/{self.label})"


only_wordchars = re.compile(r'[^A-Z0-9\-]').sub


def as_token_label(s):
    """
    Return a string derived from `s` for use as a token label. Token labels are
    strings made only of uppercase ASCII letters, digits and dash separators.
    They do not start with a digit or dash and do not end with a dash.
    """
    s = str(s).upper()
    s = only_wordchars(' ', s)
    s = ' '.join(s.split())
    s = s.replace(' ', '-').replace('--', '-').strip('-').lstrip('0123456789')
    return s

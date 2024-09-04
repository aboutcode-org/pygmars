# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/aboutcode-org/pygmars for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
A hierarchical tree structure used to store parse trees.

Some example trees:

>>> from pygmars.tree import *
>>> dp1 = Tree('dp', [Tree('d', ['the']), Tree('np', ['dog'])])
>>> dp2 = Tree('dp', [Tree('d', ['the']), Tree('np', ['cat'])])
>>> vp = Tree('vp', [Tree('v', ['chased']), dp2])
>>> tree = Tree('s', [dp1, vp])
>>> print(tree)
(label='s', children=(
  (label='dp', children=(
    (label='d', children=(
      the))
    (label='np', children=(
      dog))))
  (label='vp', children=(
    (label='v', children=(
      chased))
    (label='dp', children=(
      (label='d', children=(
        the))
      (label='np', children=(
        cat))))))))

The node label is accessed using the `label` attribute:
>>> dp1.label, dp2.label, vp.label, tree.label
('dp', 'dp', 'vp', 's')

"""

# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Nathan Bodenstab <bodenstab@cslu.ogi.edu> (tree transforms)
#
# The Natural Language Toolkit (NLTK) is an open source Python library
# for Natural Language Processing.  A free online book is available.
# (If you use the library for academic research, please cite the book.)
#
# Steven Bird, Ewan Klein, and Edward Loper (2009).
# Natural Language Processing with Python.  O'Reilly Media Inc.
# http://nltk.org/book

import re
from typing import Generator


class Tree(list):
    """
    A Tree represents a hierarchical grouping of leaves and subtrees. For
    example, each constituent in a syntax tree is represented by a single Tree.

    A tree's children are encoded as a list of leaves and subtrees, where a leaf
    is a basic (non-tree) value; and a subtree is a nested Tree.

    For example::

    >>> from pygmars.tree import Tree
    >>> print(Tree(1, [2, Tree(3, [4]), 5]))
    <BLANKLINE>
      2
    <BLANKLINE>
        4
      5

    >>> vp = Tree('VP', [Tree('V', ['saw']), Tree('NP', ['him'])])
    >>> s = Tree('S', [Tree('NP', ['I']), vp])
    >>> print(s)
    (label='S', children=(
      (label='NP', children=(
        I))
      (label='VP', children=(
        (label='V', children=(
          saw))
        (label='NP', children=(
          him))))))

    >>> print(s[1])
    (label='VP', children=(
      (label='V', children=(
        saw))
      (label='NP', children=(
        him))))

    >>> print(s[1,1])
    (label='NP', children=(
      him))

    >>> t = Tree.from_string("(S (NP I) (VP (V saw) (NP him)))")
    >>> s == t
    True
    >>> print(t)
    (label='S', children=(
      (label='NP', children=(
        I))
      (label='VP', children=(
        (label='V', children=(
          saw))
        (label='NP', children=(
          him))))))

    >>> t[0], t[1,1] = t[1,1], t[0]
    >>> print(t)
    (label='S', children=(
      (label='NP', children=(
        him))
      (label='VP', children=(
        (label='V', children=(
          saw))
        (label='NP', children=(
          I))))))


    The length of a tree is the number of children it has.

    >>> len(t)
    2

    The label attribute allow individual constituents to be labeled.  For
    example, syntax trees use this label to specify phrase tags, such as "NP"
    and "VP".

    Construct a new tree:

    - ``Tree(label, children)`` constructs a new tree with the
        specified label and list of children.

    NOTE: Not all list method are implemented. In particular comparisons
    methods, deletion and many more.
    """

    def __init__(self, label, children):
        if not label:
            raise TypeError(f"Expected a label value: {label!r}")

        if isinstance(children, Generator):
            children = list(children)

        if not isinstance(children, list):
            raise TypeError(f"Children should be a list: {children!r}")

        list.__init__(self, children)
        self.label = label

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                return self
            elif len(index) == 1:
                return self[index[0]]
            else:
                return self[index[0]][index[1:]]
        else:
            raise TypeError(
                "%s indices must be integers, not %s" % (type(self).__name__, type(index).__name__)
            )

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            return list.__setitem__(self, index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError("The tree position () may not be " "assigned to.")
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError(
                "%s indices must be integers, not %s" % (type(self).__name__, type(index).__name__)
            )

    def leaves(self):
        """
        Return a list of leave nodes of this tree.

        >>> t = Tree.from_string("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.leaves()
        ['the', 'dog', 'chased', 'the', 'cat']

        :return: a list containing this tree's leaves.
            The order reflects the order of the
            leaves in the tree's hierarchical structure.
        :rtype: list
        """
        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves())
            else:
                leaves.append(child)
        return leaves

    @classmethod
    def from_string(
        cls,
        s,
        brackets=r"()",
        read_node=None,
        read_leaf=None,
        node_pattern=None,
        leaf_pattern=None,
        contains_spaces=re.compile(r"\s").search,
    ):
        """
        Read a bracketed tree string and return the resulting tree.
        Trees are represented as nested brackettings, such as::

          (S (NP (NNP John)) (VP (V runs)))

        :type s: str
        :param s: The string to read

        :type brackets: str (length=2)
        :param brackets: The bracket characters used to mark the
            beginning and end of trees and subtrees.

        :type read_node: function
        :type read_leaf: function
        :param read_node, read_leaf: If specified, these functions
            are applied to the substrings of ``s`` corresponding to
            nodes and leaves (respectively) to obtain the values for
            those nodes and leaves.  They should have the following
            signature:

               read_node(str) -> value

            Note that by default, node strings and leaf strings are
            delimited by whitespace and brackets; to override this
            default, use the ``node_pattern`` and ``leaf_pattern``
            arguments.

        :type node_pattern: str
        :type leaf_pattern: str
        :param node_pattern, leaf_pattern: Regular expression patterns
            used to find node and leaf substrings in ``s``.  By
            default, both nodes patterns are defined to match any
            sequence of non-whitespace non-bracket characters.

        :return: A tree corresponding to the string representation ``s``.
            If this class method is called using a subclass of Tree,
            then it will return a tree of that type.
        :rtype: Tree
        """
        if not isinstance(brackets, str) or len(brackets) != 2:
            raise TypeError("brackets must be a length-2 string")
        if contains_spaces(brackets):
            raise TypeError("whitespace brackets not allowed")
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = r"[^\s%s%s]+" % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = r"[^\s%s%s]+" % (open_pattern, close_pattern)
        token_re = re.compile(
            r"%s\s*(%s)?|%s|(%s)" % (open_pattern, node_pattern, close_pattern, leaf_pattern)
        )
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])]  # list of (node, children) tuples
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(s, match, "end-of-string")
                label = token[1:].lstrip()
                if read_node is not None:
                    label = read_node(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(s, match, open_b)
                    else:
                        cls._parse_error(s, match, "end-of-string")
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(s, match, open_b)
                if read_leaf is not None:
                    token = read_leaf(token)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, "end-of-string", close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, "end-of-string", open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        return tree

    @classmethod
    def _parse_error(cls, s, match, expecting):
        """
        Display a friendly error message when parsing a tree string fails.
        :param s: The string we're parsing.
        :param match: regexp match of the problem token.
        :param expecting: what we expected to see instead.
        """
        # Construct a basic error message
        if match == "end-of-string":
            pos, token = len(s), "end-of-string"
        else:
            pos, token = match.start(), match.group()
        msg = "%s.read(): expected %r but got %r\n%sat index %d." % (
            cls.__name__,
            expecting,
            token,
            " " * 12,
            pos,
        )
        # Add a display showing the error token itself:
        s = s.replace("\n", " ").replace("\t", " ")
        offset = pos
        if len(s) > pos + 10:
            s = s[: pos + 10] + "..."
        if pos > 10:
            s = "..." + s[pos - 10 :]
            offset = 13
        msg += '\n%s"%s"\n%s^' % (" " * 16, s, " " * (17 + offset))
        raise ValueError(msg)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.pformat()

    def pprint(self, **kwargs):
        """
        Print a string representation of this Tree
        """
        print(self.pformat(**kwargs))

    def pformat(self, indent=0, *args, **kwargs):
        """
        :return: A pretty-printed string representation of this self.
        :rtype: str
        :param indent: The indentation level at which printing
            begins.  This number is used to decide how far to indent
            subsequent lines.
        :type indent: int
        """
        closings = 0
        if isinstance(self.label, str):
            s = f"(label={self.label!r}, children=("
            closings = 2
        else:
            s = ""

        for child in self:
            if isinstance(child, Tree):
                s += "\n" + " " * (indent + 2) + child.pformat(indent=indent + 2)
            elif isinstance(child, tuple):
                s += "\n" + " " * (indent + 2) + "/".join(child)
            elif isinstance(child, str):
                s += "\n" + " " * (indent + 2) + f"{child}"
            else:
                s += "\n" + " " * (indent + 2) + repr(child)
        return f"{s}" + (")" * closings)

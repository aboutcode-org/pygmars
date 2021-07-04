# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project

"""
A hierarchical tree structure used to store parse trees.
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


class Tree(list):
    """
    A Tree represents a hierarchical grouping of leaves and subtrees. For
    example, each constituent in a syntax tree is represented by a single Tree.

    A tree's children are encoded as a list of leaves and subtrees, where a leaf
    is a basic (non-tree) value; and a subtree is a nested Tree.

        >>> from pygmars.tree import Tree
        >>> print(Tree(1, [2, Tree(3, [4]), 5]))
        (1,
         ('2',
          (3, ('4',)),
          '5'))
        >>> vp = Tree('VP', [Tree('V', ['saw']), Tree('NP', ['him'])])
        >>> s = Tree('S', [Tree('NP', ['I']), vp])
        >>> print(s)
        ('S',
         (('NP', ('I',)),
          ('VP',
           (('V', ('saw',)),
            ('NP',
             ('him',))))))

        >>> print(s[1])
        ('VP',
         (('V', ('saw',)),
          ('NP', ('him',))))

        >>> print(s[1,1])
        ('NP', ('him',))

        >>> t = Tree.from_string("(S (NP I) (VP (V saw) (NP him)))")
        >>> s == t
        True
        >>> print(t)
        ('S',
         (('NP', ('I',)),
          ('VP',
           (('V', ('saw',)),
            ('NP',
             ('him',))))))

        >>> t[0], t[1,1] = t[1,1], t[0]
        >>> print(t)
        ('S',
         (('NP', ('him',)),
          ('VP',
           (('V', ('saw',)),
            ('NP',
             ('I',))))))



    The length of a tree is the number of children it has.

        >>> len(t)
        2

    The label attribute allow individual constituents to be labeled.  For
    example, syntax trees use this label to specify phrase tags, such as "NP"
    and "VP".

    Several Tree methods use "tree positions" to specify children or descendants
    of a tree.  Tree positions are defined as follows:

      - The tree position *i* specifies a Tree's *i*\ th child.
      - The tree position ``()`` specifies the Tree itself.
      - If *p* is the tree position of descendant *d*, then
        *p+i* specifies the *i*\ th child of *d*.

    I.e., every tree position is either a single index *i*,
    specifying ``tree[i]``; or a sequence *i1, i2, ..., iN*,
    specifying ``tree[i1][i2]...[iN]``.

    Construct a new tree:

    - ``Tree(label, children)`` constructs a new tree with the
        specified label and list of children.
    """

    def __init__(self, label, children=None):
        if children is None:
            raise TypeError(
                f"{type(self).__name__}: Expected a label value and child list "
            )
        elif isinstance(children, str):
            raise TypeError(
                f"{type(self).__name__}() children should be a list, not a string"
            )

        list.__init__(self, children)
        self.label = label

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self.label == other.label
            and list(self) == list(other)
        )

    def __lt__(self, other):
        if not isinstance(other, Tree):
            # Sometimes children can be pure strings,
            # so we need to be able to compare with non-trees:
            return self.__class__.__name__ < other.__class__.__name__
        elif self.__class__ is other.__class__:
            return (self.label, list(self)) < (other._label, list(other))
        else:
            return self.__class__.__name__ < other.__class__.__name__

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ne__ = lambda self, other: not self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __le__ = lambda self, other: self < other or self == other
    __ge__ = lambda self, other: not self < other

    def __mul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __rmul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __add__(self, v):
        raise TypeError("Tree does not support addition")

    def __radd__(self, v):
        raise TypeError("Tree does not support addition")

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
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
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
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def __delitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__delitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError("The tree position () may not be deleted.")
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def leaves(self):
        """
        Return the leaves of the tree without their labels

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

    def flatten(self):
        """
        Return a flat version of the tree, with all non-root non-terminals
        removed. This is a tree consisting of the tree root connected directly
        to its leaves, omitting all intervening non-terminal nodes.

        >>> t = Tree.from_string("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> print(t.flatten())
        ('S',
         ('the',
          'dog',
          'chased',
          'the',
          'cat'))
        """
        return Tree(self.label, self.leaves())

    @classmethod
    def from_string(
        cls,
        s,
        brackets="()",
        read_node=None,
        read_leaf=None,
        node_pattern=None,
        leaf_pattern=None,
        remove_empty_top_bracketing=False,
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

            For example, these functions could be used to process nodes
            and leaves whose values should be some type other than
            string (such as ``FeatStruct``).
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

        :type remove_empty_top_bracketing: bool
        :param remove_empty_top_bracketing: If the resulting tree has
            an empty node label, and is length one, then return its
            single child instead.  This is useful for treebank trees,
            which sometimes contain an extra level of bracketing.

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
            r"%s\s*(%s)?|%s|(%s)"
            % (open_pattern, node_pattern, close_pattern, leaf_pattern)
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

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == "" and len(tree) == 1:
            tree = tree[0]
        # return the tree.
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
        # Add a display showing the error token itsels:
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
        from pprint import pformat
        return pformat(self.serialized(), width=20)

    def pprint(self, **kwargs):
        print(self.pformat(**kwargs))

    def pformat(self, **kwargs):
        return repr(self)

    def serialized(self):
        return self.label, tuple([
            x.serialized() if hasattr(x, "serialized") else str(x)
            for x in self
        ])

"""

    >>> from pygmars.tree import *

Some trees to run tests on:

    >>> dp1 = Tree('dp', [Tree('d', ['the']), Tree('np', ['dog'])])
    >>> dp2 = Tree('dp', [Tree('d', ['the']), Tree('np', ['cat'])])
    >>> vp = Tree('vp', [Tree('v', ['chased']), dp2])
    >>> tree = Tree('s', [dp1, vp])
    >>> print(tree)
    (s (dp (d the) (np dog)) (vp (v chased) (dp (d the) (np cat))))

The node label is accessed using the `label` attribute:

    >>> dp1.label, dp2.label, vp.label, tree.label
    ('dp', 'dp', 'vp', 's')

    >>> print(tree[1,1,1,0])
    cat

"""


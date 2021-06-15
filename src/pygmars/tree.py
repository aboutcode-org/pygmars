# -*- coding: utf-8 -*-
# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>
#
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Nathan Bodenstab <bodenstab@cslu.ogi.edu> (tree transforms)

"""
Class for representing hierarchical tree structures, such as syntax and parser trees.
"""

import re
from abc import ABCMeta
from abc import abstractmethod


class Tree(list):
    """
    A Tree represents a hierarchical grouping of leaves and subtrees.
    For example, each constituent in a syntax tree is represented by a single Tree.

    A tree's children are encoded as a list of leaves and subtrees,
    where a leaf is a basic (non-tree) value; and a subtree is a
    nested Tree.

        >>> from pygmars.tree import Tree
        >>> print(Tree(1, [2, Tree(3, [4]), 5]))
        (1 2 (3 4) 5)
        >>> vp = Tree('VP', [Tree('V', ['saw']),
        ...                  Tree('NP', ['him'])])
        >>> s = Tree('S', [Tree('NP', ['I']), vp])
        >>> print(s)
        (S (NP I) (VP (V saw) (NP him)))
        >>> print(s[1])
        (VP (V saw) (NP him))
        >>> print(s[1,1])
        (NP him)
        >>> t = Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
        >>> s == t
        True
        >>> t[1][1].set_label('X')
        >>> t[1][1].label()
        'X'
        >>> print(t)
        (S (NP I) (VP (V saw) (X him)))
        >>> t[0], t[1,1] = t[1,1], t[0]
        >>> print(t)
        (S (X him) (VP (V saw) (NP I)))

    The length of a tree is the number of children it has.

        >>> len(t)
        2

    The set_label() and label() methods allow individual constituents
    to be labeled.  For example, syntax trees use this label to specify
    phrase tags, such as "NP" and "VP".

    Several Tree methods use "tree positions" to specify
    children or descendants of a tree.  Tree positions are defined as
    follows:

      - The tree position *i* specifies a Tree's *i*\ th child.
      - The tree position ``()`` specifies the Tree itself.
      - If *p* is the tree position of descendant *d*, then
        *p+i* specifies the *i*\ th child of *d*.

    I.e., every tree position is either a single index *i*,
    specifying ``tree[i]``; or a sequence *i1, i2, ..., iN*,
    specifying ``tree[i1][i2]...[iN]``.

    Construct a new tree.  This constructor can be called in one
    of two ways:

    - ``Tree(label, children)`` constructs a new tree with the
        specified label and list of children.

    - ``Tree.fromstring(s)`` constructs a new tree by parsing the string ``s``.
    """

    def __init__(self, node, children=None):
        if children is None:
            raise TypeError(
                "%s: Expected a node value and child list " % type(self).__name__
            )
        elif isinstance(children, str):
            raise TypeError(
                "%s() argument 2 should be a list, not a "
                "string" % type(self).__name__
            )
        else:
            list.__init__(self, children)
            self._label = node

    # ////////////////////////////////////////////////////////////
    # Comparison operators
    # ////////////////////////////////////////////////////////////

    def __eq__(self, other):
        return self.__class__ is other.__class__ and (self._label, list(self)) == (
            other._label,
            list(other),
        )

    def __lt__(self, other):
        if not isinstance(other, Tree):
            # Sometimes children can be pure strings,
            # so we need to be able to compare with non-trees:
            return self.__class__.__name__ < other.__class__.__name__
        elif self.__class__ is other.__class__:
            return (self._label, list(self)) < (other._label, list(other))
        else:
            return self.__class__.__name__ < other.__class__.__name__

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ne__ = lambda self, other: not self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __le__ = lambda self, other: self < other or self == other
    __ge__ = lambda self, other: not self < other

    # ////////////////////////////////////////////////////////////
    # Disabled list operations
    # ////////////////////////////////////////////////////////////

    def __mul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __rmul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __add__(self, v):
        raise TypeError("Tree does not support addition")

    def __radd__(self, v):
        raise TypeError("Tree does not support addition")

    # ////////////////////////////////////////////////////////////
    # Indexing (with support for tree positions)
    # ////////////////////////////////////////////////////////////

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

    # ////////////////////////////////////////////////////////////
    # Basic tree operations
    # ////////////////////////////////////////////////////////////

    def _get_node(self):
        """Outdated method to access the node value; use the label() method instead."""
        raise NotImplementedError("Use label() to access a node label.")

    def _set_node(self, value):
        """Outdated method to set the node value; use the set_label() method instead."""
        raise NotImplementedError("Use set_label() method to set a node label.")

    node = property(_get_node, _set_node)

    def label(self):
        """
        Return the node label of the tree.

            >>> t = Tree.fromstring('(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))')
            >>> t.label()
            'S'

        :return: the node label (typically a string)
        :rtype: any
        """
        return self._label

    def set_label(self, label):
        """
        Set the node label of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.set_label("T")
            >>> print(t)
            (T (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))

        :param label: the node label (typically a string)
        :type label: any
        """
        self._label = label

    def leaves(self):
        """
        Return the leaves of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
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
        Return a flat version of the tree, with all non-root non-terminals removed.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> print(t.flatten())
            (S the dog chased the cat)

        :return: a tree consisting of this tree's root connected directly to
            its leaves, omitting all intervening non-terminal nodes.
        :rtype: Tree
        """
        return Tree(self.label(), self.leaves())

    def height(self):
        """
        Return the height of the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.height()
            5
            >>> print(t[0,0])
            (D the)
            >>> t[0,0].height()
            2

        :return: The height of this tree.  The height of a tree
            containing no children is 1; the height of a tree
            containing only leaves is 2; and the height of any other
            tree is one plus the maximum of its children's
            heights.
        :rtype: int
        """
        max_child_height = 0
        for child in self:
            if isinstance(child, Tree):
                max_child_height = max(max_child_height, child.height())
            else:
                max_child_height = max(max_child_height, 1)
        return 1 + max_child_height

    def treepositions(self, order="preorder"):
        """
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.treepositions() # doctest: +ELLIPSIS
            [(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
            >>> for pos in t.treepositions('leaves'):
            ...     t[pos] = t[pos][::-1].upper()
            >>> print(t)
            (S (NP (D EHT) (N GOD)) (VP (V DESAHC) (NP (D EHT) (N TAC))))

        :param order: One of: ``preorder``, ``postorder``, ``bothorder``,
            ``leaves``.
        """
        positions = []
        if order in ("preorder", "bothorder"):
            positions.append(())
        for i, child in enumerate(self):
            if isinstance(child, Tree):
                childpos = child.treepositions(order)
                positions.extend((i,) + p for p in childpos)
            else:
                positions.append((i,))
        if order in ("postorder", "bothorder"):
            positions.append(())
        return positions

    def subtrees(self, filter=None):
        """
        Generate all the subtrees of this tree, optionally restricted
        to trees matching the filter function.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> for s in t.subtrees(lambda t: t.height() == 2):
            ...     print(s)
            (D the)
            (N dog)
            (V chased)
            (D the)
            (N cat)

        :type filter: function
        :param filter: the function to filter all local trees
        """
        if not filter or filter(self):
            yield self
        for child in self:
            if isinstance(child, Tree):
                for subtree in child.subtrees(filter):
                    yield subtree

    def pos(self):
        """
        Return a sequence of pos-tagged words extracted from the tree.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.pos()
            [('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]

        :return: a list of tuples containing leaves and pre-terminals (part-of-speech tags).
            The order reflects the order of the leaves in the tree's hierarchical structure.
        :rtype: list(tuple)
        """
        pos = []
        for child in self:
            if isinstance(child, Tree):
                pos.extend(child.pos())
            else:
                pos.append((child, self._label))
        return pos

    def leaf_treeposition(self, index):
        """
        :return: The tree position of the ``index``-th leaf in this
            tree.  I.e., if ``tp=self.leaf_treeposition(i)``, then
            ``self[tp]==self.leaves()[i]``.

        :raise IndexError: If this tree contains fewer than ``index+1``
            leaves, or if ``index<0``.
        """
        if index < 0:
            raise IndexError("index must be non-negative")

        stack = [(self, ())]
        while stack:
            value, treepos = stack.pop()
            if not isinstance(value, Tree):
                if index == 0:
                    return treepos
                else:
                    index -= 1
            else:
                for i in range(len(value) - 1, -1, -1):
                    stack.append((value[i], treepos + (i,)))

        raise IndexError("index must be less than or equal to len(self)")

    def treeposition_spanning_leaves(self, start, end):
        """
        :return: The tree position of the lowest descendant of this
            tree that dominates ``self.leaves()[start:end]``.
        :raise ValueError: if ``end <= start``
        """
        if end <= start:
            raise ValueError("end must be greater than start")
        # Find the tree positions of the start & end leaves, and
        # take the longest common subsequence.
        start_treepos = self.leaf_treeposition(start)
        end_treepos = self.leaf_treeposition(end - 1)
        # Find the first index where they mismatch:
        for i in range(len(start_treepos)):
            if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
                return start_treepos[:i]
        return start_treepos

    # ////////////////////////////////////////////////////////////
    # Convert, copy
    # ////////////////////////////////////////////////////////////

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.

        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            return cls(tree._label, children)
        else:
            return tree

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self)
        else:
            return type(self).convert(self)

    def _frozen_class(self):
        return ImmutableTree

    def freeze(self, leaf_freezer=None):
        frozen_class = self._frozen_class()
        if leaf_freezer is None:
            newcopy = frozen_class.convert(self)
        else:
            newcopy = self.copy(deep=True)
            for pos in newcopy.treepositions("leaves"):
                newcopy[pos] = leaf_freezer(newcopy[pos])
            newcopy = frozen_class.convert(newcopy)
        hash(newcopy)  # Make sure the leaves are hashable.
        return newcopy

    # ////////////////////////////////////////////////////////////
    # Parsing
    # ////////////////////////////////////////////////////////////

    @classmethod
    def fromstring(
        cls,
        s,
        brackets="()",
        read_node=None,
        read_leaf=None,
        node_pattern=None,
        leaf_pattern=None,
        remove_empty_top_bracketing=False,
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
        if re.search("\s", brackets):
            raise TypeError("whitespace brackets not allowed")
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = "[^\s%s%s]+" % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = "[^\s%s%s]+" % (open_pattern, close_pattern)
        token_re = re.compile(
            "%s\s*(%s)?|%s|(%s)"
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

    # ////////////////////////////////////////////////////////////
    # Visualization & String Representation
    # ////////////////////////////////////////////////////////////

    def __repr__(self):
        childstr = ", ".join(repr(c) for c in self)
        return "%s(%s, [%s])" % (
            type(self).__name__,
            repr(self._label),
            childstr,
        )

    def __str__(self):
        return self.pformat()

    def pprint(self, **kwargs):
        """
        Print a string representation of this Tree
        """
        print(self.pformat(**kwargs))

    def pformat(self, margin=70, indent=0, nodesep="", parens="()", quotes=False):
        """
        :return: A pretty-printed string representation of this tree.
        :rtype: str
        :param margin: The right margin at which to do line-wrapping.
        :type margin: int
        :param indent: The indentation level at which printing
            begins.  This number is used to decide how far to indent
            subsequent lines.
        :type indent: int
        :param nodesep: A string that is used to separate the node
            from the children.  E.g., the default value ``':'`` gives
            trees like ``(S: (NP: I) (VP: (V: saw) (NP: it)))``.
        """

        # Try writing it on one line.
        s = self._pformat_flat(nodesep, parens, quotes)
        if len(s) + indent < margin:
            return s

        # If it doesn't fit on one line, then write it on multi-lines.
        if isinstance(self._label, str):
            s = "%s%s%s" % (parens[0], self._label, nodesep)
        else:
            s = "%s%s%s" % (parens[0], repr(self._label), nodesep)
        for child in self:
            if isinstance(child, Tree):
                s += (
                    "\n"
                    +" " * (indent + 2)
                    +child.pformat(margin, indent + 2, nodesep, parens, quotes)
                )
            elif isinstance(child, (tuple, list)):
                s += "\n" + " " * (indent + 2) + "/".join(child)
            elif isinstance(child, str) and not quotes:
                s += "\n" + " " * (indent + 2) + "%s" % child
            else:
                s += "\n" + " " * (indent + 2) + repr(child)
        return s + parens[1]

    def _pformat_flat(self, nodesep, parens, quotes):
        childstrs = []
        for child in self:
            if isinstance(child, Tree):
                childstrs.append(child._pformat_flat(nodesep, parens, quotes))
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, str) and not quotes:
                childstrs.append("%s" % child)
            else:
                childstrs.append(repr(child))
        if isinstance(self._label, str):
            return "%s%s%s %s%s" % (
                parens[0],
                self._label,
                nodesep,
                " ".join(childstrs),
                parens[1],
            )
        else:
            return "%s%s%s %s%s" % (
                parens[0],
                repr(self._label),
                nodesep,
                " ".join(childstrs),
                parens[1],
            )


class ImmutableTree(Tree):

    def __init__(self, node, children=None):
        super(ImmutableTree, self).__init__(node, children)
        # Precompute our hash value.  This ensures that we're really
        # immutable.  It also means we only have to calculate it once.
        try:
            self._hash = hash((self._label, tuple(self)))
        except (TypeError, ValueError) as e:
            raise ValueError(
                "%s: node value and children " "must be immutable" % type(self).__name__
            ) from e

    def __setitem__(self, index, value):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __setslice__(self, i, j, value):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __delitem__(self, index):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __delslice__(self, i, j):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __iadd__(self, other):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __imul__(self, other):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def append(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def extend(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def pop(self, v=None):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def remove(self, v):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def reverse(self):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def sort(self):
        raise ValueError("%s may not be modified" % type(self).__name__)

    def __hash__(self):
        return self._hash

    def set_label(self, value):
        """
        Set the node label.  This will only succeed the first time the
        node label is set, which should occur in ImmutableTree.__init__().
        """
        if hasattr(self, "_label"):
            raise ValueError("%s may not be modified" % type(self).__name__)
        self._label = value


class AbstractParentedTree(Tree, metaclass=ABCMeta):
    """
    An abstract base class for a ``Tree`` that automatically maintains
    pointers to parent nodes.  These parent pointers are updated
    whenever any change is made to a tree's structure.  Two subclasses
    are currently defined:

      - ``ParentedTree`` is used for tree structures where each subtree
        has at most one parent.  This class should be used in cases
        where there is no"sharing" of subtrees.

      - ``MultiParentedTree`` is used for tree structures where a
        subtree may have zero or more parents.  This class should be
        used in cases where subtrees may be shared.

    Subclassing
    ===========
    The ``AbstractParentedTree`` class redefines all operations that
    modify a tree's structure to call two methods, which are used by
    subclasses to update parent information:

      - ``_setparent()`` is called whenever a new child is added.
      - ``_delparent()`` is called whenever a child is removed.
    """

    def __init__(self, node, children=None):
        super(AbstractParentedTree, self).__init__(node, children)
        # If children is None, the tree is read from node, and
        # all parents will be set during parsing.
        if children is not None:
            # Otherwise we have to set the parent of the children.
            # Iterate over self, and *not* children, because children
            # might be an iterator.
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i, dry_run=True)
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i)

    # ////////////////////////////////////////////////////////////
    # Parent management
    # ////////////////////////////////////////////////////////////
    @abstractmethod
    def _setparent(self, child, index, dry_run=False):
        """
        Update the parent pointer of ``child`` to point to ``self``.  This
        method is only called if the type of ``child`` is ``Tree``;
        i.e., it is not called when adding a leaf to a tree.  This method
        is always called before the child is actually added to the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        :raise TypeError: If ``child`` is a tree with an impropriate
            type.  Typically, if ``child`` is a tree, then its type needs
            to match the type of ``self``.  This prevents mixing of
            different tree types (single-parented, multi-parented, and
            non-parented).
        :param dry_run: If true, the don't actually set the child's
            parent pointer; just check for any error conditions, and
            raise an exception if one is found.
        """

    @abstractmethod
    def _delparent(self, child, index):
        """
        Update the parent pointer of ``child`` to not point to self.  This
        method is only called if the type of ``child`` is ``Tree``; i.e., it
        is not called when removing a leaf from a tree.  This method
        is always called before the child is actually removed from the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        """

    # ////////////////////////////////////////////////////////////
    # Methods that add/remove children
    # ////////////////////////////////////////////////////////////
    # Every method that adds or removes a child must make
    # appropriate calls to _setparent() and _delparent().

    def __delitem__(self, index):
        # del ptree[start:stop]
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            # Clear all the children pointers.
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            # Delete the children from our child list.
            super(AbstractParentedTree, self).__delitem__(index)

        # del ptree[i]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index out of range")
            # Clear the child's parent pointer.
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            # Remove the child from our child list.
            super(AbstractParentedTree, self).__delitem__(index)

        elif isinstance(index, (list, tuple)):
            # del ptree[()]
            if len(index) == 0:
                raise IndexError("The tree position () may not be deleted.")
            # del ptree[(i,)]
            elif len(index) == 1:
                del self[index[0]]
            # del ptree[i1, i2, i3]
            else:
                del self[index[0]][index[1:]]

        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def __setitem__(self, index, value):
        # ptree[start:stop] = value
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            # make a copy of value, in case it's an iterator
            if not isinstance(value, (list, tuple)):
                value = list(value)
            # Check for any error conditions, so we can avoid ending
            # up in an inconsistent state if an error does occur.
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step, dry_run=True)
            # clear the child pointers of all parents we're removing
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            # set the child pointers of the new children.  We do this
            # after clearing *all* child pointers, in case we're e.g.
            # reversing the elements in a tree.
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step)
            # finally, update the content of the child list itself.
            super(AbstractParentedTree, self).__setitem__(index, value)

        # ptree[i] = value
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index out of range")
            # if the value is not changing, do nothing.
            if value is self[index]:
                return
            # Set the new child's parent pointer.
            if isinstance(value, Tree):
                self._setparent(value, index)
            # Remove the old child's parent pointer
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            # Update our child list.
            super(AbstractParentedTree, self).__setitem__(index, value)

        elif isinstance(index, (list, tuple)):
            # ptree[()] = value
            if len(index) == 0:
                raise IndexError("The tree position () may not be assigned to.")
            # ptree[(i,)] = value
            elif len(index) == 1:
                self[index[0]] = value
            # ptree[i1, i2, i3] = value
            else:
                self[index[0]][index[1:]] = value

        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def append(self, child):
        if isinstance(child, Tree):
            self._setparent(child, len(self))
        super(AbstractParentedTree, self).append(child)

    def extend(self, children):
        for child in children:
            if isinstance(child, Tree):
                self._setparent(child, len(self))
            super(AbstractParentedTree, self).append(child)

    def insert(self, index, child):
        # Handle negative indexes.  Note that if index < -len(self),
        # we do *not* raise an IndexError, unlike __getitem__.  This
        # is done for consistency with list.__getitem__ and list.index.
        if index < 0:
            index += len(self)
        if index < 0:
            index = 0
        # Set the child's parent, and update our child list.
        if isinstance(child, Tree):
            self._setparent(child, index)
        super(AbstractParentedTree, self).insert(index, child)

    def pop(self, index=-1):
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("index out of range")
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        return super(AbstractParentedTree, self).pop(index)

    # n.b.: like `list`, this is done by equality, not identity!
    # To remove a specific child, use del ptree[i].
    def remove(self, child):
        index = self.index(child)
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        super(AbstractParentedTree, self).remove(child)

    # We need to implement __getslice__ and friends, even though
    # they're deprecated, because otherwise list.__getslice__ will get
    # called (since we're subclassing from list).  Just delegate to
    # __getitem__ etc., but use max(0, start) and max(0, stop) because
    # because negative indices are already handled *before*
    # __getslice__ is called; and we don't want to double-count them.
    if hasattr(list, "__getslice__"):

        def __getslice__(self, start, stop):
            return self.__getitem__(slice(max(0, start), max(0, stop)))

        def __delslice__(self, start, stop):
            return self.__delitem__(slice(max(0, start), max(0, stop)))

        def __setslice__(self, start, stop, value):
            return self.__setitem__(slice(max(0, start), max(0, stop)), value)


class ParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    single-parented trees.  The following are methods for querying
    the structure of a parented tree: ``parent``, ``parent_index``,
    ``left_sibling``, ``right_sibling``, ``root``, ``treeposition``.

    Each ``ParentedTree`` may have at most one parent.  In
    particular, subtrees may not be shared.  Any attempt to reuse a
    single ``ParentedTree`` as a child of more than one parent (or
    as multiple children of the same parent) will cause a
    ``ValueError`` exception to be raised.

    ``ParentedTrees`` should never be used in the same tree as ``Trees``
    or ``MultiParentedTrees``.  Mixing tree implementations may result
    in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parent = None
        """The parent of this Tree, or None if it has no parent."""
        super(ParentedTree, self).__init__(node, children)
        if children is None:
            # If children is None, the tree is read from node.
            # After parsing, the parent of the immediate children
            # will point to an intermediate tree, not self.
            # We fix this by brute force:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parent = None
                    self._setparent(child, i)

    def _frozen_class(self):
        return ImmutableParentedTree

    # /////////////////////////////////////////////////////////////////
    # Methods
    # /////////////////////////////////////////////////////////////////

    def parent(self):
        """The parent of this tree, or None if it has no parent."""
        return self._parent

    def parent_index(self):
        """
        The index of this tree in its parent.  I.e.,
        ``ptree.parent()[ptree.parent_index()] is ptree``.  Note that
        ``ptree.parent_index()`` is not necessarily equal to
        ``ptree.parent.index(ptree)``, since the ``index()`` method
        returns the first child that is equal to its argument.
        """
        if self._parent is None:
            return None
        for i, child in enumerate(self._parent):
            if child is self:
                return i
        assert False, "expected to find self in self._parent!"

    def left_sibling(self):
        """The left sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index > 0:
            return self._parent[parent_index - 1]
        return None  # no left sibling

    def right_sibling(self):
        """The right sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index < (len(self._parent) - 1):
            return self._parent[parent_index + 1]
        return None  # no right sibling

    def root(self):
        """
        The root of this tree.  I.e., the unique ancestor of this tree
        whose parent is None.  If ``ptree.parent()`` is None, then
        ``ptree`` is its own root.
        """
        root = self
        while root.parent() is not None:
            root = root.parent()
        return root

    def treeposition(self):
        """
        The tree position of this tree, relative to the root of the
        tree.  I.e., ``ptree.root[ptree.treeposition] is ptree``.
        """
        if self.parent() is None:
            return ()
        else:
            return self.parent().treeposition() + (self.parent_index(),)

    # /////////////////////////////////////////////////////////////////
    # Parent Management
    # /////////////////////////////////////////////////////////////////

    def _delparent(self, child, index):
        # Sanity checks
        assert isinstance(child, ParentedTree)
        assert self[index] is child
        assert child._parent is self

        # Delete child's parent pointer.
        child._parent = None

    def _setparent(self, child, index, dry_run=False):
        # If the child's type is incorrect, then complain.
        if not isinstance(child, ParentedTree):
            raise TypeError(
                "Can not insert a non-ParentedTree " + "into a ParentedTree"
            )

        # If child already has a parent, then complain.
        if child._parent is not None:
            raise ValueError("Can not insert a subtree that already " "has a parent.")

        # Set child's parent pointer & index.
        if not dry_run:
            child._parent = self


class MultiParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    multi-parented trees.  The following are methods for querying the
    structure of a multi-parented tree: ``parents()``, ``parent_indices()``,
    ``left_siblings()``, ``right_siblings()``, ``roots``, ``treepositions``.

    Each ``MultiParentedTree`` may have zero or more parents.  In
    particular, subtrees may be shared.  If a single
    ``MultiParentedTree`` is used as multiple children of the same
    parent, then that parent will appear multiple times in its
    ``parents()`` method.

    ``MultiParentedTrees`` should never be used in the same tree as
    ``Trees`` or ``ParentedTrees``.  Mixing tree implementations may
    result in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parents = []
        """A list of this tree's parents.  This list should not
           contain duplicates, even if a parent contains this tree
           multiple times."""
        super(MultiParentedTree, self).__init__(node, children)
        if children is None:
            # If children is None, the tree is read from node.
            # After parsing, the parent(s) of the immediate children
            # will point to an intermediate tree, not self.
            # We fix this by brute force:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parents = []
                    self._setparent(child, i)

    def _frozen_class(self):
        return ImmutableMultiParentedTree

    # /////////////////////////////////////////////////////////////////
    # Methods
    # /////////////////////////////////////////////////////////////////

    def parents(self):
        """
        The set of parents of this tree.  If this tree has no parents,
        then ``parents`` is the empty set.  To check if a tree is used
        as multiple children of the same parent, use the
        ``parent_indices()`` method.

        :type: list(MultiParentedTree)
        """
        return list(self._parents)

    def left_siblings(self):
        """
        A list of all left siblings of this tree, in any of its parent
        trees.  A tree may be its own left sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the left sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [
            parent[index - 1]
            for (parent, index) in self._get_parent_indices()
            if index > 0
        ]

    def right_siblings(self):
        """
        A list of all right siblings of this tree, in any of its parent
        trees.  A tree may be its own right sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the right sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [
            parent[index + 1]
            for (parent, index) in self._get_parent_indices()
            if index < (len(parent) - 1)
        ]

    def _get_parent_indices(self):
        return [
            (parent, index)
            for parent in self._parents
            for index, child in enumerate(parent)
            if child is self
        ]

    def roots(self):
        """
        The set of all roots of this tree.  This set is formed by
        tracing all possible parent paths until trees with no parents
        are found.

        :type: list(MultiParentedTree)
        """
        return list(self._get_roots_helper({}).values())

    def _get_roots_helper(self, result):
        if self._parents:
            for parent in self._parents:
                parent._get_roots_helper(result)
        else:
            result[id(self)] = self
        return result

    def parent_indices(self, parent):
        """
        Return a list of the indices where this tree occurs as a child
        of ``parent``.  If this child does not occur as a child of
        ``parent``, then the empty list is returned.  The following is
        always true::

          for parent_index in ptree.parent_indices(parent):
              parent[parent_index] is ptree
        """
        if parent not in self._parents:
            return []
        else:
            return [index for (index, child) in enumerate(parent) if child is self]

    def treepositions(self, root):
        """
        Return a list of all tree positions that can be used to reach
        this multi-parented tree starting from ``root``.  I.e., the
        following is always true::

          for treepos in ptree.treepositions(root):
              root[treepos] is ptree
        """
        if self is root:
            return [()]
        else:
            return [
                treepos + (index,)
                for parent in self._parents
                for treepos in parent.treepositions(root)
                for (index, child) in enumerate(parent)
                if child is self
            ]

    # /////////////////////////////////////////////////////////////////
    # Parent Management
    # /////////////////////////////////////////////////////////////////

    def _delparent(self, child, index):
        # Sanity checks
        assert isinstance(child, MultiParentedTree)
        assert self[index] is child
        assert len([p for p in child._parents if p is self]) == 1

        # If the only copy of child in self is at index, then delete
        # self from child's parent list.
        for i, c in enumerate(self):
            if c is child and i != index:
                break
        else:
            child._parents.remove(self)

    def _setparent(self, child, index, dry_run=False):
        # If the child's type is incorrect, then complain.
        if not isinstance(child, MultiParentedTree):
            raise TypeError(
                "Can not insert a non-MultiParentedTree " + "into a MultiParentedTree"
            )

        # Add self as a parent pointer if it's not already listed.
        if not dry_run:
            for parent in child._parents:
                if parent is self:
                    break
            else:
                child._parents.append(self)


class ImmutableParentedTree(ImmutableTree, ParentedTree):
    pass


class ImmutableMultiParentedTree(ImmutableTree, MultiParentedTree):
    pass


def slice_bounds(sequence, slice_obj, allow_step=False):
    """
    Given a slice, return the corresponding (start, stop) bounds,
    taking into account None indices and negative indices.  The
    following guarantees are made for the returned start and stop values:

      - 0 <= start <= len(sequence)
      - 0 <= stop <= len(sequence)
      - start <= stop

    :raise ValueError: If ``slice_obj.step`` is not None.
    :param allow_step: If true, then the slice object may have a
        non-None step.  If it does, then return a tuple
        (start, stop, step).
    """
    start, stop = (slice_obj.start, slice_obj.stop)

    # If allow_step is true, then include the step in our return
    # value tuple.
    if allow_step:
        step = slice_obj.step
        if step is None:
            step = 1
        # Use a recursive call without allow_step to find the slice
        # bounds.  If step is negative, then the roles of start and
        # stop (in terms of default values, etc), are swapped.
        if step < 0:
            start, stop = slice_bounds(sequence, slice(stop, start))
        else:
            start, stop = slice_bounds(sequence, slice(start, stop))
        return start, stop, step

    # Otherwise, make sure that no non-default step value is used.
    elif slice_obj.step not in (None, 1):
        raise ValueError(
            "slices with steps are not supported by %s" % sequence.__class__.__name__
        )

    # Supply default offsets.
    if start is None:
        start = 0
    if stop is None:
        stop = len(sequence)

    # Handle negative indices.
    if start < 0:
        start = max(0, len(sequence) + start)
    if stop < 0:
        stop = max(0, len(sequence) + stop)

    # Make sure stop doesn't go past the end of the list.  Note that
    # we avoid calculating len(sequence) if possible, because for lazy
    # sequences, calculating the length of a sequence can be expensive.
    if stop > 0:
        try:
            sequence[stop - 1]
        except IndexError:
            stop = len(sequence)

    # Make sure start isn't past stop.
    start = min(start, stop)

    # That's all folks!
    return start, stop

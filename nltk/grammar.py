# -*- coding: utf-8 -*-
# Natural Language Toolkit: Context Free Grammars
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
#         Jason Narad <jason.narad@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Basic data classes for representing context free grammars.  A
"grammar" specifies which trees can represent the structure of a
given text.  Each of these trees is called a "parse tree" for the
text (or simply a "parse").  In a "context free" grammar, the set of
parse trees for any piece of a text can depend only on that piece, and
not on the rest of the text (i.e., the piece's context).  Context free
grammars are often used to find possible syntactic structures for
sentences.  In this context, the leaves of a parse tree are word
tokens; and the node values are phrasal categories, such as ``NP``
and ``VP``.


A Grammar's "productions" specify what parent-child relationships a parse
tree can contain.  Each production specifies that a particular
node can be the parent of a particular set of children.  For example,
the production ``<S> -> <NP> <VP>`` specifies that an ``S`` node can
be the parent of an ``NP`` node and a ``VP`` node.

Grammar productions are implemented by the ``Production`` class.
Each ``Production`` consists of a left hand side and a right hand
side.  The "left hand side" is a ``Nonterminal`` that specifies the
node type for a potential parent; and the "right hand side" is a list
that specifies allowable children for that parent.  This lists
consists of ``Nonterminals`` and text types: each ``Nonterminal``
indicates that the corresponding child may be a ``TreeToken`` with the
specified node type; and each text type indicates that the
corresponding child may be a ``Token`` with the with that type.

The ``Nonterminal`` class is used to distinguish node values from leaf
values.  This prevents the grammar from accidentally using a leaf
value (such as the English word "A") as the node of a subtree.  Within
a ``CFG``, all node values are wrapped in the ``Nonterminal``
class. Note, however, that the trees that are specified by the grammar do
*not* include these ``Nonterminal`` wrappers.

Grammars can also be given a more procedural interpretation.  According to
this interpretation, a Grammar specifies any tree structure *tree* that
can be produced by the following procedure:

| Set tree to the start symbol
| Repeat until tree contains no more nonterminal leaves:
|   Choose a production prod with whose left hand side
|     lhs is a nonterminal leaf of tree.
|   Replace the nonterminal leaf with a subtree, whose node
|     value is the value wrapped by the nonterminal lhs, and
|     whose children are the right hand side of prod.

The operation of replacing the left hand side (*lhs*) of a production
with the right hand side (*rhs*) in a tree (*tree*) is known as
"expanding" *lhs* to *rhs* in *tree*.
"""

from functools import total_ordering

from nltk.internals import raise_unorderable_types


#################################################################
# Nonterminal
#################################################################


@total_ordering
class Nonterminal(object):
    """
    A non-terminal symbol for a context free grammar.  ``Nonterminal``
    is a wrapper class for node values; it is used by ``Production``
    objects to distinguish node values from leaf values.
    The node value that is wrapped by a ``Nonterminal`` is known as its
    "symbol".  Symbols are typically strings representing phrasal
    categories (such as ``"NP"`` or ``"VP"``).  However, more complex
    symbol types are sometimes used (e.g., for lexicalized grammars).
    Since symbols are node values, they must be immutable and
    hashable.  Two ``Nonterminals`` are considered equal if their
    symbols are equal.

    :see: ``CFG``, ``Production``
    :type _symbol: any
    :ivar _symbol: The node value corresponding to this
        ``Nonterminal``.  This value must be immutable and hashable.
    """

    def __init__(self, symbol):
        """
        Construct a new non-terminal from the given symbol.

        :type symbol: any
        :param symbol: The node value corresponding to this
            ``Nonterminal``.  This value must be immutable and
            hashable.
        """
        self._symbol = symbol

    def symbol(self):
        """
        Return the node value corresponding to this ``Nonterminal``.

        :rtype: (any)
        """
        return self._symbol

    def __eq__(self, other):
        """
        Return True if this non-terminal is equal to ``other``.  In
        particular, return True if ``other`` is a ``Nonterminal``
        and this non-terminal's symbol is equal to ``other`` 's symbol.

        :rtype: bool
        """
        return type(self) == type(other) and self._symbol == other._symbol

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Nonterminal):
            raise_unorderable_types("<", self, other)
        return self._symbol < other._symbol

    def __hash__(self):
        return hash(self._symbol)

    def __repr__(self):
        """
        Return a string representation for this ``Nonterminal``.

        :rtype: str
        """
        if isinstance(self._symbol, str):
            return "%s" % self._symbol
        else:
            return "%s" % repr(self._symbol)

    def __str__(self):
        """
        Return a string representation for this ``Nonterminal``.

        :rtype: str
        """
        if isinstance(self._symbol, str):
            return "%s" % self._symbol
        else:
            return "%s" % repr(self._symbol)

    def __div__(self, rhs):
        """
        Return a new nonterminal whose symbol is ``A/B``, where ``A`` is
        the symbol for this nonterminal, and ``B`` is the symbol for rhs.

        :param rhs: The nonterminal used to form the right hand side
            of the new nonterminal.
        :type rhs: Nonterminal
        :rtype: Nonterminal
        """
        return Nonterminal("%s/%s" % (self._symbol, rhs._symbol))

    def __truediv__(self, rhs):
        """
        Return a new nonterminal whose symbol is ``A/B``, where ``A`` is
        the symbol for this nonterminal, and ``B`` is the symbol for rhs.
        This function allows use of the slash ``/`` operator with
        the future import of division.

        :param rhs: The nonterminal used to form the right hand side
            of the new nonterminal.
        :type rhs: Nonterminal
        :rtype: Nonterminal
        """
        return self.__div__(rhs)



def is_nonterminal(item):
    """
    :return: True if the item is a ``Nonterminal``.
    :rtype: bool
    """
    return isinstance(item, Nonterminal)


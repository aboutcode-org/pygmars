# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project

"""
This module defines  ``Parser`` which is a regular-expression based parser to
parse list of recognized Tokens in a parse tree where each node has a label.

This is originally based on NLTK POS chunk parsing used to identify non-
overlapping linguistic groups (such as base noun phrases) in unrestricted text.

This parsing can identify and group sequences of tokens in a shallow tree called
a ParseTree. A ParseTree is a tree containing Tokens and groups, where each
group is a subtree containing only Tokens.  For example, the ParseTree for base
noun phrase groups in the sentence "I saw the big dog on the hill" is::

  (SENTENCE:
    (NP: <I>)
    <saw>
    (NP: <the> <big> <dog>)
    <on>
    (NP: <the> <hill>))

To convert a ParseTree back to a list of tokens use the ParseTree's ``leaves()``
method.


Rule
=================

``Rule`` uses regular-expressions over Token and Tree labels to parse and group
tokens and trees. The ``parse()`` method constructs a ``ParseString`` that
encodes a particular group of tokens.  Initially, nothing is grouped. ``Rule``
then applies its ``pattern`` substitution to the ``ParseString`` to modify the
token grouping that it encodes. Finally, the ``ParseString`` is transformed back
and returned as a ParseTree.

A ``Rule`` is used to parse a sequence of Tokens and assign a label to a sub-
sequence using a regular expression. Multiple ``Rule``s form a grammar used by a
``Parser``. Many ``Rule``s are typically loaded from a grammar text by a parser.


Rule pattern over labels
-------------------------

A ``pattern`` can update the grouping of tokens and trees by modifying a
``ParseString``. Each ``pattern`` ``apply_transform()`` method modifies the
grouping encoded by a ``ParseString``.

A ``pattern`` uses a modified version of regular expression patterns.  Patterns
are used to match sequence of Token or Tree labels. Examples of label patterns
are::

     r'(<DT>|<JJ>|<NN>)+'
     r'<NN>+'
     r'<NN.*>'

The differences between regular expression patterns and label patterns are:

    - In label patterns, ``'<'`` and ``'>'`` act as parentheses; so
      ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
      ``'<NN'`` followed by one or more repetitions of ``'>'``.

    - Whitespace in label patterns is ignored.  So
      ``'<DT> | <NN>'`` is equivalant to ``'<DT>|<NN>'``

    - In label patterns, ``'.'`` is equivalant to ``'[^{}<>]'``; so
      ``'<NN.*>'`` matches any single label starting with ``'NN'``.

The function ``label_pattern_to_regex`` is used to transform a label pattern to
an equivalent regular expression pattern which is then used internally over the
``ParseString`` encoding.

"""
# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit and a standalone library
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
#
# The Natural Language Toolkit (NLTK) is an open source Python library
# for Natural Language Processing.  A free online book is available.
# (If you use the library for academic research, please cite the book.)
#
# Steven Bird, Ewan Klein, and Edward Loper (2009).
# Natural Language Processing with Python.  O'Reilly Media Inc.
# http://nltk.org/book

import re
from functools import partial

from pygmars import as_token_label
from pygmars.tree import Tree as ParseTree


class ParseString:
    """
    A ParseString is a string-based encoding of a particular parsing of a
    sequence of Tokens. Internally, the ``ParseString`` class uses a tsring and
    a list:

    - a string of Token or Tree labels that encode the parsing of the input
    tokens. This is the string to which the Rule's pattern regular expression
    transformations are applied. This string contains a sequence of angle-
    bracket delimited labels (e.g. Token or Tree labels), with the grouping
    indicated by curly braces.  An example of this encoding is::

        {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}{<DT><NN>}<VBD>

    - a parallel and backing list of Tokens and Trees.

    ``ParseString`` are created from a Tokens list (built from lexed texts).
    Initially, nothing is parsed and tokens are not grouped under a label.

    The parsing of a ``ParseString`` is modified with the ``apply_transform()``
    method, which uses the Rule regular expression to transform the string
    representation.  These transformations can only add and remove braces;
    they should *not* modify the sequence of angle-bracket delimited labels.

    :type _parse_string: str
    :ivar _parse_string: The internal string representation of the text's
        encoding.  This string representation contains a sequence of
        angle-bracket delimited tags, with parsing indicated by
        braces.  An example of this encoding is::

            {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    :type _pieces: list(tokens and groups)
    :ivar _pieces: The tokens and groups encoded by this ``ParseString``.
    :ivar _debug: The debug level.  See the constructor docs.

    :cvar IN_PATTERN: A zero-width regexp pattern string that will only match
        positions that are inside groups.

    :cvar BETWEEN_PATTERN: A zero-width regexp pattern string that will only
        match positions that are between groups.
    """
    # Anything that's not a delimiter such as <> or {}
    LABEL_CHARS = r"[^\{\}<>]"

    LABEL = fr"(<{LABEL_CHARS}+?>)"

    # (?= is a lookahead assertion:
    # match and not "consume" anything that has no { and ends with }
    IN_PATTERN = r"(?=[^\{]*\})"
    # match and not "consume" anything that has no } and will return an opening {
    BETWEEN_PATTERN = r"(?=[^\}]*(\{|$))"

    # return a True'ish value if this pattern is valid
    is_valid = re.compile(r"^(\{?%s\}?)*?$" % LABEL).match

    def __init__(self, tree, debug_level=1):
        """
        Construct a new ``ParseString`` from a ``tree`` ParseTree of Tokens.

        ``debug_level`` is the int level of debugging that should be applied to
        transformations on the ``ParseString``.  The valid levels are:

        - 0: no checks
        - 1: full check on ``to_tree()`` calls
        - 2: full check on ``to_tree()`` calls and quick check after each transformation.
        - 3: full check on ``to_tree()`` calls and full check after each transformation.
        """
        self._root_label = tree.label
        self._pieces = tree[:]
        labels = "><".join(tok.label for tok in self._pieces)
        self._parse_string = f"<{labels}>"
        self._debug = debug_level

    def _verify(self, s, verify_tags=False):
        """
        Validate that the string ``s`` corresponds to a parsed version of ``_pieces``.

        If ``verify_tags`` is True, check whether the individual tags should be
        checked.  If this is False, ``_verify`` will check to make sure that
        ``_str`` encodes a parsed version of a list of tokens.  If this is
        true, then ``_verify`` will check to make sure that the tags in
        ``_str`` match those in ``_pieces``.

        :raise ValueError: if the internal string representation of
            this ``ParseString`` is invalid or not consistent with its _pieces.
        """
        if not ParseString.is_valid(s):
            raise ValueError(f"Invalid parse:\n  {s}")

        if not has_balanced_non_nested_curly_braces(s):
            raise ValueError(f"Invalid parse: unbalanced or nested curly braces:\n  {s}")

        if verify_tags <= 0:
            return

        tags1 = tag_splitter(s)[1:-1]
        tags2 = [piece.label for piece in self._pieces]
        if tags1 != tags2:
            raise ValueError(f"Invalid parse: tag changed:\n  {s}")

    def to_tree(self, label="GROUP", pieces_splitter=re.compile(r"[{}]").split):
        """
        Return a ParseTree for this ``ParseString`` using ``label`` as the root
        label. Raise a ValueError if a transformation creates an invalid
        ParseString.
        """
        if self._debug > 0:
            self._verify(self._parse_string, 1)

        # Use this alternating list to create the ParseTree.
        pieces = []
        index = 0
        piece_in_group = 0
        for piece in pieces_splitter(self._parse_string):

            # Find the list of tokens contained in this piece.
            length = piece.count("<")
            subsequence = self._pieces[index: index + length]

            # Add this list of tokens to our pieces.
            if piece_in_group:
                pieces.append(ParseTree(label, subsequence))
            else:
                pieces.extend(subsequence)

            index += length
            piece_in_group = not piece_in_group

        return ParseTree(self._root_label, pieces)

    def apply_transform(self, transformer):
        """
        Apply the given ``transformer`` callable transformation to the string
        encoding of this ``ParseString``.

        This transformation should only add and remove braces; it should *not*
        modify the sequence of angle-bracket delimited tags.  Furthermore, this
        transformation may not result in improper bracketing.  Note, in
        particular, that bracketing may not be nested.

        ``transformer`` is a callable that accepts a string and returns a string
        Raise ValueError if this transformation generates an invalid ParseString.
        """
        # Do the actual substitution
        s = transformer(self._parse_string)

        # The substitution might have generated "empty groups"
        # (substrings of the form "{}").  Remove them, so they don't
        # interfere with other transformations.
        s = s.replace("{}", "")

        # Make sure that the transformation was legal.
        if self._debug > 1:
            self._verify(s, self._debug - 2)

        # Save the transformation.
        self._parse_string = s

    def __repr__(self):
        return f"<ParseString: {self._parse_string!r}>"

    def __str__(self):
        """
        Return a formatted representation of this ``ParseString``. This
        representation includes extra spaces to ensure that labels will line up
        with the representation of other ``ParseString`` for the same text,
        regardless of the grouping.
        """
        # Add spaces to make everything line up.
        s = re.sub(r">(?!\})", r"> ", self._parse_string)
        s = re.sub(r"([^\{])<", r"\1 <", s)
        if s[0] == "<":
            s = " " + s
        return s.rstrip()


# used to split a ParseString on labels and braces delimiters
tag_splitter = re.compile(r"[\{\}<>]+").split

# return only {} curly brackets aka. braces
get_curly_braces = partial(re.compile(r"[^\{\}]+").sub, "")

# remove {4,} regex quantifiers
remove_quantifiers = partial(re.compile(r"\{\d+(?:,\d+)?\}").sub, "")


def has_balanced_non_nested_curly_braces(string):
    """
    Return True if ``string`` contains balanced and non-nested curly braces.

    Approach:
    - remove regex quantifiers
    - remove all non-braces characters
    - remove all balanced brace pairs.
    If there is nothing left, then braces are balanced and not nested.

    Balanced but nested:
    >>> "{{}}".replace("{}", "")
    '{}'

    Unbalanced:
    >>> "{{}{}".replace("{}", "")
    '{'
    >>> "{}{}}{}".replace("{}", "")
    '}'

    Balanced an not nested:
    >>> "{}{}{}".replace("{}", "")
    ''
    >>> remove_quantifiers("foo{4}")
    'foo'
    >>> remove_quantifiers("foo{}")
    'foo{}'
    >>> remove_quantifiers("foo{4}")
    'foo'
    >>> remove_quantifiers("foo{4,5}")
    'foo'

    >>> has_balanced_non_nested_curly_braces("{}{}{}")
    True
    >>> has_balanced_non_nested_curly_braces("{{}{}{}")
    False
    """
    cb = get_curly_braces(string)
    cb = remove_quantifiers(cb)
    return bool(not cb.replace("{}", ""))


# this should probably be made more strict than it is -- e.g., it
# currently accepts 'foo'.
is_label_pattern = re.compile(
    r"^((%s|<%s>)*)$" % (
        r"([^{}<>]|{\d+,?}|{\d*,\d+})+",
        r"[^{}<>]+"
    )
).match

remove_spaces = re.compile(r"\s").sub


def label_pattern_to_regex(label_pattern):
    """
    Return a regular expression pattern converted from ``label_pattern``.  A
    "label pattern" is a modified version of a regular expression, designed for
    matching sequences of labels.  The differences between regular expression
    patterns and tag patterns are:

    - In label patterns, ``'<'`` and ``'>'`` act as parentheses; so
      ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
      ``'<NN'`` followed by one or more repetitions of ``'>'``.

    - Whitespace in label patterns is ignored.  So
      ``'<DT> | <NN>'`` is equivalant to ``'<DT>|<NN>'``

    - In label patterns, ``'.'`` is equivalant to ``'[^{}<>]'``; so
      ``'<NN.*>'`` matches any single tag starting with ``'NN'``.

    In particular, ``label_pattern_to_regex`` performs the following
    transformations on the given pattern:

    - Replace '.' with '[^<>{}]'

    - Remove any whitespace

    - Add extra parens around '<' and '>', to make '<' and '>' act
      like parentheses, so that in '<NN>+', the '+' has scope
      over the entire '<NN>'; and so that in '<NN|IN>', the '|' has
      scope over 'NN' and 'IN', but not '<' or '>'.

    - Check to make sure the resulting pattern is valid.

    Raise ValueError if ``label_pattern`` is not a valid pattern. In particular,
    ``label_pattern`` should not include braces (except for quantifiers); and it
    should not contain nested or mismatched angle-brackets.
    """
    # Clean up the regular expression
    label_pattern = (
        remove_spaces("", label_pattern)
        .replace("<", "(<(")
        .replace(">", ")>)")
    )

    # Check the regular expression
    if not is_label_pattern(label_pattern):
        raise ValueError("Bad label pattern: %r" % label_pattern)

    return label_pattern.replace(".", ParseString.LABEL_CHARS)


class Rule:
    """
    A regular expression-based parsing ``Rule`` to find and label groups of
    labelled Tokens and Trees.  The grouping of the tokens is encoded using a
    ``ParseString``, and each rule acts by modifying the grouping in the
    ``ParseString`` with its ``pattern``.  The patterns are implemented using
    regular expression substitution.
    """

    def __init__(
        self,
        pattern,
        label,
        description=None,
        root_label="ROOT",
        trace=0,
    ):
        """
        Construct a new ``Rule`` from a ``pattern`` string for a ``label``
        string and an opetional ``description`` string.

        ``root_label`` is the label value used for the top/root node of the tree
        structure.

        ``trace`` is the level of tracing when parsing.  ``0`` will generate no
        tracing output; ``1`` will generate normal tracing output; and ``2`` or
        higher will generate verbose tracing output.
        """
        self.pattern = pattern
        self.label = label
        self.description = description
        self._root_label = root_label
        self._trace = trace

        regexp = label_pattern_to_regex(pattern)
        regexp = fr"(?P<group>{regexp}){ParseString.BETWEEN_PATTERN}"
        # the replacement wraps matched tokens in curly braces
        self._repl = "{\\g<group>}"
        self._transformer = partial(re.compile(regexp).sub, self._repl)

    def validate(self):
        """
        Validate this Rule and raise Exceptions on errors.
        """
        if not self.pattern:
            raise Exception("Illegal Rule: empty pattern")

        if not self.label:
            raise Exception("Illegal Rule: empty label")

        if self.label != as_token_label(self.label):
            raise Exception(f"Illegal Rule label: {self.label}")

    def parse(self, tree, trace=None):
        """
        Parse the ``tree`` ParseTree and return a new ParseTree that encodes the
        parsing in groups of a given Token sequence.  The set of nodes
        identified in the tree depends on the pattern of this ``Rule``.

        ``trace`` is the level of tracing when parsing.  ``0`` will generate no
        tracing output; ``1`` will generate normal tracing output; and ``2`` or
        higher will generate verbose tracing output. This value overrides the
        trace level value that was given to the constructor.
        """

        if len(tree) == 0:
            raise Exception(f"Warning: parsing empty tree: {tree!r}")

        # the initial tree may be a list and not yet a tree
        try:
            tree.label
        except AttributeError:
            tree = ParseTree(self._root_label, tree)

        # Use the default trace value?
        if trace is None:
            trace = self._trace

        parse_string = ParseString(tree)

        # Apply the sequence of rules to the ParseString.
        verbose = trace > 1
        if trace:
            print("# Input:")
            print(parse_string)

        parse_string.apply_transform(self._transformer)

        if verbose:
            print("#", self.description + " (" + repr(self.pattern) + "):")
        elif trace:
            print("#", self.description + ":")
            print(parse_string)

        return parse_string.to_tree(self.label)

    def __repr__(self):
        if self.description:
            return f"<Rule: {self.pattern} / {self.label} # {self.description}>"
        return f"<Rule: {self.pattern} / {self.label}>"

    __str__ = __repr__

    @classmethod
    def from_string(cls, string, root_label="ROOT", trace=0):
        """
        Create a Rule from a grammar rule ``string`` in this format::

          label: {pattern} # description

        Where ``pattern`` is a regular expression for the rule.  Any
        text following the comment marker (``#``) will be used as
        the rule's description:

        >>> from pygmars.parse import Rule
        >>> Rule.from_string('FOO: {<DT>?<NN.*>+}')
        <Rule: <DT>?<NN.*>+ / FOO>
        """
        label, _, pattern = string.partition(":")
        pattern, _, description = pattern.partition("#")
        label = label.strip()
        pattern = pattern.strip()
        description = description.strip()

        if not pattern.startswith("{") or not pattern.endswith("}"):
            raise ValueError(
                f"Illegal pattern: {pattern}: notenclosed in curly braces.")

        pattern = pattern[1:-1]

        if not pattern:
            raise ValueError(f"Empty pattern: {string}")

        if not label:
            raise ValueError(f"Missing rule label: {string}")

        return Rule(
            pattern=pattern,
            label=label,
            description=description,
            root_label=root_label,
            trace=trace,
        )

    @classmethod
    def from_grammar(cls, grammar, root_label="ROOT", trace=0):
        """
        Yield Rules from ``grammar`` string.
        Raise Exceptions on errors.

        ``trace`` is the level of tracing to use when parsing a text.  ``0``
        will generate no tracing output; ``1`` will generate normal tracing
        output; and ``2`` or higher will generate verbose tracing output.

        A grammar is a collection of Rules that can be built from a string. A
        grammar contains one or more rule (one rule per line) in this form::

         NP: {<DT|JJ>}          # determiners and adjectives

        Here NP is a label and {<DT|JJ>} is the pattern and the comment is used
        as a description.
        """
        for line in grammar.splitlines(False):
            line = line.strip()
            if not line or line.startswith("#"):
                # Skip blank & comment-only lines
                continue
            yield cls.from_string(string=line, root_label=root_label, trace=trace)


class Parser:
    """
    Parser is a grammar-based parser that uses a grammar which is a list of Rule
    with patterns that are specialized regular expression for over Token or Tree
    labels. Internally the parsing of a Token sequence is encoded using a
    ``ParseString``, and each Rule pattern acts by modifying the grouping and
    parsing in the ``ParseString``. Rule patterns are implemented using regular
    expression substitution.

    The Rule patterns of a grammar are executed in sequence.  An earlier pattern
    may introduce a parse boundary that prevents a later pattern from matching.
    Sometimes an individual pattern will match on multiple, overlapping extents
    of the input.  As with regular expression substitution, the parser will
    identify the first match possible, then continue looking for all other
    following matches.

    The maximum depth of a parse tree created by a parser is the same as the
    number of Rules in the grammar.

    When tracing is turned on, the comment portion of a line is displayed each
    time the corresponding pattern is applied.
    """

    def __init__(self, grammar, root_label="ROOT", loop=1, trace=0):
        """
        Create a new Parser from a ``grammar`` string and a ``root_label``.

        ``loop`` is the number of times to run through all the patterns (as the
        evaluation of Rules is sequential and not recursive).

        ``trace`` is the level of tracing to use when parsing a text.  ``0``
        will generate no tracing output; ``1`` will generate normal tracing
        output; and ``2`` or higher will generate verbose tracing output.
        """
        self._grammar = grammar
        self.rules = list(Rule.from_grammar(grammar, root_label, trace))
        self._root_label = root_label
        self._loop = loop
        self._trace = trace

    def parse(self, tree, trace=None):
        """
        Apply this parser to the ``tree`` ParseTree and return a ParseTree.
        The tree is modified in place and returned.

        ``trace`` is the level of tracing to use when parsing a text.  ``0``
        will generate no tracing output; ``1`` will generate normal tracing
        output; and ``2`` or higher will generate verbose tracing output.
        This value overrides the trace level given to the constructor.
        """
        if trace is None:
            trace = self._trace

        for _ in range(self._loop):
            for parse_rule in self.rules:
                tree = parse_rule.parse(tree, trace=trace)
        return tree

    def __repr__(self):
        return f"<Parser with {len(self.rules)} rules>"

    def __str__(self):
        rules = "\n".join(map(str, self.rules))
        return f"Parser with  {len(self.rules)} rules:\n{rules}"

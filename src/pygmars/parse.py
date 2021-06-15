# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>
#
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)

import re
from functools import partial

from pygmars.tree import Tree



"""
Classes to parse list of recognized tokens in a parse tree.

Originally based on NLTK POS chunk parsing used to identify non-overlapping
linguistic groups (such as base noun phrases) in unrestricted text.

This lightweight parsing can identify, group and name sequences of tokens as
"chunks" The chunked text is represented using a shallow tree called a "chunk
structure." A chunk structure is a tree containing tokens and chunks, where each
chunk is a subtree containing only tokens.  For example, the chunk structure for
base noun phrase chunks in the sentence "I saw the big dog on the hill" is::

  (SENTENCE:
    (NP: <I>)
    <saw>
    (NP: <the> <big> <dog>)
    <on>
    (NP: <the> <hill>))

To convert a chunk structure back to a list of tokens, simply use the
chunk structure's ``leaves()`` method.

This module defines ``RegexpParser``, a regular-expression based parser.

RegexpChunkParser
=================

``RegexpChunkParser`` is an implementation of the chunk parser interface
that uses regular-expressions over tags to chunk a text.  Its
``parse()`` method first constructs a ``ChunkString``, which encodes a
particular chunking of the input text.  Initially, nothing is
chunked.  ``parse.RegexpChunkParser`` then applies a sequence of
``RegexpChunkRule`` rules to the ``ChunkString``, each of which modifies
the chunking that it encodes.  Finally, the ``ChunkString`` is
transformed back into a chunk structure, which is returned.

``RegexpChunkParser`` can only be used to chunk a single kind of phrase.
For example, you can use an ``RegexpChunkParser`` to chunk the noun
phrases in a text, or the verb phrases in a text; but you can not
use it to simultaneously chunk both noun phrases and verb phrases in
the same text.  (This is a limitation of ``RegexpChunkParser``, not of
chunk parsers in general.)

RegexpChunkRules
----------------

A ``RegexpChunkRule`` is a transformational rule that updates the
chunking of a text by modifying its ``ChunkString``.  Each
``RegexpChunkRule`` defines the ``apply()`` method, which modifies
the chunking encoded by a ``ChunkString``.  The
``RegexpChunkRule`` class itself can be used to implement any
transformational rule based on regular expressions.  There are
also a number of subclasses, which can be used to implement
simpler types of rules:

    - ``ChunkRule`` chunks anything that matches a given regular
      expression.
    - ``ChinkRule`` chinks anything that matches a given regular
      expression.
    - ``UnChunkRule`` will un-chunk any chunk that matches a given
      regular expression.
    - ``MergeRule`` can be used to merge two contiguous chunks.
    - ``SplitRule`` can be used to split a single chunk into two
      smaller chunks.
    - ``ExpandLeftRule`` will expand a chunk to incorporate new
      unchunked material on the left.
    - ``ExpandRightRule`` will expand a chunk to incorporate new
      unchunked material on the right.

Tag Patterns
~~~~~~~~~~~~

A ``RegexpChunkRule`` uses a modified version of regular
expression patterns, called "tag patterns".  Tag patterns are
used to match sequences of tags.  Examples of tag patterns are::

     r'(<DT>|<JJ>|<NN>)+'
     r'<NN>+'
     r'<NN.*>'

The differences between regular expression patterns and tag
patterns are:

    - In tag patterns, ``'<'`` and ``'>'`` act as parentheses; so
      ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
      ``'<NN'`` followed by one or more repetitions of ``'>'``.
    - Whitespace in tag patterns is ignored.  So
      ``'<DT> | <NN>'`` is equivalant to ``'<DT>|<NN>'``
    - In tag patterns, ``'.'`` is equivalant to ``'[^{}<>]'``; so
      ``'<NN.*>'`` matches any single tag starting with ``'NN'``.

The function ``tag_pattern2re_pattern`` can be used to transform
a tag pattern to an equivalent regular expression pattern.

Efficiency
----------

Preliminary tests indicate that ``RegexpChunkParser`` can chunk at a
rate of about 300 tokens/second, with a moderately complex rule set.

There may be problems if ``RegexpChunkParser`` is used with more than
5,000 tokens at a time.  In particular, evaluation of some regular
expressions may cause the Python regular expression engine to
exceed its maximum recursion depth.  We have attempted to minimize
these problems, but it is impossible to avoid them completely.  We
therefore recommend that you apply the chunk parser to a single
sentence at a time.

Emacs Tip
---------

If you evaluate the following elisp expression in emacs, it will
colorize a ``ChunkString`` when you use an interactive python shell
with emacs or xemacs ("C-c !")::

    (let ()
      (defconst comint-mode-font-lock-keywords
        '(("<[^>]+>" 0 'font-lock-reference-face)
          ("[{}]" 0 'font-lock-function-name-face)))
      (add-hook 'comint-mode-hook (lambda () (turn-on-font-lock))))

You can evaluate this code by copying it to a temporary buffer,
placing the cursor after the last close parenthesis, and typing
"``C-x C-e``".  You should evaluate it before running the interactive
session.  The change will last until you close emacs.

Unresolved Issues
-----------------

If we use the ``re`` module for regular expressions, Python's
regular expression engine generates "maximum recursion depth
exceeded" errors when processing very large texts, even for
regular expressions that should not require any recursion.  We
therefore use the ``pre`` module instead.  But note that ``pre``
does not include Unicode support, so this module will not work
with unicode strings.  Note also that ``pre`` regular expressions
are not quite as advanced as ``re`` ones (e.g., no leftward
zero-length assertions).

:type CHUNK_TAG_PATTERN: regexp
:var CHUNK_TAG_PATTERN: A regular expression to test whether a tag
     pattern is valid.
"""


class ChunkString(object):
    """
    A string-based encoding of a particular chunking of a text.
    Internally, the ``ChunkString`` class uses a single string to
    encode the chunking of the input text.  This string contains a
    sequence of angle-bracket delimited tags, with chunking indicated
    by braces.  An example of this encoding is::

        {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    ``ChunkString`` are created from tagged texts (i.e., lists of
    ``tokens``.  Initially, nothing is chunked.

    The chunking of a ``ChunkString`` can be modified with the ``apply_transform()``
    method, which uses a regular expression to transform the string
    representation.  These transformations should only add and remove
    braces; they should *not* modify the sequence of angle-bracket
    delimited tags.

    :type _str: str
    :ivar _str: The internal string representation of the text's
        encoding.  This string representation contains a sequence of
        angle-bracket delimited tags, with chunking indicated by
        braces.  An example of this encoding is::

            {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    :type _pieces: list(tagged tokens and chunks)
    :ivar _pieces: The tagged tokens and chunks encoded by this ``ChunkString``.
    :ivar _debug: The debug level.  See the constructor docs.

    :cvar IN_CHUNK_PATTERN: A zero-width regexp pattern string that
        will only match positions that are in chunks.
    :cvar IN_STRIP_PATTERN: A zero-width regexp pattern string that
        will only match positions that are in strips.
    """
    # defines characters
    CHUNK_TAG_CHAR = r"[^\{\}<>]"
    CHUNK_TAG = r"(<%s+?>)" % CHUNK_TAG_CHAR
    CHUNK_TAG_CHAR_REVERSED = r"".join(reversed(CHUNK_TAG_CHAR))

    IN_CHUNK_PATTERN = r"(?=[^\{]*\})"
    IN_STRIP_PATTERN = r"(?=[^\}]*(\{|$))"

    # These are used by _verify
    _CHUNK = r"(\{%s+?\})+?" % CHUNK_TAG
    _STRIP = r"(%s+?)+?" % CHUNK_TAG

    # return a True'ish value if this tag is valid
    is_valid = re.compile(r"^(\{?%s\}?)*?$" % CHUNK_TAG).match
    # return only {} curly brackets aka. braces
    get_curly_brackets = partial(re.compile(r"[^\{\}]+").sub, "")
    # return a True'ish value if braces are balanced and NOT nested
    has_balanced_non_nested_curly_brackets = re.compile(r"(\{\})*$").match

    def __init__(self, chunk_struct, debug_level=1):
        """
        Construct a new ``ChunkString`` that encodes the chunking of
        the text ``tagged_tokens``.

        :type chunk_struct: Tree
        :param chunk_struct: The chunk structure to be further chunked.
        :type debug_level: int
        :param debug_level: The level of debugging which should be
            applied to transformations on the ``ChunkString``.  The
            valid levels are:
                - 0: no checks
                - 1: full check on to_chunkstruct
                - 2: full check on to_chunkstruct and cursory check after
                   each transformation.
                - 3: full check on to_chunkstruct and full check after
                   each transformation.
            We recommend you use at least level 1.  You should
            probably use level 3 if you use any non-standard
            subclasses of ``RegexpChunkRule``.
        """
        self._root_label = chunk_struct.label()
        self._pieces = chunk_struct[:]
        tags = [self._tag(tok) for tok in self._pieces]
        self._str = "<" + "><".join(tags) + ">"
        self._debug = debug_level

    def _tag(self, tok):
        if isinstance(tok, tuple):
            return tok[1]
        elif isinstance(tok, Tree):
            return tok.label()
        else:
            raise ValueError("chunk structures must contain tagged tokens or trees")

    def _verify(
            self,
            s,
            verify_tags,
            tag_splitter=re.compile(r"[\{\}<>]+").split,
        ):
        """
        Check to make sure that ``s`` still corresponds to some chunked
        version of ``_pieces``.

        :type verify_tags: bool
        :param verify_tags: Whether the individual tags should be
            checked.  If this is false, ``_verify`` will check to make
            sure that ``_str`` encodes a chunked version of *some*
            list of tokens.  If this is true, then ``_verify`` will
            check to make sure that the tags in ``_str`` match those in
            ``_pieces``.

        :raise ValueError: if the internal string representation of
            this ``ChunkString`` is invalid or not consistent with _pieces.
        """
        # Check overall form
        if not ChunkString.is_valid(s):
            raise ValueError(
                "Transformation generated invalid " "chunkstring:\n  %s" % s
            )

        # Check that curly brackets are balanced.  If the string is long, we
        # have to do this in pieces, to avoid a maximum recursion
        # depth limit for regular expressions.
        curly_brackets = ChunkString.get_curly_brackets(s)
        for i in range(1 + len(curly_brackets) // 5000):
            substr = curly_brackets[i * 5000 : i * 5000 + 5000]
            if not ChunkString.has_balanced_non_nested_curly_brackets(substr):
                raise ValueError(
                    f"Transformation generated invalid chunkstring:\n  {s}"
                )

        if verify_tags <= 0:
            return

        tags1 = tag_splitter(s)[1:-1]
        tags2 = [self._tag(piece) for piece in self._pieces]
        if tags1 != tags2:
            raise ValueError(
                "Transformation generated invalid chunkstring: tag changed"
            )

    def to_chunkstruct(
        self,
        chunk_label="CHUNK",
        get_chunk_pieces=re.compile(r"[{}]").split,
    ):
        """
        Return the chunk structure encoded by this ``ChunkString``.

        :rtype: Tree
        :raise ValueError: If a transformation has generated an
            invalid chunkstring.
        """
        if self._debug > 0:
            self._verify(self._str, 1)

        # Use this alternating list to create the chunkstruct.
        pieces = []
        index = 0
        piece_in_chunk = 0
        for piece in get_chunk_pieces(self._str):

            # Find the list of tokens contained in this piece.
            length = piece.count("<")
            subsequence = self._pieces[index: index + length]

            # Add this list of tokens to our pieces.
            if piece_in_chunk:
                pieces.append(Tree(chunk_label, subsequence))
            else:
                pieces += subsequence

            # Update index, piece_in_chunk
            index += length
            piece_in_chunk = not piece_in_chunk

        return Tree(self._root_label, pieces)

    def apply_transform(self, regexp, repl):
        """
        Apply the given transformation to the string encoding of this
        ``ChunkString``.  In particular, find all occurrences that match
        ``regexp``, and replace them using ``repl`` (as done by
        ``re.sub``).

        This transformation should only add and remove braces; it
        should *not* modify the sequence of angle-bracket delimited
        tags.  Furthermore, this transformation may not result in
        improper bracketing.  Note, in particular, that bracketing may
        not be nested.

        :type regexp: str or regexp
        :param regexp: A regular expression matching the substring
            that should be replaced.  This will typically include a
            named group, which can be used by ``repl``.
        :type repl: str
        :param repl: An expression specifying what should replace the
            matched substring.  Typically, this will include a named
            replacement group, specified by ``regexp``.
        :rtype: None
        :raise ValueError: If this transformation generated an
            invalid chunkstring.
        """
        # Do the actual substitution
        s = re.sub(regexp, repl, self._str)

        # The substitution might have generated "empty chunks"
        # (substrings of the form "{}").  Remove them, so they don't
        # interfere with other transformations.
        s = s.replace("{}", "")

        # Make sure that the transformation was legal.
        if self._debug > 1:
            self._verify(s, self._debug - 2)

        # Commit the transformation.
        self._str = s

    def __repr__(self):
        return f"<ChunkString: {self._str!r}>"

    def __str__(self):
        """
        Return a formatted representation of this ``ChunkString``.
        This representation will include extra spaces to ensure that
        tags will line up with the representation of other
        ``ChunkStrings`` for the same text, regardless of the chunking.

       :rtype: str
        """
        # Add spaces to make everything line up.
        s = re.sub(r">(?!\})", r"> ", self._str)
        s = re.sub(r"([^\{])<", r"\1 <", s)
        if s[0] == "<":
            s = " " + s
        return s.rstrip()


class RegexpChunkRule:
    """
    A rule specifying how to modify the chunking in a ``ChunkString``,
    using a transformational regular expression.  The
    ``RegexpChunkRule`` class itself can be used to implement any
    transformational rule based on regular expressions.  There are
    also a number of subclasses, which can be used to implement
    simpler types of rules, based on matching regular expressions.

    Each ``RegexpChunkRule`` has a regular expression and a
    replacement expression.  When a ``RegexpChunkRule`` is "applied"
    to a ``ChunkString``, it searches the ``ChunkString`` for any
    substring that matches the regular expression, and replaces it
    using the replacement expression.  This search/replace operation
    has the same semantics as ``re.sub``.

    Each ``RegexpChunkRule`` also has a description string, which
    gives a short (typically less than 75 characters) description of
    the purpose of the rule.

    This transformation defined by this ``RegexpChunkRule`` should
    only add and remove braces; it should *not* modify the sequence
    of angle-bracket delimited tags.  Furthermore, this transformation
    may not result in nested or mismatched bracketing.
    """

    def __init__(self, regexp, repl, descr):
        """
        Construct a new RegexpChunkRule.

        :type regexp: regexp or str
        :param regexp: The regular expression for this ``RegexpChunkRule``.
            When this rule is applied to a ``ChunkString``, any
            substring that matches ``regexp`` will be replaced using
            the replacement string ``repl``.  Note that this must be a
            normal regular expression, not a tag pattern.
        :type repl: str
        :param repl: The replacement expression for this ``RegexpChunkRule``.
            When this rule is applied to a ``ChunkString``, any substring
            that matches ``regexp`` will be replaced using ``repl``.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        if isinstance(regexp, str):
            regexp = re.compile(regexp)
        self._repl = repl
        self.descr = descr
        self._regexp = regexp
        self.regexp_pattern = regexp.pattern

    def apply(self, chunkstr):
        # Keep docstring generic so we can inherit it.
        """
        Apply this rule to the given ``ChunkString``.  See the
        class reference documentation for a description of what it
        means to apply a rule.

        :type chunkstr: ChunkString
        :param chunkstr: The chunkstring to which this rule is applied.
        :rtype: None
        :raise ValueError: If this transformation generated an
            invalid chunkstring.
        """
        chunkstr.apply_transform(self._regexp, self._repl)

    def __repr__(self):
        return f"<RegexpChunkRule: {self.regexp_pattern!r}->{self._repl!r}>"

    @staticmethod
    def fromstring(
        s,
        comment_splitter=re.compile(r"(?P<rule>(\.|[^#])*)(?P<comment>#.*)?").match,
        is_context_rule=re.compile(r"[^{}]*{[^{}]*}[^{}]*").match,
        context_splitter=re.compile(r"[{}]").split,
    ):
        """
        Create a RegexpChunkRule from a string description.
        Currently, the following formats are supported::

          {regexp}         # chunk rule
          }regexp{         # strip rule
          regexp}{regexp   # split rule
          regexp{}regexp   # merge rule

        Where ``regexp`` is a regular expression for the rule.  Any
        text following the comment marker (``#``) will be used as
        the rule's description:

        >>> from pygmars.parse import RegexpChunkRule
        >>> RegexpChunkRule.fromstring('{<DT>?<NN.*>+}')
        <ChunkRule: '<DT>?<NN.*>+'>
        """
        # Split off the comment (but don't split on '\#')
        m = comment_splitter(s)
        rule = m.group("rule").strip()
        comment = (m.group("comment") or "")[1:].strip()

        # Pattern bodies: chunk, strip, split, merge
        try:
            if not rule:
                raise ValueError("Empty chunk pattern")
            if rule[0] == "{" and rule[-1] == "}":
                return ChunkRule(rule[1:-1], comment)
            else:
                raise ValueError("Illegal chunk pattern: %s" % rule)
        except (ValueError, re.error) as e:
            raise ValueError("Illegal chunk pattern: %s" % rule) from e


class ChunkRule(RegexpChunkRule):
    """
    A rule specifying how to add chunks to a ``ChunkString``, using a
    matching tag pattern.  When applied to a ``ChunkString``, it will
    find any substring that matches this tag pattern and that is not
    already part of a chunk, and create a new chunk containing that
    substring.
    """

    def __init__(self, tag_pattern, descr):
        """
        Construct a new ``ChunkRule``.

        :type tag_pattern: str
        :param tag_pattern: This rule's tag pattern.  When
            applied to a ``ChunkString``, this rule will
            chunk any substring that matches this tag pattern and that
            is not already part of a chunk.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        self._pattern = tag_pattern
        regexp = (
            r"(?P<chunk>%s)%s"
            % (tag_pattern2re_pattern(tag_pattern), ChunkString.IN_STRIP_PATTERN)
        )
        RegexpChunkRule.__init__(self, regexp, "{\\g<chunk>}", descr)

    def __repr__(self):
        return f"<ChunkRule: {self._pattern!r}>"


# this should probably be made more strict than it is -- e.g., it
# currently accepts 'foo'.
is_chunk_tag_pattern = re.compile(
    r"^((%s|<%s>)*)$" % (r"([^{}<>]|{\d+,?}|{\d*,\d+})+", "[^{}<>]+")
).match


def tag_pattern2re_pattern(tag_pattern, remove_spaces=re.compile(r"\s").sub, _cache={}):
    """
    Convert a tag pattern to a regular expression pattern.  A "tag
    pattern" is a modified version of a regular expression, designed
    for matching sequences of tags.  The differences between regular
    expression patterns and tag patterns are:

        - In tag patterns, ``'<'`` and ``'>'`` act as parentheses; so
          ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
          ``'<NN'`` followed by one or more repetitions of ``'>'``.
        - Whitespace in tag patterns is ignored.  So
          ``'<DT> | <NN>'`` is equivalant to ``'<DT>|<NN>'``
        - In tag patterns, ``'.'`` is equivalant to ``'[^{}<>]'``; so
          ``'<NN.*>'`` matches any single tag starting with ``'NN'``.

    In particular, ``tag_pattern2re_pattern`` performs the following
    transformations on the given pattern:

        - Replace '.' with '[^<>{}]'
        - Remove any whitespace
        - Add extra parens around '<' and '>', to make '<' and '>' act
          like parentheses.  E.g., so that in '<NN>+', the '+' has scope
          over the entire '<NN>'; and so that in '<NN|IN>', the '|' has
          scope over 'NN' and 'IN', but not '<' or '>'.
        - Check to make sure the resulting pattern is valid.

    :type tag_pattern: str
    :param tag_pattern: The tag pattern to convert to a regular
        expression pattern.
    :raise ValueError: If ``tag_pattern`` is not a valid tag pattern.
        In particular, ``tag_pattern`` should not include braces; and it
        should not contain nested or mismatched angle-brackets.
    :rtype: str
    :return: A regular expression pattern corresponding to
        ``tag_pattern``.
    """
    cached = _cache.get(tag_pattern)
    if cached:
        return cached
    orig_tag_pattern = tag_pattern

    # Clean up the regular expression
    tag_pattern = (
        remove_spaces("", tag_pattern)
        .replace("<", "(<(")
        .replace(">", ")>)")
    )

    # Check the regular expression
    if not is_chunk_tag_pattern(tag_pattern):
        raise ValueError("Bad tag pattern: %r" % tag_pattern)

    # Replace "." with CHUNK_TAG_CHAR.
    # We have to do this after, since it adds {}[]<>s, which would
    # confuse CHUNK_TAG_PATTERN.
    # PRE doesn't have lookback assertions, so reverse twice, and do
    # the pattern backwards (with lookahead assertions).  This can be
    # made much cleaner once we can switch back to SRE.
    revrsd = "".join(reversed(tag_pattern))
    
    # (?! is a negative lookahead assertion
    # therefore this would match:
    # a dot not followed
    revrsd = re.sub(r"\.(?!\\(\\\\)*($|[^\\]))", ChunkString.CHUNK_TAG_CHAR_REVERSED, revrsd)
    tag_pattern = "".join(reversed(revrsd))

    _cache[orig_tag_pattern] = tag_pattern
    return tag_pattern


class RegexpChunkParser:
    """
    A regular expression based chunk parser.  ``RegexpChunkParser`` uses a
    sequence of "rules" to find chunks of a single type within a
    text.  The chunking of the text is encoded using a ``ChunkString``,
    and each rule acts by modifying the chunking in the
    ``ChunkString``.  The rules are all implemented using regular
    expression matching and substitution.

    The ``RegexpChunkRule`` class and its subclasses (``ChunkRule``,
    ``StripRule``, ``UnChunkRule``, ``MergeRule``, and ``SplitRule``)
    define the rules that are used by ``RegexpChunkParser``.  Each rule
    defines an ``apply()`` method, which modifies the chunking encoded
    by a given ``ChunkString``.

    :type _rules: list(RegexpChunkRule)
    :ivar _rules: The list of rules that should be applied to a text.
    :type _trace: int
    :ivar _trace: The default level of tracing.

    """

    def __init__(self, rules, chunk_label="NP", root_label="S", trace=0):
        """
        Construct a new ``RegexpChunkParser``.

        :type rules: list(RegexpChunkRule)
        :param rules: The sequence of rules that should be used to
            generate the chunking for a tagged text.
        :type chunk_label: str
        :param chunk_label: The node value that should be used for
            chunk subtrees.  This is typically a short string
            describing the type of information contained by the chunk,
            such as ``"NP"`` for base noun phrases.
        :type root_label: str
        :param root_label: The node value that should be used for the
            top node of the chunk structure.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        """
        self.rules = rules
        self._trace = trace
        self._chunk_label = chunk_label
        self._root_label = root_label

    def _trace_apply(self, chunkstr, verbose):
        """
        Apply each rule of this ``RegexpChunkParser`` to ``chunkstr``, in
        turn.  Generate trace output between each rule.  If ``verbose``
        is true, then generate verbose output.

        :type chunkstr: ChunkString
        :param chunkstr: The chunk string to which each rule should be
            applied.
        :type verbose: bool
        :param verbose: Whether output should be verbose.
        :rtype: None
        """
        print("# Input:")
        print(chunkstr)
        for rule in self.rules:
            rule.apply(chunkstr)
            if verbose:
                print("#", rule.descr + " (" + repr(rule) + "):")
            else:
                print("#", rule.descr + ":")
            print(chunkstr)

    def _notrace_apply(self, chunkstr):
        """
        Apply each rule of this ``RegexpChunkParser`` to ``chunkstr``, in
        turn.

        :param chunkstr: The chunk string to which each rule should be
            applied.
        :type chunkstr: ChunkString
        :rtype: None
        """

        for rule in self.rules:
            rule.apply(chunkstr)

    def parse(self, chunk_struct, trace=None):
        """
        :type chunk_struct: Tree
        :param chunk_struct: the chunk structure to be (further) chunked
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            highter will generate verbose tracing output.  This value
            overrides the trace level value that was given to the
            constructor.
        :rtype: Tree
        :return: a chunk structure that encodes the chunks in a given
            tagged sentence.  A chunk is a non-overlapping linguistic
            group, such as a noun phrase.  The set of chunks
            identified in the chunk structure depends on the rules
            used to define this ``RegexpChunkParser``.
        """
        if len(chunk_struct) == 0:
            print("Warning: parsing empty text")
            return Tree(self._root_label, [])

        try:
            chunk_struct.label()
        except AttributeError:
            chunk_struct = Tree(self._root_label, chunk_struct)

        # Use the default trace value?
        if trace is None:
            trace = self._trace

        chunkstr = ChunkString(chunk_struct)

        # Apply the sequence of rules to the chunkstring.
        verbose = trace > 1
        if trace:
            print("# Input:")
            print(chunkstr)
        for rule in self.rules:
            # rule.apply(chunkstr)
            chunkstr.apply_transform(rule._regexp, rule._repl)
            if verbose:
                print("#", rule.descr + " (" + repr(rule) + "):")
            elif trace:
                print("#", rule.descr + ":")

            if trace:
                print(chunkstr)

        # Use the chunkstring to create a chunk structure.
        return chunkstr.to_chunkstruct(self._chunk_label)

    def __repr__(self):
        return "<RegexpChunkParser with %d rules>" % len(self.rules)

    def __str__(self):
        s = "RegexpChunkParser with %d rules:\n" % len(self.rules)
        margin = 0
        for rule in self.rules:
            margin = max(margin, len(rule.descr))
        if margin < 35:
            formt = "    %" + repr(-(margin + 3)) + "s%s\n"
        else:
            formt = "    %s\n      %s\n"
        for rule in self.rules:
            s += formt % (rule.descr, repr(rule))
        return s[:-1]


class RegexpParser:
    """
    A grammar based chunk parser.  ``chunk.RegexpParser`` uses a set of
    regular expression patterns to specify the behavior of the parser.
    The chunking of the text is encoded using a ``ChunkString``, and
    each rule acts by modifying the chunking in the ``ChunkString``.
    The rules are all implemented using regular expression matching
    and substitution.

    A grammar contains one or more clauses in the following form::

     NP:
       {<DT|JJ>}          # chunk determiners and adjectives

    The patterns of a clause are executed in order.  An earlier
    pattern may introduce a chunk boundary that prevents a later
    pattern from executing.  Sometimes an individual pattern will
    match on multiple, overlapping extents of the input.  As with
    regular expression substitution more generally, the chunker will
    identify the first match possible, then continue looking for matches
    after this one has ended.

    The clauses of a grammar are also executed in order.  A cascaded
    chunk parser is one having more than one clause.  The maximum depth
    of a parse tree created by this chunk parser is the same as the
    number of clauses in the grammar.

    When tracing is turned on, the comment portion of a line is displayed
    each time the corresponding pattern is applied.

    :type _start: str
    :ivar _start: The start symbol of the grammar (the root node of
        resulting trees)
    :type _stages: int
    :ivar _stages: The list of parsing stages corresponding to the grammar
    """

    def __init__(self, grammar, root_label="S", loop=1, trace=0):
        """
        Create a new chunk parser, from the given start state
        and set of chunk patterns.

        :param grammar: The grammar, or a list of RegexpChunkParser objects
        :type grammar: str or list(RegexpChunkParser)
        :param root_label: The top node of the tree being created
        :type root_label: str or Nonterminal
        :param loop: The number of times to run through the patterns
        :type loop: int
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        """
        self._trace = trace
        self._stages = []
        self._grammar = grammar
        self._loop = loop

        if isinstance(grammar, str):
            self._read_grammar(grammar, root_label, trace)
        else:
            # Make sur the grammar looks like it has the right type:
            type_err = (
                "Expected string or list of RegexpChunkParsers " "for the grammar."
            )
            try:
                grammar = list(grammar)
            except BaseException as e:
                raise TypeError(type_err) from e
            for elt in grammar:
                if not isinstance(elt, RegexpChunkParser):
                    raise TypeError(type_err)
            self._stages = grammar

    def _read_grammar(
        self,
        grammar,
        root_label,
        trace,
        new_stage=re.compile(r"(?P<nonterminal>(\.|[^:])*)(:(?P<rule>.*))").match,
    ):
        """
        Helper function for __init__: read the grammar if it is a
        string.
        """
        rules = []
        lhs = None
        for line in grammar.split("\n"):
            line = line.strip()

            # New stage begins if there's an unescaped ':'
            m = new_stage(line)
            if m:
                # Record the stage that we just completed.
                self._add_stage(rules, lhs, root_label, trace)
                # Start a new stage.
                lhs = m.group("nonterminal").strip()
                rules = []
                line = m.group("rule").strip()

            # Skip blank & comment-only lines
            if line == "" or line.startswith("#"):
                continue

            # Add the rule
            rules.append(RegexpChunkRule.fromstring(line))

        # Record the final stage
        self._add_stage(rules, lhs, root_label, trace)

    def _add_stage(self, rules, lhs, root_label, trace):
        """
        Helper function for __init__: add a new stage to the parser.
        """
        if rules != []:
            if not lhs:
                raise ValueError("Expected stage marker (eg NP:)")
            parser = RegexpChunkParser(
                rules, chunk_label=lhs, root_label=root_label, trace=trace
            )
            self._stages.append(parser)

    def parse(self, chunk_struct, trace=None):
        """
        Apply the chunk parser to this input.

        :type chunk_struct: Tree
        :param chunk_struct: the chunk structure to be (further) chunked
            (this tree is modified, and is also returned)
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            highter will generate verbose tracing output.  This value
            overrides the trace level value that was given to the
            constructor.
        :return: the chunked output.
        :rtype: Tree
        """
        if trace is None:
            trace = self._trace
        for _ in range(self._loop):
            for parser in self._stages:
                chunk_struct = parser.parse(chunk_struct, trace=trace)
        return chunk_struct

    def __repr__(self):
        return "<chunk.RegexpParser with %d stages>" % len(self._stages)

    def __str__(self):
        s = "chunk.RegexpParser with %d stages:\n" % len(self._stages)
        for parser in self._stages:
            s += "%s\n" % parser
        return s[:-1]


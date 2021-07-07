Pygmars
========


https://github.com/nexB/pygmars

pygmars is a simple lexing and parsing library designed to craft lightweight
lexers and parsers using regular expressions.

pygmars allows you to craft simple lexers that recognizes words based on
regular expressions and identify sequences of words using lightweight grammars
to obtain a parse tree.

The lexing task transforms a sequence of words or strings (e.g. already split
in words) in an sequence of Token objects assigning a label to each word and
tracking position and line numbers. 

In particular, the lexing output is designed to be compatible with the output
of Pygments lexers. It becomes possible to build simple grammars on top of
existing Pygments lexers to perform lightweight parsing of the many (130+)
programming languages supported by Pygments.

(NOTE: Using Pygments lexers is WIP and NOT YET tested.)

The parsing task transforms a sequence of Tokens in a parse Tree where each node
in the tree is recognized and assigned a label. Parsing is using regular
expression-based grammar rules applied to the recognized Token sequences.

These rules are evaluated sequentially and not recursively: this keeps things
simple and works very well in practice. This approach and the rules syntax has
been battle-tested with NLTK from which pygmars is derived.


What about the name?
-----------------------

"pygmars" is a portmanteau of Pyg-ments and Gram-mars.


Origin
-------

This library is based on heavily modified, simplified and remixed original code
from NLTK regex POS tagger (renamed lexer) and regex chunker (renamed parser).

Users
-------

pygmars is used by ScanCode Toolkit for copyright detection and for
lightweight programming language parsing.


Why pygmars?
--------------

Why create this seemingly redundant library? Why not use NLTK directly?

- NLTK has a specific focus on NLP and lexing/tagging and parsing using regexes
  is a tiny part of its overall feature set. These are part of rich set of
  taggers and parsers and implement a common API. We do not have the need for
  these richer APIs and they make evolving the API and refactoring the code
  difficult.

- In particular NLTK POS tagging and chunking has been the engine used in
  ScanCode toolkit copyright and author detection and there are some
  improvements, simplification  and optimizations that would be difficult to
  implement in NLTK directly and unlikely to be accepted upstream. For instance,
  simplification of the code subset used for copyright detection enabled a big
  boost in performances. Improvements to track the Token lines and positions may
  not have been possible withing the NLTK API.

- Newer versions of NLTK have several extra required dependencies that we do
  not need. This is turn makes every tool heavier and complex when they only use
  this limited NLTK subset. By stripping unused NLTK code, we get a small and
  focused library with no dependencies.

- ScanCode toolkit also needs lightweight parsing of several programming
  languages to extract metadata (such as dependencies) from package manifests.
  Some parsers have been built by hand (such as gemfileparser), or use the
  Python ast module (for Python setup.py), or they use existing Pygments lexers
  as a base. A goal of this library is to be able to build lightweight parsers
  based on reusing Pygments lexer's output as an input. This is fairly different
  from NLP in terms of goals.


Theory of operations
---------------------

A ``pygmars.lex.Lexer`` creates a sequence of ``pygmars.Token`` objects
such as::

    Token(value="for" label="KEYWORD", start_line=12, pos=4)

where the label is a symbol name assigned to this token.

A Token is a terminal symbol and the grammar is composed of rules where the left
hand side is a label aka. a non-terminal symbol and  the right hand side is a
regular expression-like pattern over labels.

See https://en.wikipedia.org/wiki/Terminal_and_nonterminal_symbols

A ``pygmars.parse.Parser`` is built from a ``pygmars.parse.Grammmar`` and
calling its ``parse`` function transforms a sequence of Tokens in a
``pygmars.tree.Tree`` parse tree.

The grammar is composed of Rules and loaded from a text with one rule per line
such as::

    ASSIGNMENT: {<VARNAME> <EQUAL> <STRING|INT|FLOAT>} # variable assignment


Here the left hand side "ASSIGNMENT" label is produced when the right hand side
sequence of Token labels "<VARNAME> <EQUAL> <STRING|INT|FLOAT>" is matched.
"# variable assignment" is kept as a description for this rule.



License
--------

- SPDX-License-Identifier: Apache-2.0

Based on a substantially modified subset of the Natural Language Toolkit (NLTK)
http://nltk.org/

Copyright (c) nexB Inc. and others.
Copyright (C) NLTK Project
Pygmars
=======


https://github.com/nexB/pygmars

pygmars is a simple lexing and parsing library designed to craft lightweight
lexers and parsers using regular expressions.

pygmars allows you to craft simple lexers that recognizes words based on
regular expressions and identify sequences of words using lightweight grammars
to obtain a parse tree.

The lexing task transforms a sequence of words (a.k.a. tokens) pre-split or
tokenized in an output sequence of recognized tokens list of two-tuples
(token, token type). 

In particular, the lexing output is designed to be compatible with the output
of Pygments lexers. It becomes possible to build simple grammars on top of
existing Pygments lexers to perform lightweight parsing of the many (130+)
programming languages supported by Pygments.

(NOTE: Using Pygments lexers is WIP and NOT YET tested.)

The parsing task transforms a sequence of lexed and recognized tokens in a parse
tree. Parsing is using regex-based grammar rules applied to the recognized
token/types tuples. These rules are evaluated sequentially and not recursively:
this keeps things simple and works very well in practice. This approach and the
rules syntax has been battle-tested with NLTK.


Name?
------

"pygmars" is a portmanteau of Pyg-ments and Gram-mars.


Origin
-------

This library is based on heavily modified and remixed original code from NLTK
regex POS tagger (renamed lexer) and regex chunker (renamed parser).

Users
-------

pygmars is used by ScanCode Toolkit for copyright detection and soon for
programming language parsing.


Why?
----

Why create this seemingly redundant library? Why not use NLTK directly?

- NLTK has a specific focus on NLP and lexing/tagging and chunking/parsing using
  regexes is a tiny part of its overall feature set. These are part of rich set
  of taggers and parsers and implement a common API. We do not have the need for
  these richer APIs and they make evolving the API and refactoring the code
  difficult.

- In particular NLTK POS tagging and chunking has been the engine used in
  ScanCode toolkit copyright and author detection and there are some
  improvements, simplification  and optimizations that would be difficult to
  implement in NLTK directly and unlikely to be accepted upstream.
  Simplification of the code subset used for copyright detection enabled a big
  boost in performances. Future improvements to track the tokens lines and
  positions may not have been possible withing the NLTK API.

- The newer versions of NLTK include several new required dependencies and we do
  not need any of these. This is turn makes every tool heavier and complex when
  they only use this limited NLTK subset. By stripping unused NLTK code, we get
  a small and focused library with no dependencies.

- ScanCode toolkit also needs lightweight parsing of several programming
  languages to extract data (such as dependencies) from package manifests.
  Some parsers have been built by hand (such as gemfileparser), or use the
  Python ast module (for Python setup.py), or they use existing Pygments lexers
  as a base. A goal of this library is to be able to build lightweight parsers
  reusing Pygments lexer's output as an input. This is fairly different from NLP
  in terms of goals.



License
-------

- SPDX-License-Identifier: Apache-2.0

Based on substantially modified Natural Language Toolkit (NLTK) http://nltk.org/

Copyright (c) nexB Inc. and others.
Copyright (C) NLTK Project
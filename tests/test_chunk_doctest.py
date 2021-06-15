# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# Copyright (C) 2001-2020 NLTK Project
# SPDX-License-Identifier: Apache-2.0
# URL: <http://nltk.org/>
"""
==========
 Chunking
==========

    >>> from pygmars.parse import tag_pattern2re_pattern
    >>> from pygmars.parse import ChunkRule, ChunkString, RegexpChunkRule
    >>> from pygmars.parse import RegexpChunkParser, RegexpParser
    >>> from pygmars.tree import Tree
    >>> tagged_text = "[ The/DT cat/NN ] sat/VBD on/IN [ the/DT mat/NN ] [ the/DT dog/NN ] chewed/VBD ./."

    >>> unchunked_text = "[ The/DT cat/NN ] sat/VBD on/IN [ the/DT mat/NN ] [ the/DT dog/NN ] chewed/VBD ./."
    >>> unchunked_text = unchunked_text.replace('[', '').replace(']', '').split()
    >>> unchunked_text = [tuple(x.split('/')) for x in unchunked_text]
    >>> plain_text = ' '.join([x[0] for x in unchunked_text])
    >>> tag_pattern = "<DT>?<JJ>*<NN.*>"
    >>> regexp_pattern = tag_pattern2re_pattern(tag_pattern)
    >>> regexp_pattern
    '(<(DT)>)?(<(JJ)>)*(<(NN[^\\\\{\\\\}<>]*)>)'

Construct some new chunking rules.

    >>> chunk_rule = ChunkRule("<.*>+", "Chunk everything")

Create a series of chunk parsers, successively more complex.

    >>> chunk_parser = RegexpChunkParser([chunk_rule], chunk_label='NP')
    >>> chunked_text = chunk_parser.parse(unchunked_text)
    >>> print(chunked_text)
    (S
      (NP
        The/DT
        cat/NN
        sat/VBD
        on/IN
        the/DT
        mat/NN
        the/DT
        dog/NN
        chewed/VBD
        ./.))


Printing parsers:

    >>> print(repr(chunk_parser))
    <RegexpChunkParser with 1 rules>
    >>> print(chunk_parser)
    RegexpChunkParser with 1 rules:
        Chunk everything   <ChunkRule: '<.*>+'>

Regression Tests
~~~~~~~~~~~~~~~~


ChunkString
-----------
ChunkString can be built from a tree of tagged tuples, a tree of
trees, or a mixed list of both:

    >>> t1 = Tree('S', [('w%d' % i, 't%d' % i) for i in range(10)])
    >>> t2 = Tree('S', [Tree('t0', []), Tree('t1', ['c1'])])
    >>> t3 = Tree('S', [('w0', 't0'), Tree('t1', ['c1'])])
    >>> ChunkString(t1)
    <ChunkString: '<t0><t1><t2><t3><t4><t5><t6><t7><t8><t9>'>
    >>> ChunkString(t2)
    <ChunkString: '<t0><t1>'>
    >>> ChunkString(t3)
    <ChunkString: '<t0><t1>'>

Other values generate an error:

    >>> ChunkString(Tree('S', ['x']))
    Traceback (most recent call last):
      . . .
    ValueError: chunk structures must contain tagged tokens or trees

The `str()` for a chunk string adds spaces to it, which makes it line
up with `str()` output for other chunk strings over the same
underlying input.

    >>> cs = ChunkString(t1)
    >>> print(cs)
     <t0>  <t1>  <t2>  <t3>  <t4>  <t5>  <t6>  <t7>  <t8>  <t9>
    >>> cs.xform('<t3>', '{<t3>}')
    >>> print(cs)
     <t0>  <t1>  <t2> {<t3>} <t4>  <t5>  <t6>  <t7>  <t8>  <t9>

The `_verify()` method makes sure that our transforms don't corrupt
the chunk string.  By setting debug_level=2, `_verify()` will be
called at the end of every call to `xform`.

    >>> cs = ChunkString(t1, debug_level=3)

    >>> # tag not marked with <...>:
    >>> cs.xform('<t3>', 't3')
    Traceback (most recent call last):
      . . .
    ValueError: Transformation generated invalid chunkstring:
      <t0><t1><t2>t3<t4><t5><t6><t7><t8><t9>

    >>> # brackets not balanced:
    >>> cs.xform('<t3>', '{<t3>')
    Traceback (most recent call last):
      . . .
    ValueError: Transformation generated invalid chunkstring:
      <t0><t1><t2>{<t3><t4><t5><t6><t7><t8><t9>

    >>> # nested brackets:
    >>> cs.xform('<t3><t4><t5>', '{<t3>{<t4>}<t5>}')
    Traceback (most recent call last):
      . . .
    ValueError: Transformation generated invalid chunkstring:
      <t0><t1><t2>{<t3>{<t4>}<t5>}<t6><t7><t8><t9>

    >>> # modified tags:
    >>> cs.xform('<t3>', '<t9>')
    Traceback (most recent call last):
      . . .
    ValueError: Transformation generated invalid chunkstring: tag changed

    >>> # added tags:
    >>> cs.xform('<t9>', '<t9><t10>')
    Traceback (most recent call last):
      . . .
    ValueError: Transformation generated invalid chunkstring: tag changed

Chunking Rules
--------------

Test the different rule constructors & __repr__ methods:
    >>> import re
    >>> r1 = RegexpChunkRule('<a|b>'+ChunkString.IN_STRIP_PATTERN,
    ...                      '{<a|b>}', 'chunk <a> and <b>')
    >>> r2 = RegexpChunkRule(re.compile('<a|b>'+ChunkString.IN_STRIP_PATTERN),
    ...                      '{<a|b>}', 'chunk <a> and <b>')
    >>> r3 = ChunkRule('<a|b>', 'chunk <a> and <b>')
    >>> for rule in r1, r2, r3:
    ...     print(rule)
    <RegexpChunkRule: '<a|b>(?=[^\\\\}]*(\\\\{|$))'->'{<a|b>}'>
    <RegexpChunkRule: '<a|b>(?=[^\\\\}]*(\\\\{|$))'->'{<a|b>}'>
    <ChunkRule: '<a|b>'>

`tag_pattern2re_pattern()` complains if the tag pattern looks problematic:

    >>> tag_pattern2re_pattern('{}')
    Traceback (most recent call last):
      . . .
    ValueError: Bad tag pattern: '{}'

RegexpChunkParser
-----------------

A warning is printed when parsing an empty sentence:

    >>> parser = RegexpChunkParser([ChunkRule('<a>', '')])
    >>> parser.parse(Tree('S', []))
    Warning: parsing empty text
    Tree('S', [])

RegexpParser
------------

    >>> parser = RegexpParser('''
    ... NP: {<DT>? <JJ>* <NN>*} # NP
    ... P: {<IN>}           # Preposition
    ... V: {<V.*>}          # Verb
    ... PP: {<P> <NP>}      # PP -> P NP
    ... VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
    ... ''')
    >>> print(repr(parser))
    <chunk.RegexpParser with 5 stages>
    >>> print(parser)
    chunk.RegexpParser with 5 stages:
    RegexpChunkParser with 1 rules:
        NP   <ChunkRule: '<DT>? <JJ>* <NN>*'>
    RegexpChunkParser with 1 rules:
        Preposition   <ChunkRule: '<IN>'>
    RegexpChunkParser with 1 rules:
        Verb   <ChunkRule: '<V.*>'>
    RegexpChunkParser with 1 rules:
        PP -> P NP   <ChunkRule: '<P> <NP>'>
    RegexpChunkParser with 1 rules:
        VP -> V (NP|PP)*   <ChunkRule: '<V> <NP|PP>*'>
    >>> print(parser.parse(unchunked_text, trace=True))
    # Input:
     <DT>  <NN>  <VBD>  <IN>  <DT>  <NN>  <DT>  <NN>  <VBD>  <.>
    # NP:
    {<DT>  <NN>} <VBD>  <IN> {<DT>  <NN>}{<DT>  <NN>} <VBD>  <.>
    # Input:
     <NP>  <VBD>  <IN>  <NP>  <NP>  <VBD>  <.>
    # Preposition:
     <NP>  <VBD> {<IN>} <NP>  <NP>  <VBD>  <.>
    # Input:
     <NP>  <VBD>  <P>  <NP>  <NP>  <VBD>  <.>
    # Verb:
     <NP> {<VBD>} <P>  <NP>  <NP> {<VBD>} <.>
    # Input:
     <NP>  <V>  <P>  <NP>  <NP>  <V>  <.>
    # PP -> P NP:
     <NP>  <V> {<P>  <NP>} <NP>  <V>  <.>
    # Input:
     <NP>  <V>  <PP>  <NP>  <V>  <.>
    # VP -> V (NP|PP)*:
     <NP> {<V>  <PP>  <NP>}{<V>} <.>
    (S
      (NP The/DT cat/NN)
      (VP
        (V sat/VBD)
        (PP (P on/IN) (NP the/DT mat/NN))
        (NP the/DT dog/NN))
      (VP (V chewed/VBD))
      ./.)

Illegal patterns give an error message:

    >>> print(RegexpParser('X: {<foo>} {<bar>}'))
    Traceback (most recent call last):
      . . .
    ValueError: Illegal chunk pattern: {<foo>} {<bar>}

"""
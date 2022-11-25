# SPDX-License-Identifier: Apache-2.0
# Copyright (C) nexB Inc. and others
# Copyright (C) 2001-2020 NLTK Project
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/pygmars for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.

# Originally based on: Natural Language Toolkit
# substantially modified for use in ScanCode-toolkit
#
# Natural Language Toolkit (NLTK)
# URL: <http://nltk.org/>
"""
==========
 Parsing
==========
    >>> from functools import partial
    >>> import re
    >>> from pygmars import Token
    >>> from pygmars.parse import label_pattern_to_regex
    >>> from pygmars.parse import ParseString
    >>> from pygmars.parse import Rule, Parser
    >>> from pygmars.tree import Tree

    >>> tagged_text = "The/DT cat/NN sat/VBD on/IN  the/DT mat/NN the/DT dog/NN chewed/VBD".split()
    >>> tagged_text = [x.split('/') for x in tagged_text]
    >>> tokens = [Token(v, t) for v, t in tagged_text]
    >>> plain_text = ' '.join([x[0] for x in tagged_text])
    >>> tag_pattern = "<DT>?<JJ>*<NN.*>"
    >>> regexp_pattern = label_pattern_to_regex(tag_pattern)
    >>> regexp_pattern
    '(?:<(?:DT)>)?(?:<(?:JJ)>)*(?:<(?:NN[^\\\\{\\\\}<>]*)>)'

Create a Rule and parse something:

    >>> pattern = "<.*>+"
    >>> description = "Parse everything"
    >>> rule = Rule(pattern, label='NP', description=description)
    >>> parsed = rule.parse(tokens)
    >>> parsed.pprint(margin=100)
    (label='ROOT', children=(
      (label='NP', children=(
        (label='DT', value='The')
        (label='NN', value='cat')
        (label='VBD', value='sat')
        (label='IN', value='on')
        (label='DT', value='the')
        (label='NN', value='mat')
        (label='DT', value='the')
        (label='NN', value='dog')
        (label='VBD', value='chewed')))

Printing Rule:

    >>> print(repr(rule))
    <Rule: <.*>+ / NP # Parse everything>

    >>> print(rule)
    <Rule: <.*>+ / NP # Parse everything>


ParseString
-----------

ParseString can be built from a tree of Tokens, a tree of
trees, or a mixed list of both:

    >>> t1 = Tree('S', [Token(f'w{i}', f't{i}') for i in range(10)])
    >>> t2 = Tree('S', [Tree('t0', []), Tree('t1', ['c1'])])
    >>> t3 = Tree('S', [Token('w0', 't0'), Tree('t1', ['c1'])])
    >>> ParseString(t1)
    <ParseString: '<T0><T1><T2><T3><T4><T5><T6><T7><T8><T9>'>
    >>> ParseString(t2)
    <ParseString: '<t0><t1>'>
    >>> ParseString(t3)
    <ParseString: '<T0><t1>'>

Other values generate an error:

    >>> ParseString(Tree('S', ['x']))
    Traceback (most recent call last):
      . . .
    AttributeError: 'str' object has no attribute 'label'

The `str()` for a parse string adds spaces to it, which makes it line
up with `str()` output for other parse strings over the same
underlying input.

    >>> cs = ParseString(t1)
    >>> print(cs)
     <T0>  <T1>  <T2>  <T3>  <T4>  <T5>  <T6>  <T7>  <T8>  <T9>
    >>> cs.apply_transform(partial(re.compile('<T3>').sub, '{<T3>}'))
    '<T0><T1><T2>{<T3>}<T4><T5><T6><T7><T8><T9>'
    >>> print(cs)
     <T0>  <T1>  <T2> {<T3>} <T4>  <T5>  <T6>  <T7>  <T8>  <T9>

The `validate()` method makes sure that the parsing does not corrupt
the parse string.  By setting validate=True, `validate()` will be
called at the end of every call to `apply_transform`.

    >>> cs = ParseString(t1, validate=True)

    Tag not marked with <...>:
    >>> cs.apply_transform(partial(re.compile('<T3>').sub, 'T3'))
    Traceback (most recent call last):
      . . .
    ValueError: Invalid parse:
      <T0><T1><T2>T3<T4><T5><T6><T7><T8><T9>

    Brackets not balanced:
    >>> cs.apply_transform(partial(re.compile('<T3>').sub, '{<T3>'))
    Traceback (most recent call last):
      . . .
    ValueError: Invalid parse: unbalanced or nested curly braces:
      <T0><T1><T2>{<T3><T4><T5><T6><T7><T8><T9>

    Nested brackets:
    >>> cs.apply_transform(partial(re.compile('<T3><T4><T5>').sub, '{<T3>{<T4>}<T5>}'))
    Traceback (most recent call last):
      . . .
    ValueError: Invalid parse: unbalanced or nested curly braces:
      <T0><T1><T2>{<T3>{<T4>}<T5>}<T6><T7><T8><T9>

    Transformer that modifies tags:
    >>> cs.apply_transform(partial(re.compile('<T3>').sub, '<T9>'))
    Traceback (most recent call last):
      . . .
    ValueError: Invalid parse: tag changed:
      <T0><T1><T2><T9><T4><T5><T6><T7><T8><T9>

    Transformer that adds tags:
    >>> cs.apply_transform(partial(re.compile('<T9>').sub, '<T9><T10>'))
    Traceback (most recent call last):
      . . .
    ValueError: Invalid parse: tag changed:
      <T0><T1><T2><T3><T4><T5><T6><T7><T8><T9><T10>


Parsing Rules
--------------

Test the different rule constructors and __repr__ methods:

    >>> import re
    >>> r1 = Rule(
    ...    pattern='<a|b>', label='R1',
    ...    description='chunk <a> and <b>'
    ... )
    >>> r2 = Rule('<a|b>', label='R2', description='chunk <a> and <b>')
    >>> r3 = Rule('<a|b|c>', label='R3', description='chunk <a> and <b>')
    >>> for rule in r1, r2, r3:
    ...     print(rule)
    <Rule: <a|b> / R1 # chunk <a> and <b>>
    <Rule: <a|b> / R2 # chunk <a> and <b>>
    <Rule: <a|b|c> / R3 # chunk <a> and <b>>

`label_pattern_to_regex()` complains if the label pattern looks problematic:

    >>> label_pattern_to_regex('{}')
    Traceback (most recent call last):
      . . .
    ValueError: Bad label pattern: '{}'

Rule
-----------------

An exception is raise when parsing an empty tree:

    >>> parser = Rule('<a>', '')
    >>> parser.parse(Tree('S', []))
    Traceback (most recent call last):
      . . .
    Exception: Warning: parsing empty tree: (label='S', children=())


Parser
------------
    >>> grammar = '''
    ... NP: <DT>? <JJ>* <NN>* # NP
    ... P: <IN>               # Preposition
    ... V: <V.*>              # Verb
    ... PP: <P> <NP>          # PP -> P NP
    ... VP: <V> <NP|PP>*      # VP -> V (NP|PP)*
    ... '''


    >>> parser = Parser(grammar)
    >>> print(repr(parser))
    <Parser with 5 rules>
    >>> print(parser)
    <Parser with  5 rules>
    <Rule: <DT>? <JJ>* <NN>* / NP # NP>
    <Rule: <IN> / P # Preposition>
    <Rule: <V.*> / V # Verb>
    <Rule: <P> <NP> / PP # PP -> P NP>
    <Rule: <V> <NP|PP>* / VP # VP -> V (NP|PP)*>

    >>> parser = Parser(grammar, trace=True)
    >>> print("parse tree:", parser.parse(tokens))
    parse: loop# 0
    parse: parse_rule: <Rule: <DT>? <JJ>* <NN>* / NP # NP>
    parse: parse_tree len: 9
    Rule.parse transformer regex: (?P<group>(?:<(?:DT)>)?(?:<(?:JJ)>)*(?:<(?:NN)>)*)
    <BLANKLINE>
    # Input parsed as label: 'NP'
      before: ' <DT>  <NN>  <VBD>  <IN>  <DT>  <NN>  <DT>  <NN>  <VBD>'
      after : '{<DT>  <NN>} <VBD>  <IN> {<DT>  <NN>}{<DT>  <NN>} <VBD>'
    parse: parse_tree new len: 6
    parse: parse_rule: <Rule: <IN> / P # Preposition>
    parse: parse_tree len: 6
    Rule.parse transformer regex: (?P<group>(?:<(?:IN)>))
    <BLANKLINE>
    # Input parsed as label: 'P'
      before: ' <NP>  <VBD>  <IN>  <NP>  <NP>  <VBD>'
      after : ' <NP>  <VBD> {<IN>} <NP>  <NP>  <VBD>'
    parse: parse_tree new len: 6
    parse: parse_rule: <Rule: <V.*> / V # Verb>
    parse: parse_tree len: 6
    Rule.parse transformer regex: (?P<group>(?:<(?:V[^\{\}<>]*)>))
    <BLANKLINE>
    # Input parsed as label: 'V'
      before: ' <NP>  <VBD>  <P>  <NP>  <NP>  <VBD>'
      after : ' <NP> {<VBD>} <P>  <NP>  <NP> {<VBD>}'
    parse: parse_tree new len: 6
    parse: parse_rule: <Rule: <P> <NP> / PP # PP -> P NP>
    parse: parse_tree len: 6
    Rule.parse transformer regex: (?P<group>(?:<(?:P)>)(?:<(?:NP)>))
    <BLANKLINE>
    # Input parsed as label: 'PP'
      before: ' <NP>  <V>  <P>  <NP>  <NP>  <V>'
      after : ' <NP>  <V> {<P>  <NP>} <NP>  <V>'
    parse: parse_tree new len: 5
    parse: parse_rule: <Rule: <V> <NP|PP>* / VP # VP -> V (NP|PP)*>
    parse: parse_tree len: 5
    Rule.parse transformer regex: (?P<group>(?:<(?:V)>)(?:<(?:NP|PP)>)*)
    <BLANKLINE>
    # Input parsed as label: 'VP'
      before: ' <NP>  <V>  <PP>  <NP>  <V>'
      after : ' <NP> {<V>  <PP>  <NP>}{<V>}'
    parse: parse_tree new len: 3
    parse tree: (label='ROOT', children=(
      (label='NP', children=(
        (label='DT', value='The')
        (label='NN', value='cat'))
      (label='VP', children=(
        (label='V', children=(
          (label='VBD', value='sat'))
        (label='PP', children=(
          (label='P', children=(
            (label='IN', value='on'))
          (label='NP', children=(
            (label='DT', value='the')
            (label='NN', value='mat')))
        (label='NP', children=(
          (label='DT', value='the')
          (label='NN', value='dog')))
      (label='VP', children=(
        (label='V', children=(
          (label='VBD', value='chewed'))))


Illegal patterns give an error message:

    >>> print(Parser('X: {<foo>} {<bar>}'))
    Traceback (most recent call last):
      . . .
    ValueError: Bad label pattern: '(?:<(?:foo)>)}{(?:<(?:bar)>)'
"""

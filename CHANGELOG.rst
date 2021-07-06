================
Changelog
================

Version 0.7.0 (next)
--------------------------


Version 0.6.0
----------------	

- Use raw strings for regexes to avoid deprecation warnings
- Significantly refactor code: top level are lex, parse and tree
- Parsing is now restructured with only Parser and Rule, plus ParseString
  used internally for parsing.
- Code has been streamlined and simplified removing all parts that are not used
  As an unintended side effect, there is about a 25% speedup
- Lexer accepts regex and also callabale that behave like re.match
- Add minimal support to reuse Pygments lexer outputs as a Token to feed the
  Parser.


Version 0.5.0
---------------

- Initial version derived from NLTK HEAD at a5690ee

# coding/llm_sanitizer/util.py
from __future__ import annotations
import re
from textwrap import dedent

_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<code>.*?)(?:```|$)", re.DOTALL
)

def extract_code_block(raw: str, prefer_langs=("python","py","cpp","c++","java"), strict: bool = False) -> str:
    """
    Return the best-looking fenced code block; else the raw text.

    Args:
        raw:
            The raw text possibly containing fenced code blocks.
        prefer_langs:
            Languages to prefer when selecting a code block.
        strict:
            If True, only return code from preferred languages. If no matching
            code block found, strip markdown fences and return what looks like code.

    Returns:
        The extracted code block, or the raw text if no fenced blocks found.

    Examples:
        >>> extract_code_block("Here is some code:\\n```python\\ndef foo(): pass\\n```")
        'def foo(): pass'
        >>> extract_code_block("No code blocks here.")
        'No code blocks here.'
        >>> extract_code_block("Multiple:\\n```java\\nclass A {}\\n```\\n```python\\ndef f(): pass\\n```")
        'def f(): pass'
    """
    matches = list(_FENCE_RE.finditer(raw))
    if not matches:
        return strip_triple_quotes(raw)

    def score(m):
        lang = (m.group("lang") or "").lower()
        pref = 1 if lang in prefer_langs else 0
        return (pref, len(m.group("code")))

    if strict:
        # Only consider blocks from preferred languages
        preferred_matches = [m for m in matches if (m.group("lang") or "").lower() in prefer_langs]
        if preferred_matches:
            m = max(preferred_matches, key=lambda m: len(m.group("code")))
            return m.group("code").strip()
        # No preferred language found - try unlabeled code blocks
        unlabeled = [m for m in matches if not m.group("lang")]
        if unlabeled:
            m = max(unlabeled, key=lambda m: len(m.group("code")))
            return m.group("code").strip()
        # Fall back to stripping all markdown and returning what's left
        return strip_triple_quotes(raw)

    m = max(matches, key=score)
    return m.group("code").strip()

def strip_triple_quotes(s: str) -> str:
    """
    If the string is wrapped in triple quotes, strip them.
    
    Args:
        s:
            The input string.
            
    Returns:
        The string with triple quotes removed if they were present.
        
    Examples:
        >>> strip_triple_quotes('\"\"\"def foo(): pass\"\"\"')
        'def foo(): pass'
        >>> strip_triple_quotes("'''def foo(): pass'''")
        'def foo(): pass'
        >>> strip_triple_quotes('def foo(): pass')
        'def foo(): pass'
    """
    s = s.strip()
    if s.startswith('"""') and s.endswith('"""'):
        return s[3:-3].strip()
    if s.startswith("'''") and s.endswith("'''"):
        return s[3:-3].strip()
    return s

def normalize_whitespace(code: str) -> str:
    """
    Normalize line endings to LF, dedent, and strip leading/trailing whitespace.
    
    arguments:
        code:
            The input code string.

    returns:
        The normalized code string.

    examples:
        >>> normalize_whitespace("  def foo():\\n    pass  ")
        'def foo():\\n    pass'
        >>> normalize_whitespace("def foo():\\r\\n    pass\\r")
        'def foo():\\n    pass'
    """
    code = code.replace("\r\n","\n").replace("\r","\n")
    code = dedent(code).strip()
    return code

def maybe_black(code: str) -> str:
    """
    If Black is installed, format; otherwise return as-is.
    
    arguments:
        code:
            The input Python code string.
            
    returns:
        The formatted code string if Black is available; else the original code.
        
    examples:
        >>> maybe_black("def foo():pass")
        'def foo():\\n    pass\\n'
    """
    try:
        import black 
        return black.format_str(code, mode=black.FileMode())
    except Exception:
        return code

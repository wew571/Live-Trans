"""Read setup configuration before third-party dependencies are installed."""

from __future__ import annotations

import os
from pathlib import Path
import re


_ENTRY = re.compile(
    r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"""
    r"""(?:"((?:\\.|[^"\\])*)"|'((?:\\'|[^'])*)'|([^'"].*?|))\s*(?:\s+#.*)?\s*$"""
)
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_DOUBLE_ESCAPE = re.compile(r"""\\[\\'"abfnrtv]""")
_ESCAPED = {letter: chr(code) for letter, code in zip("abfnrtv", (7, 8, 12, 10, 13, 9, 11))}


def read_setup_environment(root: Path) -> dict[str, str]:
    """Read simple .env assignments with process environment taking precedence."""
    values = {}
    try:
        lines = (root / ".env").read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        lines = ()
    for line in lines:
        match = _ENTRY.fullmatch(line)
        if match is None:
            continue
        key, double, single, unquoted = match.groups()
        if double is not None:
            value = _DOUBLE_ESCAPE.sub(lambda escape: _ESCAPED.get(escape[0][1], escape[0][1]), double)
        elif single is not None:
            value = single.replace("\\'", "'").replace("\\\\", "\\")
        else:
            value = unquoted.strip()
        values[key] = _REFERENCE.sub(
            lambda ref: os.environ.get(ref[1], values.get(ref[1], ref[2] or "")), value
        )

    values.update({key: os.environ.get(key, value) for key, value in values.items()})
    values.update(os.environ)
    return values

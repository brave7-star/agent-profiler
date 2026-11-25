"""Which language the reports come out in.

Two audiences, neither of them secondary. The findings were written in Chinese because the
the owner reads Chinese and a report nobody reads is a failed report; the repository is public
because the project's largest unknown — `n = 1` — can only be closed by someone else running
`verify` on their own machine. An English-only tool loses the first audience and a
Chinese-only one loses the second.

So the language is resolved, not fixed:

    --lang en|zh            explicit, wins over everything
    AGENT_PROFILER_LANG     for a wrapper script or a CI job
    LC_ALL / LC_MESSAGES / LANG   the environment's own answer
    en                      the fallback, because `LANG=C` says nothing about a reader

Translations are written as `tr(zh=..., en=...)` at the point of use rather than as keys into
a catalog. With 349 strings a key namespace is a second thing to keep correct, and a key says
nothing at the call site about what the other language says. Side by side, a reviewer sees
both, drift is visible in the diff, and an f-string stays an f-string — no placeholder
rewriting, so no chance of the two languages disagreeing about their own fields.

The cost is that both strings are built even though one is discarded. These render a report
once per process, so that is not a cost worth designing around.
"""

from __future__ import annotations

import os

EN = "en"
ZH = "zh"
LANGUAGES = (EN, ZH)
# English, not Chinese: a machine whose environment says nothing is likeliest not the the owner's.
FALLBACK = EN

_ENVIRONMENT = ("AGENT_PROFILER_LANG", "LC_ALL", "LC_MESSAGES", "LANG")

_current: str | None = None


def _normalize(value: str | None) -> str | None:
    """A language tag reduced to one of ours, or `None` if it names neither.

    Accepts what the environment actually holds — `zh_CN.UTF-8`, `en-GB`, `zh`. `C` and
    `POSIX` normalize to `None` rather than to English, so that the fallback is reached
    deliberately instead of by a coincidence of spelling.
    """
    if not value:
        return None
    tag = str(value).split(".")[0].split("@")[0].replace("_", "-").lower()
    if tag.startswith("zh"):
        return ZH
    if tag.startswith("en"):
        return EN
    return None


def resolve(explicit: str | None = None) -> str:
    """The language to use, from the explicit flag down to the fallback."""
    for candidate in (explicit, *(os.environ.get(name) for name in _ENVIRONMENT)):
        found = _normalize(candidate)
        if found:
            return found
    return FALLBACK


def set_lang(lang: str | None) -> str:
    """Pin the language for this process. Returns what it resolved to.

    Module state rather than a parameter on every renderer: a run produces one report, and
    threading a language through ninety call sites would be ninety chances to drop it. The
    tests pin it explicitly so that a developer's own locale cannot change what they assert.
    """
    global _current
    _current = resolve(lang)
    return _current


def current() -> str:
    """The pinned language, resolving from the environment on first use."""
    global _current
    if _current is None:
        _current = resolve()
    return _current


def tr(*, en: str, zh: str) -> str:
    """One string in the current language.

    Keyword-only on purpose: `tr(a, b)` would silently swap the languages, and a swapped
    pair reads as correct in whichever language the reviewer does not speak.
    """
    return zh if current() == ZH else en


def count(n, singular: str, plural: str | None = None) -> str:
    """`1 file` / `3 files`, for **English** strings only.

    Chinese measure words do not inflect, so a mechanical translation produces this error
    in one direction only — `1 files`, `1 components are declared`. Three of those shipped
    in the first bilingual report, which is why this exists rather than 44 hand-written
    conditionals.

    `plural` is needed whenever the plural is irregular or a verb travels with the noun:

        count(n, "file")                        → 1 file      / 3 files
        count(n, "copy", "copies")              → 1 copy      / 3 copies
        count(n, "component is", "components are")

    Numbers are formatted with thousands separators, since every count in this project's
    output already is.
    """
    word = singular if n == 1 else (plural if plural is not None else f"{singular}s")
    return f"{n:,} {word}" if isinstance(n, int) else f"{n} {word}"

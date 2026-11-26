"""Regression suite for language resolution.

The failure mode worth testing is not a missing translation — that is visible. It is the
language being decided by something other than what the caller asked for: a developer's own
locale leaking into a test, `LANG=C` reading as a deliberate choice of English, or the two
halves of a `tr()` pair getting swapped so the output is fluent and wrong.
"""

import os
import re
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import i18n


@contextmanager
def _env(**values):
    """A controlled environment, so no test reads the developer's own locale."""
    saved = {k: os.environ.get(k) for k in i18n._ENVIRONMENT}
    for name in i18n._ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ.update({k: v for k, v in values.items() if v is not None})
    try:
        yield
    finally:
        for name in i18n._ENVIRONMENT:
            os.environ.pop(name, None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def test_the_explicit_flag_beats_every_environment_variable():
    with _env(AGENT_PROFILER_LANG="zh", LC_ALL="zh_CN.UTF-8", LANG="zh_CN.UTF-8"):
        assert i18n.resolve("en") == i18n.EN


def test_the_environment_is_read_in_priority_order():
    with _env(AGENT_PROFILER_LANG="en", LC_ALL="zh_CN.UTF-8"):
        assert i18n.resolve() == i18n.EN
    with _env(LC_ALL="zh_CN.UTF-8", LANG="en_US.UTF-8"):
        assert i18n.resolve() == i18n.ZH
    with _env(LANG="zh_CN.UTF-8"):
        assert i18n.resolve() == i18n.ZH


def test_the_tags_the_environment_actually_holds_are_understood():
    for value in ("zh", "zh_CN.UTF-8", "zh-Hans", "zh_TW", "ZH_cn"):
        with _env(LANG=value):
            assert i18n.resolve() == i18n.ZH, value
    for value in ("en", "en_US.UTF-8", "en-GB", "en_AU@foo"):
        with _env(LANG=value):
            assert i18n.resolve() == i18n.EN, value


def test_a_locale_that_names_no_language_falls_back_rather_than_matching():
    """`C` and `POSIX` say nothing about a reader. They must reach the fallback through the
    fallback, not by being mistaken for an English tag — otherwise adding a third language
    would silently inherit every `C` locale."""
    for value in ("C", "POSIX", "", "de_DE.UTF-8"):
        with _env(LANG=value):
            assert i18n.resolve() == i18n.FALLBACK, value
        assert i18n._normalize(value) is None, value


def test_nothing_in_the_environment_means_english():
    """A machine that says nothing is likeliest not the owner's."""
    with _env():
        assert i18n.resolve() == i18n.EN


def test_tr_refuses_positional_arguments():
    """`tr(a, b)` would swap the languages, and a swapped pair reads as correct in whichever
    language the reviewer does not speak."""
    try:
        i18n.tr("hello", "你好")
    except TypeError:
        pass
    else:
        raise AssertionError("tr() accepted positional arguments")


def test_count_inflects_only_at_one():
    """`1 files` shipped three times in the first bilingual report. Chinese measure words
    do not inflect, so this error appears in the English direction only."""
    assert i18n.count(1, "file") == "1 file"
    assert i18n.count(0, "file") == "0 files"
    assert i18n.count(2, "file") == "2 files"
    # Irregular plurals and a verb travelling with the noun both need the explicit form.
    assert i18n.count(1, "copy", "copies") == "1 copy"
    assert i18n.count(3, "copy", "copies") == "3 copies"
    assert i18n.count(1, "component is", "components are") == "1 component is"
    assert i18n.count(4, "component is", "components are") == "4 components are"
    # Thousands separators, because every other count in the output has them.
    assert i18n.count(1234, "call") == "1,234 calls"
    # A non-integer is a magnitude, not a count, and is left alone.
    assert i18n.count(1.5, "second") == "1.5 seconds"


def test_set_lang_pins_the_choice_against_the_environment():
    saved = i18n._current
    try:
        with _env(LANG="zh_CN.UTF-8"):
            i18n.set_lang("en")
            assert i18n.current() == i18n.EN
            assert i18n.tr(en="x", zh="y") == "x"
            i18n.set_lang(None)          # None re-reads the environment
            assert i18n.current() == i18n.ZH
            assert i18n.tr(en="x", zh="y") == "y"
    finally:
        i18n._current = saved


# -- the translated documents -------------------------------------------------------------
#
# A translation rots silently: nothing breaks when one side gains a section, and a reader who
# picked the other language is quietly served a different document. These tests compare
# structure rather than prose, which is the part that can be checked without a translator.
#
# The schema notes are here for a specific reason: they are the part another machine can
# refute, so a reader who verifies in Chinese needs them in Chinese. `docs/principles.md` is
# deliberately *not* a pair — both languages live in one file, because the list is short
# enough that two files would only add a way to drift. `skill/` is not translated either: it
# is loaded into an agent's context, where a second copy doubles the token cost and no human
# reads it.

REPO = Path(__file__).resolve().parent.parent
READMES = {"en": REPO / "README.md", "zh": REPO / "README.zh-CN.md"}
PAIRS = {
    "README": READMES,
    "trace-schema": {"en": REPO / "docs/dogfood/trace-schema.md",
                     "zh": REPO / "docs/dogfood/trace-schema.zh-CN.md"},
    "codex-schema": {"en": REPO / "docs/dogfood/codex-schema.md",
                     "zh": REPO / "docs/dogfood/codex-schema.zh-CN.md"},
}


def _headings(text):
    """Heading levels in order, without the text, so wording is free to differ."""
    return [len(m.group(1)) for m in re.finditer(r"^(#{1,6}) +\S", text, re.M)]


def _code_blocks(text):
    return re.findall(r"```[a-z]*\n(.*?)```", text, re.S)


def test_both_languages_are_published():
    for name, pair in PAIRS.items():
        for lang, path in pair.items():
            assert path.is_file(), f"{name} is missing its {lang} half: {path}"


def test_each_document_links_to_its_other_language():
    """The switcher is the whole feature: a translation nobody can navigate to is not one."""
    for name, pair in PAIRS.items():
        en, zh = (pair[k].read_text(encoding="utf-8") for k in ("en", "zh"))
        assert pair["zh"].name in en.split("\n")[0], name
        assert pair["en"].name in zh.split("\n")[0], name
        # English is the default, so its line names the other language as the link.
        assert en.split("\n")[0].startswith("**English**"), name
        assert zh.split("\n")[0].endswith("**简体中文**"), name


def test_each_pair_has_the_same_section_structure():
    """Catches the actual failure: a section added to one side and not the other."""
    for name, pair in PAIRS.items():
        en, zh = (_headings(pair[k].read_text(encoding="utf-8")) for k in ("en", "zh"))
        assert en == zh, f"{name} heading structure diverged: en={en} zh={zh}"


def test_commands_are_identical_in_both():
    """Prose is translated; a command is not. A localized flag name would be a bug that
    only the reader of one language ever hits. The schema notes carry record shapes in their
    blocks rather than commands, and those must not be translated either."""
    for name, pair in PAIRS.items():
        blocks = {k: _code_blocks(pair[k].read_text(encoding="utf-8")) for k in ("en", "zh")}
        assert len(blocks["en"]) == len(blocks["zh"]), name
        for left, right in zip(blocks["en"], blocks["zh"]):
            # A trailing comment inside a block is prose and may be translated; the command
            # before it may not.
            def commands(block):
                return [line.split("#")[0].strip() for line in block.splitlines()]
            assert commands(left) == commands(right), \
                f"{name} commands differ:\n{left}\n---\n{right}"


def _published_counts():
    counts = {}
    for key, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        counts[key] = set(re.findall(r"\b(\d{3})(?= regression tests| 个回归测试)", text))
    return counts


def test_the_test_count_agrees_between_the_two():
    """One number, two places, and it has already been stale once in a single file."""
    counts = _published_counts()
    assert counts["en"] and counts["en"] == counts["zh"], counts


def test_the_published_test_count_is_the_real_one():
    """Two files agreeing proves only that they were edited together. The number is a claim
    about the repository, and it was stale by a dozen while both halves agreed."""
    real = sum(len(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.M))
               for path in sorted((REPO / "tests").glob("test_*.py")))
    assert real > 50, real  # the glob found the suite, not an empty directory
    assert _published_counts()["en"] == {str(real)}, (real, _published_counts())


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} language-resolution tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The published repository must not describe the machine it was written on.

A tool that profiles a private environment gets published; the environment does not. The
tool is the general mechanism, and a figure induced from one corpus is not part of that
mechanism — it is a fact about one person's week.

This file exists because the earlier sweeps for this were one-off scripts, each given a list
of figures already known to be in the text. Such a scan can only find what someone
remembered to look for, and it found nothing the second time precisely because it was
looking for the first time's numbers. These rules are the *pattern classes* instead, and
they run with the suite.

Two of them work from a declared set rather than a shape, because no shape tells a real
identifier from an invented one. Adding an entry is then a deliberate act, which is the
point: the friction belongs on the side of publishing a new identifier.

This file is excluded from its own sweep — it has to contain examples of what it forbids.
"""

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = "tests/test_privacy.py"

# Identifiers that may appear in tracked files. Every one is invented: a repeated or
# sequential hex ladder, which no harness would ever emit.
SYNTHETIC_UUIDS = {
    "11111111-2222-3333-4444-555555555555",
    "77777777-6666-5555-4444-333333333333",
    "99999999-8888-7777-6666-555555555555",
    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "dddddddd-eeee-ffff-0000-111111111111",
    "abcdef01-2345-6789-abcd-ef0123456789",
}

# A home directory in a fixture path is a placeholder, never a name.
PLACEHOLDER_HOMES = {"u", "h"}

# Hosts a tracked file may name. Every entry is reserved for examples by RFC 2606, so none
# of them resolves to anything.
#
# This rule was missing, and the sweep had no shape that would have caught what it is for.
# A CLI can be pointed at a relay instead of at the model provider, and that endpoint sits
# in the harness config of the machine being profiled. Writing one into a comment while
# explaining a measurement defect would publish which service this environment's requests
# go through — the same class of fact as a session count, and arguably worse, because it
# names a third party too. The mechanism is what belongs here: *a relay may answer two
# identical requests differently*, which holds for any relay and names none.
ALLOWED_HOSTS = {"x.invalid", "api.invalid", "example.invalid"}

ENDPOINT = re.compile(r"\bhttps?://([A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)")

MAGNITUDE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_COUNTED_NOUN = (r"sessions?|records?|transcripts?|rollouts?|files?"
                 r"|bytes?|tokens?|chars?|characters?")
COUNTED_CORPUS = re.compile(
    # Up to two words may sit between the number and the noun. A rule requiring them to be
    # adjacent missed a real sentence of the form "N of N Codex sessions", which is the whole
    # reason this is a shape and not a list of phrasings.
    rf"\b\d{{3,}}(?:\s+(?:of|\/)?\s*[A-Za-z][\w.-]*){{0,2}}\s+(?:{_COUNTED_NOUN})\b"
    rf"|\b\d{{3,}}\s*(?:{_COUNTED_NOUN})\b"
    # Chinese measure words. The English half of a bilingual string was generalised while
    # the Chinese half kept the real figures, so a rule in one language only is half a rule.
    r"|\d{3,}\s*(?:个|条|次|份)?\s*(?:会话|记录|轮次|文件|字符|令牌)"
    r"|\d{2,}(?:\.\d+)?\s*(?:亿|万)\s*(?:个)?\s*(?:token|输入|字符)")
MESSAGE_ID = re.compile(r"\bmsg_[A-Za-z0-9]{6,}\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
# An eight-hex session-id prefix. Required to carry both a digit and a letter, so an
# ordinary English word of eight letters (`deadbeef` aside) cannot match.
HEX_ID = re.compile(r"(?<![0-9a-zA-Z_./-])(?=[0-9a-f]{8}(?![0-9a-zA-Z]))"
                    r"(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8}")
HOME_PATH = re.compile(r"(?:/Users|/home)/([A-Za-z0-9._-]+)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOSTNAME = re.compile(r"\b[A-Za-z0-9-]+(?:MacBook|iMac|Mac-mini)[A-Za-z0-9-]*\b")


def _tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    files = []
    for name in out:
        if name == SELF:
            continue
        try:
            files.append((name, (ROOT / name).read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    assert len(files) > 20, f"the sweep read only {len(files)} files"
    return files


def _prose():
    return [(n, t) for n, t in _tracked() if n.endswith(".md")]


def _python_prose():
    """Docstrings and comments, which are prose that happens to live in a `.py` file.

    This is where most of the figures were: a docstring explaining why a test exists is the
    natural place to write down what was observed. Code is excluded on purpose — a fixture
    literal is invented input, and no pattern separates `287_000` chosen for a test from a
    real measurement, so that distinction is made by a reader and not by this sweep.
    """
    out = []
    for name, text in _tracked():
        if not name.endswith(".py"):
            continue
        chunks = []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    chunks.append(doc)
        try:
            chunks += [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
                       if t.type == tokenize.COMMENT]
        except (tokenize.TokenError, IndentationError):
            pass
        if chunks:
            out.append((name, "\n".join(chunks)))
    assert len(out) > 10, f"the sweep read prose from only {len(out)} modules"
    return out


def _hits(files, pattern):
    found = {}
    for name, text in files:
        seen = sorted(set(pattern.findall(text)))
        if seen:
            found[name] = seen
    return found


def test_no_prose_states_the_size_of_a_corpus():
    """A session count and a byte total were the opening line of a schema document.

    It reads as the evidence for everything below it, which is exactly why it was there —
    and it says how much work one person did in a month. The claim a reader needs is
    whether a property has a counterexample; `verify` re-checks that against their own
    corpus, which is a better answer than a number from someone else's.
    """
    assert not _hits(_prose(), MAGNITUDE)
    assert not _hits(_prose(), COUNTED_CORPUS)


def test_no_comment_or_docstring_states_a_corpus_magnitude():
    """"parsing all N rollouts", "empty in N of N sessions", "+N tokens" — the reason a
    piece of code exists, written down with the figure that prompted it. That is where most
    of them were, because a docstring explaining why a test exists is the natural place to
    record what was observed.

    Only the counted-corpus rule runs here, not the thousands-separator one: a comment
    legitimately discusses a synthetic component stack, or a constant that was once a round
    guess, and neither is a fact about anyone's machine.

    What this cannot catch is a small inventory count — a skill count, a baseline size —
    because no shape separates those from any other small number. Those were removed by
    reading, and this rule does not pretend to cover them.
    """
    assert not _hits(_python_prose(), COUNTED_CORPUS)


def test_no_document_quotes_a_real_identifier():
    """Session ids, request uuids and message ids were used as citations — "verified on",
    followed by an id prefix. They prove nothing to a reader who cannot open that file, and
    they name one machine's sessions. Documents use `<uuid A>` and identify a session only
    by its role in the argument."""
    prose = _prose()
    assert not _hits(prose, MESSAGE_ID)
    assert not _hits(prose, UUID)
    assert not _hits(prose, HEX_ID)


def test_every_identifier_in_the_repository_is_a_declared_invention():
    """Fixtures need identifier-shaped strings, and no pattern can tell an invented one from
    a captured one. So they are listed, and a new one has to be added on purpose."""
    undeclared = {name: [u for u in found if u not in SYNTHETIC_UUIDS]
                  for name, found in _hits(_tracked(), UUID).items()}
    undeclared = {k: v for k, v in undeclared.items() if v}
    assert not undeclared, undeclared


def test_no_path_in_the_repository_names_a_person():
    """A fixture home directory is a placeholder. A real one turns every path built from it
    into a byline, and absolute paths are what measurement artifacts join on."""
    named = {}
    for name, text in _tracked():
        bad = sorted({m for m in HOME_PATH.findall(text) if m not in PLACEHOLDER_HOMES})
        if bad:
            named[name] = bad
    assert not named, named


def test_no_tracked_file_names_a_service_this_environment_talks_to():
    """A configured endpoint is a fact about the measured machine, not about the tool.

    The gap this closes: the sweep had rules for identifiers, paths, magnitudes and
    contact details, and none of them matched a URL. A diagnosis that turns on the presence
    of a relay in front of the model provider is exactly the kind of note a developer
    writes down with the endpoint beside it, and that endpoint names both the machine's
    configuration and a third party.
    """
    named = {}
    for name, text in _tracked():
        bad = sorted({h for h in ENDPOINT.findall(text) if h not in ALLOWED_HOSTS})
        if bad:
            named[name] = bad
    assert not named, named


def test_nothing_in_the_repository_carries_contact_details_or_a_hostname():
    """The first history of this project put a real name and address in 15 commits before
    anyone looked. A machine name arrives the same way, through a pasted shell prompt."""
    assert not _hits(_tracked(), EMAIL)
    assert not _hits(_tracked(), HOSTNAME)


def test_each_rule_catches_the_leak_it_was_written_for():
    """Without this, a pattern that silently stopped matching would read as a clean repo.

    Every string here is invented to the shape of a leak that really did ship, never copied
    from one.
    """
    planted = [
        (MAGNITUDE, "Corpus: 123,456,789 bytes"),
        (COUNTED_CORPUS, "across 444 sessions"),
        (COUNTED_CORPUS, "2222 records examined"),
        # The gap that let one through: words between the number and the noun.
        (COUNTED_CORPUS, "zero for 555 of 666 Codex sessions"),
        (COUNTED_CORPUS, "777 compressed rollout files"),
        # And the gap that let the next one through: the same claim in Chinese.
        (COUNTED_CORPUS, "888 个会话里有 555 个报零"),
        (COUNTED_CORPUS, "藏掉 44.4 亿输入 token"),
        (MESSAGE_ID, "msg_ZxQwVuTsRp carried the discrepancy"),
        (UUID, "api-request.id = 0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"),
        (HEX_ID, "verified on 0a1b2c3d, which is a session"),
        (EMAIL, "someone@example.com"),
        (HOSTNAME, "user@somebodysMacBook-Pro ~ %"),
        # The endpoint class, which had no rule at all until a diagnosis turned on one.
        (ENDPOINT, "base_url = https://relay.madeupvendor.test/v1"),
        (ENDPOINT, "ANTHROPIC_BASE_URL=http://gateway.madeupvendor.test"),
    ]
    for pattern, sample in planted:
        assert pattern.search(sample), (pattern.pattern, sample)

    # The endpoint rule is a declared set, not a shape, so the extraction has to be checked
    # too: a pattern that matched but produced a host no allow-list could contain would
    # fail every file, and one that produced nothing would pass every file.
    assert ENDPOINT.findall("see https://relay.madeupvendor.test/v1 for the route") == \
        ["relay.madeupvendor.test"]
    assert not {h for h in ENDPOINT.findall("https://example.invalid/mcp")
                if h not in ALLOWED_HOSTS}

    # And the rules must not fire on the shapes the repository legitimately contains.
    for pattern, sample in [
        (COUNTED_CORPUS, "428 regression tests"),
        (HEX_ID, "commit 13ccfe7"),
        (HEX_ID, "/private/var/folders/abcdef12"),
        (MAGNITUDE, "a four-digit figure"),
        (HOSTNAME, "a MacBook is not a hostname on its own"),
    ]:
        assert not pattern.search(sample), (pattern.pattern, sample)


def test_this_file_does_not_quote_the_leaks_it_forbids():
    """Found by sweeping the rebuilt history: the sweep's own prose had cited the real
    figures and a real session-id prefix as illustrations, so the one file exempt from the
    rules was reintroducing what they existed to remove. Fixing the docstrings and checking
    only those then left the same figure in a *comment* one line away — which is why this
    reads the whole file's prose, by the same definition `_python_prose` uses.

    An example has to be invented, which costs nothing: the rules match a *shape*, so a
    made-up string of that shape exercises them exactly as well as a captured one. The
    exemption therefore covers only the samples planted in the rule-coverage test below.
    """
    text = (ROOT / SELF).read_text(encoding="utf-8")
    chunks = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                chunks.append(doc)
    chunks += [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
               if t.type == tokenize.COMMENT]
    prose = [(SELF, "\n".join(chunks))]
    for pattern in (MAGNITUDE, COUNTED_CORPUS, MESSAGE_ID, UUID, HEX_ID, EMAIL, HOSTNAME):
        assert not _hits(prose, pattern), (pattern.pattern, _hits(prose, pattern))
    # The endpoint rule applies to this file's prose through its allow-list, the same way
    # it applies everywhere else: the samples it plants live in code, not in a docstring.
    assert not {h for _, text in prose for h in ENDPOINT.findall(text)
                if h not in ALLOWED_HOSTS}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} privacy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

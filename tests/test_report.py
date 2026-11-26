"""Regression suite for the readable report.

A summary is where discipline goes to die: prose invites confident sentences, and
"unproven" quietly becomes "fine". So most of these tests assert what the report *refuses*
to say, and that its caveats are generated from the data rather than written once and left
to rot.
"""

import json
import os
import pathlib
import re
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import report as rp
from agent_profiler import score as sc
from agent_profiler.cli import main as cli_main
from agent_profiler.report import _confirmed_blind_spots, report_root
from agent_profiler import i18n

# Pinned so a developer's own locale cannot change what these assert; the zh strings
# are the ones this suite was written against. English rendering is covered separately.
i18n.set_lang("zh")


@contextmanager
def _lang(tag):
    """Render in one language, then put the suite's pinned language back."""
    previous = i18n.current()
    i18n.set_lang(tag)
    try:
        yield
    finally:
        i18n.set_lang(previous)

SESSION = "abcdef01-2345-6789-abcd-ef0123456789"


def _rec(**kw):
    base = {"sessionId": SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, blocks):
    return _rec(type="assistant", timestamp="2026-11-19T10:00:01.000Z", message={
        "id": msg_id, "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0, "output_tokens": 5},
        "content": blocks})


def _trace(tools=(), *, attachments=()):
    records = [_rec(type="user", timestamp="2026-11-19T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for item in attachments:
        records.append(_rec(type="attachment", timestamp="2026-11-19T10:00:00.500Z",
                            attachment=item))
    for i, tool in enumerate(tools):
        name, tool_input = tool if isinstance(tool, tuple) else (tool, {"q": "x"})
        records.append(_assistant(f"m{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": name, "input": tool_input}]))
        records.append(_rec(type="user", timestamp="2026-11-19T10:00:02.000Z",
                            message={"role": "user", "content": [
                                {"type": "tool_result", "tool_use_id": f"t{i}",
                                 "is_error": False, "content": "x"}]}))
    return records


@contextmanager
def _env(records, servers=None, *, skills=()):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        (root / ".mcp.json").write_text(json.dumps({"mcpServers": servers or {}}),
                                        encoding="utf-8")
        for name in skills:
            folder = root / ".claude" / "skills" / name
            folder.mkdir(parents=True)
            (folder / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n", encoding="utf-8")
        key = str(root.resolve()).replace("/", "-").replace(".", "-")
        proj = home / ".claude" / "projects" / key
        proj.mkdir(parents=True)
        (proj / f"{SESSION}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            yield root
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _evaluation(server="s", schema=3331, usd=31.43, sessions=1):
    return {"corpus": {"sessions": sessions, "responses": 1}, "evaluations": [{
        "server": server, "status": "measured", "calls_observed": 60,
        "measurement": {"ok": True, "schema_tokens": schema},
        "projection": {"schema_tokens": schema, "total_tokens": 17354510,
                       "estimated_usd": usd,
                       # Verbatim from the artifact, which `evaluate` writes in English:
                       # a quoted assumption reads the same here as where it came from.
                       "assumptions": ["the schema sits in the cached prefix",
                                       "published rates were used"]},
    }]}


# -- what the report refuses to say ------------------------------------------------------


def test_no_recommendation_or_counterfactual_language():
    """The standing project rule, enforced on prose where it is easiest to break."""
    with _env(_trace([("mcp__s__go", {"q": "x"})] * 3), {"s": {"command": "x"}}) as root:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(_evaluation(), fh)
            path = fh.name
        try:
            data = report_root(str(root), evaluation_path=path)
        finally:
            os.unlink(path)
        text = data["text"]
        for banned in ("该删", "建议删除", "可以省", "能省", "没用", "浪费", "该留"):
            assert banned not in text, banned
        blob = json.dumps(data, ensure_ascii=False)
        for banned in ("would save", "should be removed", "useless", "wasted"):
            assert banned not in blob.lower(), banned


BANNED_ZH = ("该删", "建议删除", "可以省", "能省", "没用", "浪费", "该留")
BANNED_EN = ("would save", "should be removed", "useless", "wasted", "not worth keeping",
             "safe to delete", "can be dropped")


def _scorable_component(name="AGENTS.md (~)"):
    """A component with every scored link present — the state real data rarely reaches.

    Built here because the banned-word guard above never saw this shape: with nothing
    scorable, `_verdict_lines` and the `keep` finding both return empty, so the two
    sections most likely to contain recommendation language were the two the project's
    most important test never rendered.
    """
    link = lambda status, **kw: {"status": status, "blocker": None, "evidence": "x",
                                 "strength": "observed", "value": None, **kw}
    return {
        "kind": "agents_md", "name": name, "harness": "codex", "harnesses": ["codex"],
        "scope": "user", "path": "/x/AGENTS.md",
        "links": {
            "installed": link(sc.PROVEN), "loaded": link(sc.PROVEN),
            "used": link(sc.PROVEN, value=12),
            "useful": link(sc.PROVEN, value=0.9,
                           rules_proven=["subject_ends_with_reference"]),
            "cost": link(sc.PARTIAL, value=450, unit="tokens_per_request",
                         strength="estimated"),
            "recommendation": link(sc.UNPROVEN),
        },
        "coverage": {"scorable": True},
        "next_measurement": None,
    }


def _with_real_sensitivity(component):
    """Scored the way `report_root` scores it, so the fixture cannot drift from the code."""
    component["sensitivity"] = sc.sensitivity(component)
    return component


def test_the_no_recommendation_guard_covers_both_languages_and_the_verdict_section():
    """The guard that has caught this project five times had two holes after the report
    became bilingual, and both were in the flattering direction.

    First, it ran only in Chinese, so 324 new English strings were never checked — and one
    of them did carry a banned substring (`...not measured-and-useless`), legitimate as a
    negation and indistinguishable from a claim to a grep.

    Second, its fixture has nothing scorable, so `_verdict_lines` and the `keep` finding
    returned empty. The one section that exists to publish a conclusion was the one section
    the guard never rendered.
    """
    component = _with_real_sensitivity(_scorable_component())

    for lang, banned in (("zh", BANNED_ZH), ("en", BANNED_EN)):
        with _lang(lang):
            verdicts = rp._verdicts([component], "balanced")
            assert verdicts, "the fixture must actually reach a verdict, or this proves nothing"
            rendered = "\n".join(rp._verdict_lines(verdicts, "balanced"))
            findings = rp._findings({
                "verdicts": verdicts, "diverged_duplicates": [], "reliability": [],
                "next_steps": [], "always_on": [], "components": [component],
                "sessions": 10})
            keep = next(f for f in findings if f["action"] == "keep")
            surfaces = [rendered, keep["headline"], keep["detail"], keep["next"],
                        *keep["evidence"]]
            for surface in surfaces:
                for word in banned:
                    assert word not in surface.lower(), (lang, word, surface[:120])

    # And the verdict must still say the one thing it is allowed to say.
    with _lang("en"):
        text = "\n".join(rp._verdict_lines(rp._verdicts([component], "balanced"), "balanced"))
        assert "keep" in text.lower()
    with _lang("zh"):
        text = "\n".join(rp._verdict_lines(rp._verdicts([component], "balanced"), "balanced"))
        assert "保留" in text


def test_no_english_count_reads_as_one_plural():
    """`1 files`, `1 components are declared`, `1 items have a measured cost`. Three of
    these shipped in the first bilingual report, and the class is invisible from the
    Chinese side: measure words do not inflect, so every such bug lives in the English
    direction only and never shows up in the language the author reads.

    Every count is forced to exactly 1 and the rendered English is grepped for a digit `1`
    followed by a plural noun or plural verb. Magnitudes are exempt — a formatted token
    count is a measured quantity, not an enumeration — so the check targets the countable
    nouns the report actually enumerates.
    """
    one = {
        "diverged_duplicates": [
            {"kind": "skills", "name": "solo", "harness": "codex", "days_apart": 1,
             "stale": "/b/solo",
             "copies": [{"path": "/a/solo", "modified": "2026-11-20", "bytes": 1},
                        {"path": "/b/solo", "modified": "2026-11-19", "bytes": 2}]}],
        "reliability": [{"name": "s", "calls": 1, "errors": 1, "error_rate": 1.0,
                         "suspicious": True}],
        "always_on": [{"name": "k", "kind": "skills", "tokens": 1, "calls": 1,
                       "ambiguous": False, "harness_label": "codex",
                       "also": [{"harness": "claude_code", "calls": 1}]}],
        "next_steps": [{"link": "cost", "method": "stage 6 paired probe",
                        "estimated_usd": 0.3, "components": ["one"], "note": "n"}],
        "components": [], "verdicts": [], "sessions": 1,
    }
    with _lang("en"):
        rendered = "\n".join([
            *rp._duplicate_lines(one["diverged_duplicates"]),
            *rp._always_on_lines(one["always_on"], one["sessions"]),
            *rp._reliability_lines(one["reliability"]),
            *rp._installed_lines({"files": 1, "bytes": 1, "by_kind": {"claude_md": 1,
                                                                     "skills": 1},
                                  "entities": {"skills": 1}, "entity_files": {"skills": 1},
                                  "bundled_files": 1}),
            *rp._unanswerable(
                [{"kind": "skills", "name": "p", "links": {
                    "loaded": {"status": sc.UNPROVEN, "blocker": rp.NO_EVENT_EMITTED},
                    "used": {"status": sc.PARTIAL, "blocker": None},
                    "useful": {"status": sc.UNPROVEN, "blocker": None}}},
                 {"kind": "mcp", "name": "q", "links": {
                     "loaded": {"status": sc.PROVEN, "blocker": None},
                     "used": {"status": sc.UNPROVEN, "blocker": "harness_not_traced"},
                     "useful": {"status": sc.UNPROVEN, "blocker": None}}}],
                {"useful": {"proven": 0, "partial": 0, "unproven": 2},
                 "used": {"proven": 0, "partial": 1, "unproven": 1},
                 "installed": {"proven": 2, "partial": 0, "unproven": 0},
                 "loaded": {"proven": 1, "partial": 0, "unproven": 1},
                 "cost": {"proven": 0, "partial": 0, "unproven": 2}}),
            *[f["headline"] + " " + f["detail"] + " " + " ".join(f["evidence"])
              for f in rp._findings(one)],
        ])

    countable = ("files", "sessions", "calls", "skills", "components", "items",
                 "responses", "failures", "times", "days", "copies", "rules")
    verbs = ("are", "have", "ships", "were")
    bad = re.findall(r"(?<![\d,.])1 (" + "|".join(countable + verbs) + r")\b", rendered)
    assert not bad, f"singular count rendered with a plural: {sorted(set(bad))}"
    # And the fixture really did render something, or this proves nothing.
    assert len(rendered) > 600 and "1 " in rendered


def test_the_headline_says_what_is_missing_rather_than_a_score():
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        assert data["stack_score"]["withheld"] is True
        headline = data["headline"]
        assert "有用" in headline, "the missing link must be named in the headline"
        assert "总分被扣住" in headline


def test_estimated_and_observed_are_never_merged():
    """A measured schema size and a projected dollar figure are different kinds of claim."""
    with _env(_trace([("mcp__s__go", {"q": "x"})]), {"s": {"command": "x"}}) as root:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(_evaluation(), fh)
            path = fh.name
        try:
            data = report_root(str(root), evaluation_path=path)
        finally:
            os.unlink(path)
        strengths = {e["strength"] for e in data["cost"]}
        assert strengths == {"observed", "estimated"}
        projected = next(e for e in data["cost"] if e["strength"] == "estimated")
        assert projected["detail"], "an estimate must carry its assumptions"
        assert "实测" in data["text"] and "推算" in data["text"]


def test_an_inconclusive_measurement_is_shown_rather_than_dropped():
    """Real case: the server's token had expired, so the probes isolated nothing. Skipping
    the entry made the report read as if cost had never been measured at all, which hides
    both the attempt and the reason it failed."""
    evaluation = {"corpus": {"sessions": 1, "responses": 1}, "evaluations": [{
        "server": "s", "status": "measurement_inconclusive", "calls_observed": 0,
        "reason": "the two arms differ by -6 tokens, which is inside the 50-token noise band",
    }]}
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(evaluation, fh)
            path = fh.name
        try:
            data = report_root(str(root), evaluation_path=path)
        finally:
            os.unlink(path)
        line = next(e for e in data["cost"] if "schema" in e["label"])
        assert line["value"] == "测了，但没测出来"
        assert "noise band" in line["detail"]
        # And no dollar figure may appear for it.
        assert not any(e["strength"] == "estimated" for e in data["cost"])


def test_a_projection_from_another_corpus_is_not_claimed_as_this_one():
    """Real contradiction caught on a moved project: the report showed "0 sessions
    collected" and a 17-million-token projection as if both described this path."""
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(_evaluation(sessions=94), fh)
            path = fh.name
        try:
            data = report_root(str(root), evaluation_path=path)
        finally:
            os.unlink(path)
        projected = next(e for e in data["cost"] if e["strength"] == "estimated")
        assert "另一批语料" in projected["label"]
        assert "不是同一批" in projected["detail"]
        assert "94 个会话" in projected["detail"]


def test_an_empty_corpus_is_not_reported_as_zero_spend():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        home, moved = Path(h), Path(d) / "demo"
        moved.mkdir()
        (moved / ".mcp.json").write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        (home / ".claude" / "projects").mkdir(parents=True)
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            data = report_root(str(moved))
            spend = data["cost"][0]
            assert spend["value"] == "无法统计"
            assert "不是「花了 0」" in spend["detail"]
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_caveats_are_generated_from_the_data():
    """Hand-written caveats rot. These must follow the real link coverage."""
    with _env(_trace(["Read"]), {"s": {"command": "x"}}, skills=("demo",)) as root:
        data = report_root(str(root))
        joined = " ".join(data["unanswerable"])
        assert "没有任何组件的「有用」得到证明" in joined
        # A skill never explicitly invoked must be described as unproven, not as unused.
        assert "不能当事实" in joined


# -- the always-on list: the only actionable output that does not need `useful` -----------


def _priced(name, kind, harness, tokens, calls, *, used_status=None, evidence="x"):
    from agent_profiler import score as sc
    return {
        "kind": kind, "name": name, "harness": harness, "harnesses": [harness],
        "links": {
            "used": sc._link(used_status or sc.PARTIAL, evidence=evidence,
                             strength="observed", value=calls),
            "cost": sc._link(sc.PARTIAL, evidence="x", strength="observed", value=tokens,
                             unit="tokens_per_request", harness=harness),
        },
    }


def test_the_always_on_list_is_ordered_by_calls_and_keeps_one_row_per_listed_name():
    """Two facts per row and nothing else: measured per-request cost, and counted calls.

    A skill installed under two roots is two components and one listed skill. Two rows
    would double it, which is the mistake the cost floor made with the same data.
    """
    from agent_profiler.report import _always_on_but_uncalled

    rows = _always_on_but_uncalled([
        _priced("search-server", "mcp", "claude_code", 470, 159),
        _priced("second-skill", "skills", "claude_code", 143, 9),
        _priced("second-skill", "skills", "shared", 141, 9),
        _priced("demo-skill", "skills", "claude_code", 160, 0),
    ])

    assert [r["name"] for r in rows] == ["demo-skill", "second-skill", "search-server"]
    # One row for the duplicated name, carrying the lower figure, as a bound should.
    assert len([r for r in rows if r["name"] == "second-skill"]) == 1
    assert next(r for r in rows if r["name"] == "second-skill")["tokens"] == 141


def test_an_instruction_file_is_not_listed_as_having_been_called_zero_times():
    """An instruction file is never invoked, so a call count for it is a category error,
    not a low number — and `0 次` beside a 709-token file would read as an accusation."""
    from agent_profiler.report import _always_on_but_uncalled

    rows = _always_on_but_uncalled([
        _priced("CLAUDE.md (~/.claude)", "claude_md", "claude_code", 709, 2),
        _priced("demo-skill", "skills", "claude_code", 160, 0),
    ])
    assert [r["name"] for r in rows] == ["demo-skill"]


def test_a_component_used_through_another_harness_says_so_on_the_row():
    """`demo-skill` is never explicitly invoked under Claude Code *and* its Codex copy was
    read 3 times. A row showing only the zero would be true and misleading."""
    from agent_profiler.report import _always_on_but_uncalled, _always_on_lines

    rows = _always_on_but_uncalled([
        _priced("demo-skill", "skills", "claude_code", 160, 0),
        # The Codex copy has usage but no measured cost: `--plugin-dir` is a Claude flag.
        {"kind": "skills", "name": "demo-skill", "harness": "codex", "harnesses": ["codex"],
         "links": {"used": {"status": "partial", "evidence": "read 3 times", "value": 3},
                   "cost": {"status": "unproven", "evidence": "x", "value": None,
                            "unit": None}}},
    ])

    assert len(rows) == 1 and rows[0]["also"][0]["calls"] == 3
    text = "\n".join(_always_on_lines(rows, 337))
    assert "不是没在用" in text
    # The limit travels with the list, not as a footnote somewhere else.
    assert "0 次不代表它没起作用" in text


def test_the_always_on_list_never_recommends_removal():
    """It is deliberately weaker than a verdict: two facts, and the judgment left to the
    reader. The evidence for "remove" does not exist anywhere in this project."""
    from agent_profiler.report import _always_on_but_uncalled, _always_on_lines

    rows = _always_on_but_uncalled([_priced("demo-skill", "skills", "claude_code", 160, 0)])
    text = "\n".join(_always_on_lines(rows, 337))
    # The same words the whole report bans, including in a negation: this prose has to fit
    # the existing blunt guard rather than the guard being loosened to admit it.
    for word in ("该删", "建议删除", "可以省", "能省", "没用", "浪费", "该留"):
        assert word not in text, word


def test_confirmed_blind_spots_are_carried_with_their_evidence():
    """A reported blind spot keeps its evidence; one that does not apply is set aside.

    This previously asserted that every blind spot is always reported, which is how the
    report came to tell an environment with no hooks that it could not tell whether its
    hooks had run.
    """
    with _env(_trace(["Read"]), skills=("demo",)) as root:
        data = report_root(str(root))
        reported = {spot["what"] for spot in data["blind_spots"]}
        aside = {spot["what"] for spot in data["blind_spots_not_applicable"]}
        # Every known spot is accounted for, in exactly one of the two lists.
        assert reported | aside == {s["what"] for s in _confirmed_blind_spots()}
        assert not reported & aside
        # This fixture has skills and no hooks, so the split follows the inventory.
        assert "被描述匹配中的 skill，正文是不是随后才进上下文" in reported
        assert "历史会话里 hook 是否执行过" in aside
        for spot in data["blind_spots"]:
            assert spot["evidence"], spot["what"]
            assert spot["evidence"] in data["text"]
        for spot in data["blind_spots_not_applicable"]:
            assert spot["why"] in data["text"]


# -- the plain answer, and the one finding that needs no `useful` link -------------------


def _dup(name, harness, path, body, mtime):
    return {"kind": "skills", "name": name, "harness": harness, "path": path,
            "links": {}, "_body": body, "_mtime": mtime}


@contextmanager
def _installed(tmp: Path, copies: list[dict]):
    """Write real files, because this finding is about bytes and mtimes on disk."""
    for copy in copies:
        target = tmp / copy["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(copy["_body"], encoding="utf-8")
        os.utime(target, (copy["_mtime"], copy["_mtime"]))
        copy["path"] = str(target)
    yield copies


def test_two_copies_with_different_content_are_reported_with_their_age_gap():
    """The only actionable finding in this report that needs no `useful` link, because it is
    not about value: the harness lists the name once, the files differ, and which one loaded
    is unrecoverable — so the owner cannot know which version is in effect."""
    day = 86_400
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        with _installed(tmp, [
            _dup("second-skill", "claude_code", ".claude/skills/second-skill/SKILL.md",
                 "new rules", 1_760_000_000),
            _dup("second-skill", "shared", ".agents/skills/second-skill/SKILL.md",
                 "old rules", 1_760_000_000 - 77 * day),
        ]) as copies:
            out = rp._diverged_duplicates(copies)

    assert len(out) == 1
    assert out[0]["name"] == "second-skill" and out[0]["harness"] == "claude_code"
    assert out[0]["days_apart"] == 77
    assert ".agents" in out[0]["stale"]
    text = "\n".join(rp._duplicate_lines(out))
    assert "77 天" in text and "较旧" in text
    # The fix is the owner's: real configuration is the object under measurement.
    assert "由你定" in text and "不改你的真实配置" in text


def _rule_dup(name, harness, relative, body, mtime):
    """A duplicate whose body is real prose, so the rule classifier has something to read."""
    return _dup(name, harness, relative, body, mtime)


def test_copies_that_instruct_differently_say_so_not_just_that_they_differ():
    """A date and a byte count establish only that the files are not the same. Whether the
    difference *matters* then needs the reader to open every copy.

    Found on a real environment: of three copies of one name, two stated machine-checkable
    rules and the third stated none. So which file the harness happened to load decided
    whether the agent was given the rule at all — a much sharper statement than "79 days
    apart", and free, from the same classification the `useful` link already uses.
    """
    day = 86_400
    stating = ("Commit convention: the subject must end with the issue reference, "
               "and stay within 50 characters.\n")
    silent = "Background on how this project came about, with no rule about output.\n"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        with _installed(tmp, [
            _rule_dup("third-skill", "claude_code", ".claude/skills/third-skill/SKILL.md",
                      stating, 1_760_000_000),
            _rule_dup("third-skill", "shared", ".agents/skills/third-skill/SKILL.md",
                      silent, 1_760_000_000 - 40 * day),
        ]) as copies:
            out = rp._diverged_duplicates(copies)

    assert len(out) == 1
    assert out[0]["rules_differ"] is True
    assert out[0]["some_copy_states_no_rule"] is True
    by_path = {c["path"]: c for c in out[0]["copies"]}
    stated = next(c for p, c in by_path.items() if ".claude" in p)
    assert stated["rules"], "a copy stating a rule must carry its name"
    assert not next(c for p, c in by_path.items() if ".agents" in p)["rules"]

    text = "\n".join(rp._duplicate_lines(out))
    assert "没有声明可判定的规则" in text
    assert "决定了这条规则到底有没有被下达" in text
    # Still no instruction about what to do: the configuration stays the owner's.
    assert "由你定" in text


def test_copies_that_agree_on_their_rules_do_not_claim_they_disagree():
    """The guard against the enrichment inventing a second problem: files can differ in
    wording while instructing the same thing, and saying otherwise would be a false alarm
    on top of a real one."""
    day = 86_400
    first = "The subject must end with the issue reference.\n"
    second = "Always end the subject line with the issue reference, please.\n"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        with _installed(tmp, [
            _rule_dup("fourth-skill", "claude_code", ".claude/skills/fourth-skill/SKILL.md",
                      first, 1_760_000_000),
            _rule_dup("fourth-skill", "shared", ".agents/skills/fourth-skill/SKILL.md",
                      second, 1_760_000_000 - 5 * day),
        ]) as copies:
            out = rp._diverged_duplicates(copies)

    assert len(out) == 1, "differing bytes are still reported"
    assert out[0]["rules_differ"] is False
    assert out[0]["some_copy_states_no_rule"] is False
    text = "\n".join(rp._duplicate_lines(out))
    assert "规则不同" not in text and "没有声明可判定的规则" not in text


def test_identical_copies_are_not_reported_as_ambiguous():
    """Redundancy is not ambiguity: whichever loads, the bytes are the same. Reporting it
    would be inventing a problem."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        with _installed(tmp, [
            _dup("demo-skill", "claude_code", ".claude/skills/demo-skill/SKILL.md", "same\n", 1),
            _dup("demo-skill", "shared", ".agents/skills/demo-skill/SKILL.md", "same\n", 1),
        ]) as copies:
            assert rp._diverged_duplicates(copies) == []


def test_ambiguity_is_judged_per_harness_because_the_roots_differ():
    """Claude Code reads `.claude` + `.agents`; Codex reads `.codex` + `.agents`. Judging
    this globally put a skill on the list on the strength of a Codex copy that Claude Code
    never sees — and hid a real Codex-side divergence behind an identical Claude pair."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        with _installed(tmp, [
            _dup("demo-skill", "claude_code", ".claude/skills/demo-skill/SKILL.md", "same\n", 1),
            _dup("demo-skill", "shared", ".agents/skills/demo-skill/SKILL.md", "same\n", 1),
            _dup("demo-skill", "codex", ".codex/skills/demo-skill/SKILL.md", "different\n", 1),
        ]) as copies:
            out = rp._diverged_duplicates(copies)

    # One row, on the Codex side only: that is where the two visible copies really differ.
    assert [(d["name"], d["harness"]) for d in out] == [("demo-skill", "codex")]
    # Same-day copies get no staleness label, because the report prints dates and a label
    # derived from a sub-day mtime could not be checked against what the reader is shown.
    assert out[0]["stale"] is None
    assert "较旧" not in "\n".join(rp._duplicate_lines(out))


def test_the_findings_come_first_and_answer_the_readers_question():
    """The report's original sin: its first screen was coverage statistics, which answer
    "how well did the tool do" rather than "how is my environment". A reader does not reach
    the fourth screen, so the discipline below it was being enforced for nobody."""
    with _env(_trace([("mcp__s__go", {"q": "x"})] * 3), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
    text = data["text"]
    assert text.index("## 发现了什么") < text.index("## 证据链覆盖情况")
    assert text.index("## 发现了什么") < text.index("## 装了什么")
    # It must still refuse the question this evidence cannot answer. That needs a measured
    # *absence* of effect, which no component in this project is in.
    answer = text.split("## 发现了什么")[1].split("\n## ")[0]
    for banned in ("该删", "建议删除", "可以省", "能省", "没用", "浪费", "该留"):
        assert banned not in answer, banned


# -- findings: the report's top-level output ----------------------------------------------


def test_findings_group_by_class_so_one_problem_is_one_finding():
    """The first version emitted one finding per duplicated skill, and four
    identically-worded `fix` rows filled the whole first screen and pushed out every other
    kind — the same overload the section exists to prevent, in a new costume."""
    data = {
        "diverged_duplicates": [
            {"kind": "skills", "name": n, "harness": "claude_code", "days_apart": d,
             "copies": [{"path": f"/a/{n}", "modified": "2026-11-18", "bytes": 10},
                        {"path": f"/b/{n}", "modified": "2026-09-02", "bytes": 11}]}
            for n, d in (("second-skill", 77), ("third-skill", 3), ("nested-skill", 0))],
        "components": [], "verdicts": [], "always_on": [], "reliability": [],
        "next_steps": [], "sessions": 10,
    }
    findings = rp._findings(data)
    assert len(findings) == 1
    finding = findings[0]
    assert finding["action"] == "fix" and finding["instances"] == 3
    assert "3 个 skill" in finding["headline"]
    # Ranked by the worst instance, and every instance still reachable in the evidence.
    assert finding["magnitude"] == 77 and len(finding["evidence"]) == 3


def test_findings_are_ordered_by_how_ready_they_are_to_act_on():
    """The order is not an invented severity score — this project does not get to
    manufacture an impact number any more than it gets to manufacture a saving."""
    data = {
        "diverged_duplicates": [
            {"kind": "skills", "name": "d", "harness": "codex", "days_apart": 1,
             "copies": [{"path": "/a", "modified": "2026-11-01", "bytes": 1},
                        {"path": "/b", "modified": "2026-10-31", "bytes": 2}]}],
        "always_on": [{"kind": "skills", "name": "u", "tokens": 160, "calls": 0,
                       "also": [], "harness_label": "claude_code"}],
        "verdicts": [{"name": "k", "kind": "claude_md", "cost_value": 709,
                      "cost_unit": "tokens_per_request", "stable_across_profiles": True,
                      "basis": "实测分离", "limit": "只说保留"}],
        "next_steps": [{"link": "used", "method": "collect traces for this root",
                        "estimated_usd": 0.0, "note": "n", "components": ["x"]}],
        "components": [], "reliability": [], "sessions": 10,
    }
    assert [f["action"] for f in rp._findings(data)] == ["fix", "test", "investigate", "keep"]


def test_no_finding_recommends_removal():
    """`fix` is about consistency and says nothing about value; `keep` is an abstention.
    Removal needs a measured *absence* of effect, which no component is ever in."""
    with _env(_trace([("mcp__s__go", {"q": "x"})] * 3), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
    blob = json.dumps(data["findings"], ensure_ascii=False)
    for banned in ("该删", "建议删除", "可以省", "能省", "没用", "浪费", "该留"):
        assert banned not in blob, banned
    assert all(f["action"] in rp.FINDING_ORDER for f in data["findings"])


def test_the_ask_finding_counts_a_duplicated_name_once():
    """Two installed copies are two components and one listed skill. Listing both put the
    same name in this finding twice — the aggregation trap in a seventh unit."""
    def costed(name, harness, tokens, calls):
        return {"kind": "skills", "name": name, "harness": harness,
                "next_measurement": {"achievable": False},
                "links": {"used": {"status": "proven", "evidence": "e", "value": calls},
                          "cost": {"status": "partial", "evidence": "c", "value": tokens,
                                   "unit": "tokens_per_request"},
                          "useful": {"status": rp.UNPROVEN, "evidence": "u"}}}
    data = {"components": [costed("second-skill", "claude_code", 143, 9),
                           costed("second-skill", "shared", 141, 9)],
            "diverged_duplicates": [], "verdicts": [], "always_on": [],
            "reliability": [], "next_steps": [], "sessions": 10}
    findings = rp._findings(data)
    assert len(findings) == 1 and findings[0]["action"] == "ask"
    assert findings[0]["instances"] == 1
    assert len(findings[0]["evidence"]) == 1
    # The lower figure, as a floor should keep.
    assert "141" in findings[0]["evidence"][0]


def test_the_ask_finding_never_fabricates_an_experiment():
    """The one place this tool must refuse its own advice pattern. The arm for an MCP server
    exists and is verified; what is missing is a machine-checkable task. Two proxies were
    invented before and both were self-deception, so the honest output is a question for the
    the owner, not a third proxy."""
    data = {"components": [{
        "kind": "mcp", "name": "search-server", "harness": "claude_code",
        "next_measurement": {"achievable": False},
        "links": {"used": {"status": "proven", "evidence": "159 calls", "value": 159},
                  "cost": {"status": "partial", "evidence": "470 tokens", "value": 470,
                           "unit": "tokens_per_request"},
                  "useful": {"status": rp.UNPROVEN, "evidence": "not measured"}}}],
        "diverged_duplicates": [], "verdicts": [], "always_on": [],
        "reliability": [], "next_steps": [], "sessions": 10}
    finding = rp._findings(data)[0]
    assert finding["action"] == "ask"
    # It asks for a criterion rather than proposing a design.
    assert "你看一眼就知道" in finding["next"]
    assert "两次都是自欺" in finding["next"]
    assert "缺的不是实验手段" in finding["detail"]


def test_a_component_with_a_real_priced_measurement_is_test_not_ask():
    """`ask` means the tool is out of methods. If a priced measurement exists, saying "I
    need something from you" would be passing off work the tool can do."""
    data = {"components": [{
        "kind": "skills", "name": "s", "harness": "claude_code",
        "next_measurement": {"achievable": True, "link": "cost", "estimated_usd": 0.18,
                             "method": "skill prefix differencing (`measure-skills`)"},
        "links": {"used": {"status": "proven", "evidence": "u", "value": 5},
                  "cost": {"status": "partial", "evidence": "c", "value": 200,
                           "unit": "tokens_per_request"},
                  "useful": {"status": rp.UNPROVEN, "evidence": "n"}}}],
        "diverged_duplicates": [], "verdicts": [], "always_on": [], "reliability": [],
        "next_steps": [{"link": "cost", "method": "skill prefix differencing (`measure-skills`)",
                        "estimated_usd": 0.18, "note": "n", "components": ["s"]}],
        "sessions": 10}
    assert [f["action"] for f in rp._findings(data)] == ["test"]


def test_the_coverage_gap_survives_the_redesign_in_one_line():
    """A reader who finishes five findings and concludes "so my environment is fine" has
    made a worse error than one who drowned in tables."""
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
    section = data["text"].split("## 发现了什么")[1].split("\n## ")[0]
    assert "给不出" in section and "不是说它们没问题" in section


def test_the_first_screen_is_capped_and_says_what_it_held_back():
    """Ranking belongs to `_findings` and rendering to `_finding_lines`, so this passes
    already-ranked input — the renderer must not re-sort behind the caller's back."""
    findings = [{"action": "fix", "magnitude": 9 - i, "headline": f"h{i}", "detail": "d",
                 "evidence": ["e"], "next": "n", "instances": 1, "id": f"i{i}"}
                for i in range(9)]
    data = {"components": [], "sessions": 1}
    text = "\n".join(rp._finding_lines(findings, data))
    assert f"前 {rp.FIRST_SCREEN} 条" in text
    assert f"另有 {9 - rp.FIRST_SCREEN} 条" in text
    shown = text.split("除此之外")[0]
    assert "h0" in shown and "h4" in shown
    # Everything past the cap is held back, and still reachable in the detail sections.
    assert "h5" not in shown and "h8" not in shown


# -- reliability: free, and about working rather than about value -------------------------


def test_reliability_is_grouped_per_server_and_flags_only_a_real_failure_rate():
    traces = {"observed": {
        "calls_by_tool": {"mcp__broken__go": 20, "mcp__fine__go": 500, "Read": 900},
        "errors_by_tool": {"mcp__broken__go": 9, "mcp__fine__go": 4, "Read": 100},
    }}
    rows = {r["name"]: r for r in rp._reliability(traces)}
    assert set(rows) == {"broken", "fine"}          # non-MCP tools are not servers
    assert rows["broken"]["suspicious"] is True
    assert rows["fine"]["suspicious"] is False
    text = "\n".join(rp._reliability_lines(list(rows.values())))
    # It must not be mistaken for evidence about value.
    assert "不回答「它值不值」" in text


def test_a_tiny_sample_is_not_a_failure_rate():
    """Two failures out of three looks like 67% and means nothing."""
    traces = {"observed": {"calls_by_tool": {"mcp__new__go": 3},
                           "errors_by_tool": {"mcp__new__go": 2}}}
    assert rp._reliability(traces)[0]["suspicious"] is False


def test_a_failing_server_becomes_a_fix_finding_not_a_value_judgment():
    data = {"reliability": [{"name": "broken", "calls": 20, "errors": 9,
                             "error_rate": 0.45, "suspicious": True,
                             "worst_tools": [("go", 20, 9)]}],
            "diverged_duplicates": [], "components": [], "verdicts": [],
            "always_on": [], "next_steps": [], "sessions": 10}
    finding = rp._findings(data)[0]
    assert finding["action"] == "fix"
    assert "45%" in finding["headline"]
    assert "跟「值不值得装」是两件事" in finding["detail"]


# -- the packaged skill must not drift from the code -------------------------------------


def _skill_text() -> str:
    root = pathlib.Path(__file__).resolve().parent.parent / "skill" / "agent-profiler"
    return "\n".join(p.read_text(errors="replace") for p in sorted(root.rglob("*.md")))


def test_the_packaged_skill_documents_every_command_the_cli_has():
    """The shipped skill was a snapshot from the first round: it told its reader to run
    `inspect`, `trace` and `compare`, and knew nothing about the three commands that
    actually measure anything. A stale entry point misleads the next reader — including
    this project's own author, months later."""
    from agent_profiler import cli

    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        import argparse
        # The parser is built inside main(); recover the command names from the source.
        source = pathlib.Path(cli.__file__).read_text()
        names = set(re.findall(r'add_parser\(\s*"([a-z-]+)"', source))
    else:  # pragma: no cover - only if the CLI grows a factory later
        names = set(parser._subparsers._actions[-1].choices)

    text = _skill_text()
    missing = sorted(n for n in names if n not in text)
    assert not missing, f"the packaged skill does not mention: {missing}"


def test_the_packaged_skill_carries_the_refusals_and_not_a_stale_version_claim():
    """Its guardrails are the part that must never lag: each one exists because this project
    violated it and the output looked fine anyway."""
    text = _skill_text()
    for rule in ("Cost is not waste", "only say \"keep\"", "Zero calls is not zero use",
                 "n = 1", "never by editing in place"):
        assert rule in text, rule
    # "in v0.1" dated the old guardrail to a version that has since moved.
    assert "0-100 score in v0.1" not in text
    # The disproved blocker must not reappear in the thing that ships to a reader.
    assert "cannot be toggled individually" not in text


# -- next steps are measurements, not advice ---------------------------------------------


def test_a_blind_spot_that_measurement_closed_is_narrowed_to_what_is_still_unobserved():
    """The list's own rule: a closed blind spot must leave it, or the report keeps telling
    the user it cannot answer something it just answered.

    This one said "whether an auto-matched skill's body enters context", on the evidence
    that no record shows it. A measured skill costs an order of magnitude less per request
    than its body is long, so at listing time the body is demonstrably not in the prefix.
    What is still unobserved is only the step after a description matches.

    The narrowed wording must not carry the figures that narrowed it. This string prints on
    whoever's machine runs the report, so a measurement from another stack quoted here reads
    as theirs — the overclaim the list itself exists to prevent.
    """
    spot = next(s for s in _confirmed_blind_spots() if s["requires"] == "skills")
    evidence = spot["evidence"]
    # Either rendering, so this does not depend on the process language.
    assert any(k in evidence for k in ("measurable", "可测")), evidence
    assert any(k in evidence for k in ("order of magnitude", "数量级")), evidence
    assert not re.search(r"\d{3,}|\d+\s*(?:KB|tokens)", evidence), evidence
    assert "没有记录显示正文被载入" not in evidence
    assert "随后" in spot["what"]


def test_next_steps_only_list_achievable_measurements():
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        for step in data["next_steps"]:
            assert step["estimated_usd"] is not None
        assert "不是「该怎么改配置」" in data["text"]


def test_the_reports_own_prose_is_not_mixed_language():
    """Quoted evidence stays English, but the headline and the advice are the report's own
    words and must read as one language."""
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        assert "components have the evidence" not in data["headline"]
        advice = data["text"].split("## 想往前推一步")[1]
        for method in (s["method"] for s in data["next_steps"]):
            assert method in rp._method_labels(), f"unlocalized method reached the advice: {method}"
            assert rp._method_labels()[method] in advice


def test_the_english_report_leaves_no_chinese_behind():
    """The failure this guards is a half-translated report: one forgotten `tr()` leaves a
    Chinese fragment mid-paragraph on an English page, which reads as a bug in the tool
    rather than as either language. 313 strings were converted by hand, so this is checked
    mechanically over a report exercising every section — findings, verdicts, duplicates,
    the always-on list, reliability, inventory, coverage, cost, blind spots and next steps.

    Identifiers are exempt: a record type, a flag and a tool name should read the same in
    both languages as they do in the artifact they came from."""
    evaluation = _evaluation()
    with _env(_trace([("mcp__s__go", {"q": "x"})] * 12, attachments=[
                  {"type": "file", "names": ["CLAUDE.md"]}]),
              {"s": {"command": "x"}}, skills=("alpha", "beta")) as root:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(evaluation, fh)
            path = fh.name
        try:
            with _lang("en"):
                data = report_root(str(root), evaluation_path=path)
                text = data["text"]
        finally:
            os.unlink(path)

    leaked = re.findall(r"[一-鿿]+", text)
    assert not leaked, f"untranslated fragments reached the English report: {leaked}"
    # And it is the English report, not an empty one: the section headings must be there.
    for heading in ("## What was found", "## What it costs", "## What is installed",
                    "## Evidence-chain coverage", "## Component by component",
                    "## What this report cannot answer",
                    "## What could be done to get further"):
        assert heading in text, heading
    # The structured fields reach JSON consumers too, so they are checked separately.
    blob = json.dumps({k: v for k, v in data.items() if k != "text"}, ensure_ascii=False)
    assert not re.findall(r"[一-鿿]+", blob), "Chinese reached the structured output"


def test_an_unachievable_gap_never_reaches_next_steps():
    """`useful` needs an experiment Stage 6 refused; it must not appear with a price."""
    with _env(_trace([("mcp__s__go", {"q": "x"})]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        assert all(s["link"] != "useful" for s in data["next_steps"])


def test_next_steps_are_deduped_across_components():
    with _env(_trace(["Read"]), {"a": {"command": "x"}, "b": {"command": "y"}}) as root:
        data = report_root(str(root))
        keys = [(s["link"], s["method"]) for s in data["next_steps"]]
        assert len(keys) == len(set(keys)), "one method should be listed once, with its components"


# -- relocation --------------------------------------------------------------------------


def test_a_moved_project_is_flagged_before_any_zero_is_shown():
    """A zero-usage table under a moved path is a wrong answer that looks right."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        home, moved = Path(h), Path(d) / "demo"
        moved.mkdir()
        (moved / ".mcp.json").write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}),
                                         encoding="utf-8")
        old_cwd = "/old/path/demo"
        old = home / ".claude" / "projects" / old_cwd.replace("/", "-")
        old.mkdir(parents=True)
        (old / f"{SESSION}.jsonl").write_text(json.dumps({
            "type": "user", "cwd": old_cwd, "sessionId": SESSION,
            "timestamp": "2026-11-19T10:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "x"}]}}) + "\n",
            encoding="utf-8")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            data = report_root(str(moved))
            assert len(data["relocation_candidates"]) == 1
            text = data["text"]
            # The warning must precede the component table, or it is useless.
            assert text.index(old_cwd) < text.index("## 逐个组件")
            assert "这是零记录，不是真的没用过" in text
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# -- shape and cost ----------------------------------------------------------------------


def test_loaded_evidence_reaches_the_report():
    attachments = [
        {"type": "skill_listing", "isInitial": True, "names": ["demo"],
         "content": "- demo: does a thing\n"},
        {"type": "instructions", "files": [
            {"path": "/proj/CLAUDE.md", "type": "Project", "content": "y" * 1200}]},
    ]
    with _env(_trace(["Read"], attachments=attachments), skills=("demo",)) as root:
        data = report_root(str(root))
        demo = next(c for c in data["components"] if c["name"] == "demo")
        assert demo["links"]["loaded"]["status"] == sc.PROVEN
        # Evidence strings are structured data shared by every stage, so they stay English
        # and are quoted verbatim; only the report's own prose is localized.
        assert "one-line description" in demo["links"]["loaded"]["evidence"]


def test_report_spends_nothing_and_is_read_only():
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        assert data["read_only"] is True
        assert data["profiler_cost"]["cost_usd"] == 0.0
        assert data["profiler_cost"]["llm_input_tokens"] == 0
        assert data["schema_version"] == rp.REPORT_SCHEMA_VERSION


def test_every_component_appears_in_the_table():
    with _env(_trace(["Read"]), {"a": {"command": "x"}}, skills=("s1", "s2")) as root:
        data = report_root(str(root))
        assert len(data["components"]) >= 3
        for component in data["components"]:
            assert f"`{component['name']}`" in data["text"]


def test_cli_text_and_json_and_errors():
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        with tempfile.TemporaryDirectory() as out:
            text_path = Path(out) / "r.md"
            json_path = Path(out) / "r.json"
            assert cli_main(["report", "--root", str(root), "--output", str(text_path)]) == 0
            assert text_path.read_text(encoding="utf-8").startswith("# Agent 环境体检报告")
            assert cli_main(["report", "--root", str(root), "--format", "json",
                             "--output", str(json_path)]) == 0
            assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == "report"
        assert cli_main(["report", "--root", ""]) == 2
        assert cli_main(["report", "--root", str(root / "nope")]) == 2


def test_an_excluded_component_is_marked_and_its_blank_cost_is_attributed():
    """Marked-not-hidden only works if the readable output marks it. A reader who asked for
    something to be left alone must see that it was, and must not read the empty cost cell
    as a shortfall in the tool."""
    with _env(_trace([("mcp__s__go", {"q": "x"})]), {"s": {"command": "x"}}) as root:
        exclusions = {"names": {"mcp": {"s"}}, "source": "--exclude", "reason": "private"}
        data = report_root(str(root), exclusions=exclusions)

    component = next(c for c in data["components"] if c["name"] == "s")
    assert component["excluded_by_request"] is True
    assert component["links"]["cost"]["blocker"] == "excluded_by_request"
    text = data["text"]
    assert "按要求不接触" in text and "按要求排除" in text
    # The reason has to name the user's decision, and must not drop the standing caveat
    # that the cost is still real.
    assert "你要求这个工具不去接触它" in text
    assert "这笔开销是真实存在的" in text


def test_a_name_under_two_harnesses_is_distinguished_in_the_table():
    """Real output had two `demo-skill` rows with contradictory evidence and nothing to tell
    them apart; the same name under two harnesses is deliberately two components."""
    with _env(_trace(["Read"]), {"s": {"command": "x"}}) as root:
        data = report_root(str(root))
        names = [c["name"] for c in data["components"]]
        for name in {n for n in names if names.count(n) > 1}:
            rows = [l for l in data["text"].splitlines() if l.startswith(f"| `{name}`")]
            assert len({r.split("|")[1] for r in rows}) == len(rows), name


def test_a_blind_spot_is_not_reported_for_something_the_user_does_not_have():
    """Real defect: the report told an environment with ZERO hooks that it could not
    tell whether its hooks had run. A gap about a component class that is absent is an
    invented limitation, and inventing gaps is the mirror image of hiding them."""
    from agent_profiler.report import _confirmed_blind_spots, _present_classes

    assert all("requires" in spot for spot in _confirmed_blind_spots())

    empty = {"observed": {"by_kind": {}}, "files": []}
    assert _present_classes(empty, []) == set()

    with_skills = {"observed": {"by_kind": {"skills": 3}}, "files": []}
    present = _present_classes(with_skills, [])
    assert "skills" in present and "hooks" not in present

    hooked = {"observed": {"by_kind": {}}, "files": [{"features": ["hooks"]}]}
    assert "hooks" in _present_classes(hooked, [])

    codex_skill = [{"kind": "skills", "harnesses": ["codex"]}]
    assert "codex_skills" in _present_classes(with_skills, codex_skill)
    assert "codex_skills" not in _present_classes(
        with_skills, [{"kind": "skills", "harnesses": ["claude_code"]}])


def test_a_cumulative_shared_total_is_never_shown_as_a_per_request_cost():
    """The aggregation trap, caught a third time.

    `shared_context_chars` is summed across every collected session, so it is a whole-corpus
    total. Printing that beside per-request figures reads as though one skill listing costs
    millions of characters. Both forms must appear, each labelled with what it is.
    """
    traces = {"observed": {
        "sessions_collected": 10, "assistant_messages": 20,
        "tokens": {"total_tokens": 1000},
        "shared_context_chars": {"skill_listing": 70_000},
    }}
    line = next(l for l in rp._cost_lines(traces, None)
                if "skill 列表" in l["label"])
    assert "7,000" in line["value"], line["value"]
    assert "70,000" in line["detail"] and "累计" in line["detail"]
    # A shared block must say it is shared and must not be divided among its members.
    assert "共享" in line["label"]
    assert "没有按 skill 拆分" in line["detail"]


def test_the_codex_listing_cost_states_how_little_of_the_corpus_carries_it():
    """`world_state` is recent, so only a small minority of sessions carry it: a figure read
    from those sessions is not a figure about the history."""
    traces = {"observed": {"sessions_collected": 3, "assistant_messages": 5,
                           "tokens": {"total_tokens": 10}}}
    shared = {"skill_listing_tokens": [1500, 1700], "skills_covered": ["a", "b"],
              "sessions_scanned": 400, "sessions_carrying_record": 6}
    line = next(l for l in rp._cost_lines(traces, None, shared)
                if "Codex" in l["label"])
    assert "1,500–1,700 token" in line["value"]
    assert "400" in line["detail"] and "6" in line["detail"]
    assert line["strength"] == "observed"


def test_no_shared_cost_line_when_there_is_nothing_to_report():
    traces = {"observed": {"sessions_collected": 3, "assistant_messages": 5,
                           "tokens": {"total_tokens": 10}}}
    labels = [l["label"] for l in rp._cost_lines(traces, None, None)]
    assert not any("skill 列表" in l for l in labels)


def test_a_scorable_component_is_actually_shown_to_the_reader():
    """The report computed scores and rendered none of them, which was invisible only
    while no component was scorable. A conclusion the tool reached and did not print is
    the same as one it never reached."""
    verdicts = [{
        "name": "AGENTS.md (~)", "kind": "agents_md", "verdict": "keep", "score": 0.9082,
        "signals": {"used": 1.0, "useful": 1.0, "cost": 0.7246},
        "stable_across_profiles": True, "cost_value": 450,
        "cost_unit": "tokens_per_request", "cost_status": "partial",
        "basis": "带这个文件的那一组「任务号写在标题末尾」每次都做到，不带的一次都没做到",
        "basis_evidence": "the arm carrying this file satisfied 1 stated rule in every valid run",
        "limit": "只说保留。",
    }]
    text = "\n".join(rp._verdict_lines(verdicts, "balanced"))
    assert "建议保留" in text and "0.9082" in text
    # The reader gets the Chinese rendering; the English string stays in the artifact. The
    # verdict section was printing a three-line English paragraph into Chinese prose, which
    # is the one place a person goes to find a conclusion.
    assert "每次都做到" in text and "stated rule" not in text
    assert "450 token/次" in text          # the unit is spelled out, not left as a bare number
    assert "三种权重下结论一致" in text
    assert "只说保留" in text              # and the limit travels with it

    # An environment with nothing scorable gets no section at all, not an empty heading.
    assert rp._verdict_lines([], "balanced") == []


def test_the_useful_verdict_is_rendered_from_the_rule_names_not_translated():
    """Built from `rules_proven` rather than by rewording the evidence string, so the
    sentence cannot drift from the numbers it describes."""
    link = {"status": "proven", "evidence": "english original",
            "rules_proven": ["subject_ends_with_reference", "body_line_length_limit"]}
    text = rp._useful_in_words(link)
    assert "标题末尾带议题编号" in text
    assert "正文每行不超过它自己写明的长度" in text
    # The control is part of the claim: without it a separation could be a broken arm.
    assert "对照组" in text and "可比" in text
    # An unknown rule name falls through rather than being dropped from the sentence.
    assert "brand_new_rule" in rp._useful_in_words(
        {"status": "proven", "evidence": "e", "rules_proven": ["brand_new_rule"]})
    # With no structured rules there is nothing to build from, so the original stands.
    assert rp._useful_in_words({"status": "proven", "evidence": "fallback"}) == "fallback"


def test_an_unstable_verdict_says_so_rather_than_reading_as_settled():
    verdicts = [{"name": "x", "kind": "skills", "verdict": "keep", "score": 0.5,
                 "signals": {}, "stable_across_profiles": False, "cost_value": 1,
                 "cost_unit": "tokens_per_request", "cost_status": "partial",
                 "basis": "b", "limit": "l"}]
    text = "\n".join(rp._verdict_lines(verdicts, "balanced"))
    assert "换一组权重结论就变" in text


def test_the_next_steps_preamble_tracks_what_has_actually_been_proven():
    """This sentence has been wrong twice, both times by lagging behind the data: it said
    the `useful` link did not exist after it was proven, then said those components' cost
    was unmeasured after it was measured — while the verdict section above it was already
    publishing conclusions for them."""
    base = {"observed": {"link_coverage": {k: {"proven": 0, "partial": 0, "unproven": 1}
                                           for k in rp.SCORED_LINKS}},
            "next_steps": [], "blind_spots": [], "blind_spots_not_applicable": [],
            "unanswerable": [], "components": [], "verdicts": [],
            "profiler_cost": {"elapsed_ms": 1, "llm_input_tokens": 0,
                              "llm_output_tokens": 0, "cost_usd": 0.0}}

    def preamble(coverage_useful, verdicts):
        data = json.loads(json.dumps(base))
        data["observed"]["link_coverage"]["useful"] = coverage_useful
        data["verdicts"] = verdicts
        return rp._next_steps_preamble(data)

    assert "还没有" in preamble({"proven": 0, "partial": 0, "unproven": 9}, [])
    mid = preamble({"proven": 2, "partial": 0, "unproven": 7}, [])
    assert "证明了 2 个" in mid and "成本还没测出来" in mid
    done = preamble({"proven": 2, "partial": 0, "unproven": 7},
                    [{"name": "a"}, {"name": "b"}])
    assert "2 个组件同时成立" in done
    assert "成本还没测出来" not in done


def test_the_headline_does_not_call_a_measured_link_unmeasured():
    """Real defect: the headline said 「成本还没有任何组件证明得了」 in the same sentence as
    「齐备的组件有 1 个」, because it counted only `proven` and ignored `partial`."""
    coverage = {"installed": {"proven": 2, "partial": 0, "unproven": 0},
                "loaded": {"proven": 2, "partial": 0, "unproven": 0},
                "used": {"proven": 1, "partial": 0, "unproven": 1},
                "useful": {"proven": 1, "partial": 0, "unproven": 1},
                "cost": {"proven": 0, "partial": 1, "unproven": 1}}
    components = [{"coverage": {"scorable": True}}, {"coverage": {"scorable": False}}]
    line = rp._headline(components, coverage, {"withheld": True, "reason": "only 1 of 2"})
    assert "成本** 还没有任何组件证明得了" not in line
    assert "只测到部分证明" in line
    assert "1 个" in line
    # The withheld reason is restated in Chinese rather than pasted in as English.
    assert "only 1 of 2" not in line


def test_the_inventory_reports_skills_as_entities_not_as_files():
    """Real defect: the `skills` row printed a *file* count while every row of the component
    table below it was a skill, so the two sections disagreed by a factor of ten and a reader
    comparing them concludes the profiler found one number and scored another."""
    lines = rp._installed_lines({
        "files": 256, "bytes": 982432,
        "by_kind": {"skills": 147, "memory": 96},
        "entities": {"skills": 12}, "entity_files": {"skills": 122},
        "bundled_files": 25})
    assert "- skills：12 个（122 个文件；另有 25 个宿主程序自带的）" in lines
    # memory has no entity rollup, so its count is labelled as what it is.
    assert "- memory：96 个文件" in lines


def test_the_inventory_does_not_claim_bundled_content_was_excluded():
    """The old line read 「已排除…自带内容」 while 25 bundled files sat inside the very total
    it annotated. Bundled files are counted and excluded from components, and saying so is
    different from saying they were dropped.

    The negative guards name both vocabularies: the Chinese prose moved from the bare English
    `harness` to 「宿主程序」 after a reader read the English term as a dependency they had
    failed to install, and a guard that tracks only the new wording would stop catching the
    claim it exists to catch."""
    lines = rp._installed_lines({
        "files": 256, "bytes": 982432, "by_kind": {},
        "entities": {}, "entity_files": {}, "bundled_files": 25})
    assert "已排除会话记录、运行时状态、缓存与依赖树" in lines[0]
    assert "harness 自带内容" not in lines[0] and "宿主程序自带内容" not in lines[0]
    assert "25 个是宿主程序自带的" in lines[1] and "排除" not in lines[1]

    assert not any("harness 自带" in l or "宿主程序自带" in l for l in rp._installed_lines(
        {"files": 10, "bytes": 20, "by_kind": {}, "entities": {},
         "entity_files": {}, "bundled_files": 0}))


def test_the_reports_own_cost_includes_rendering_it():
    """The figure was computed inside the dict literal, which is evaluated before
    `_headline`, `_findings` and `_render` run — so it structurally excluded the rendering
    and could only ever under-report. On a synthetic 9,000-component stack it understated
    for work that took 6.74 seconds, because a quadratic duplicate-name check in the
    rendering was most of the cost.

    A tool whose subject is honest accounting must not be optimistic about its own.
    """
    slow = 0.25
    original = rp._render

    def slowly(data):
        time.sleep(slow)
        return original(data)

    with _env(_trace(['Read'])) as root:
        rp._render = slowly
        try:
            data = report_root(str(root))
        finally:
            rp._render = original

    reported = data["profiler_cost"]["elapsed_ms"]
    assert reported >= slow * 1000, f"{reported} ms excludes a {slow * 1000:.0f} ms render"
    # The text must carry the same figure the JSON does, with no placeholder left behind.
    assert rp.ELAPSED_PLACEHOLDER not in data["text"]
    assert f"{reported}" in data["text"]


def test_duplicate_component_names_are_still_marked_after_the_quadratic_fix():
    """The duplicate-name check went from a per-component rescan to one count. It decides
    whether a row is disambiguated, so collapsing it wrongly would silently merge two real
    components — the same name under two harnesses is deliberately two components."""
    with _env(_trace(['Read']), skills=("shared-name",)) as root:
        data = report_root(str(root))

    names = [c["name"] for c in data["components"]]
    repeated = {n for n in names if names.count(n) > 1}
    counted = {n for n, times in Counter(names).items() if times > 1}
    assert repeated == counted
    # Every repeated name must be disambiguated in the table rather than printed twice bare.
    for name in repeated:
        assert data["text"].count(f"| {name} |") == 0, name


def test_carried_rows_are_never_summed_into_one_total():
    """Two copies of one skill carry one prefix, so adding their rows double counts it.

    This is the aggregation trap this project has hit more than once, so the guard is on
    the rendered text: no grand total may appear, and the prose has to say why.
    """
    rows = [
        {"name": "dup", "scope": "user", "harness": "claude_code",
         "measured_under": "claude_code", "path": "/h/.claude/skills/dup/SKILL.md",
         "tokens_per_request": 350, "requests": 1000, "tokens_carried": 350000,
         "lower_bound": False, "excludes": [], "where": ".claude/skills"},
        {"name": "dup", "scope": "user", "harness": "shared",
         "measured_under": "claude_code", "path": "/h/.agents/skills/dup/SKILL.md",
         "tokens_per_request": 340, "requests": 1000, "tokens_carried": 340000,
         "lower_bound": True, "excludes": ["codex"], "where": ".agents/skills"},
    ]
    text = "\n".join(rp._render_carried(rows))
    assert f"{350000 + 340000:,}" not in text
    assert "不可相加" in text or "not summed" in text
    # The two rows have to be distinguishable, or the reader sees one row printed twice.
    assert ".claude/skills" in text and ".agents/skills" in text
    # The bound is marked where the number is, not only in the prose.
    assert "≥340,000" in text


def test_a_carried_figure_says_whose_traffic_it_leaves_out():
    rows = [{"name": "search-server", "scope": "user", "harness": "claude_code+codex",
             "measured_under": "claude_code", "path": None,
             "tokens_per_request": 470, "requests": 1000, "tokens_carried": 470000,
             "lower_bound": True, "excludes": ["codex"]}]
    text = "\n".join(rp._render_carried(rows))
    assert "codex" in text
    assert "≥470,000" in text


def test_the_carried_section_does_not_put_a_price_on_the_tokens():
    """A token count is observed; money needs a rate, and this tool does not invent rates."""
    rows = [{"name": "x", "scope": "user", "harness": "claude_code",
             "measured_under": "claude_code", "path": None, "tokens_per_request": 709,
             "requests": 1000, "tokens_carried": 709000, "lower_bound": False,
             "excludes": []}]
    text = "\n".join(rp._render_carried(rows))
    assert "$" not in text
    assert "单价" in text or "price per token" in text


def test_the_carried_rows_are_empty_when_no_rate_was_measured():
    assert rp._render_carried([]) == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} report regression tests passed (0 tokens spent)")


if __name__ == "__main__":
    main()

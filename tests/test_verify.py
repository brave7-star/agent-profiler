"""Regression suite for the parser-assumption checker.

This command exists to be run on a *second* machine, which means its failure mode is the
dangerous direction: a verifier that reports "no evidence" where evidence exists hands that
machine a false all clear, and a verifier that reports "violated" on a healthy corpus gets
ignored. Both happened while it was being written, on the machine the whole project was
built on:

* sampling newest-first picked this profiler's own one-turn probe sessions, and **five of
  seven** assumptions came back `no_evidence`
* Codex's `token_usage_record` lives only in the **compressed** rollouts, so reading plain
  `.jsonl` found none of them
* the first token-identity check included zero-input bookkeeping rows, one of which carries
  a non-zero cumulative total — it would have reported the assumption violated on the
  machine the assumption came from
* the `host_skills` records were looked for under `world_state`, which is what several of
  this project's own comments said; the real key is `state`

So these tests are mostly about the three-way outcome being correct, and about rare records
still being found.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import verify as vf
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


@contextmanager
def _home(claude_records=None, codex_records=None):
    """A controlled pair of harness homes, so no test reads the real environment."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        claude, codex = root / ".claude", root / ".codex"
        if claude_records is not None:
            project = claude / "projects" / "-proj"
            project.mkdir(parents=True)
            (project / "s.jsonl").write_text(
                "\n".join(json.dumps(r) for r in claude_records) + "\n", encoding="utf-8")
        if codex_records is not None:
            sessions = codex / "sessions"
            sessions.mkdir(parents=True)
            (sessions / "rollout-1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in codex_records) + "\n", encoding="utf-8")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(claude)
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(codex)
        try:
            yield root
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _assistant(msg_id, ref, usage=None):
    return {"type": "assistant", "requestRef": ref,
            "message": {"id": msg_id, "role": "assistant",
                        "usage": usage or {"input_tokens": 10, "output_tokens": 2,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0}}}


def _healthy_claude():
    """A corpus that satisfies all four Claude assumptions, several times over."""
    records = []
    for i in range(4):
        records.append({"type": "api-request", "id": f"req{i}"})
        # One response, written as two records repeating the same usage.
        records.append(_assistant(f"m{i}", f"req{i}"))
        records.append(_assistant(f"m{i}", f"req{i}"))
        records.append({"type": "api-request-blob",
                        "message": {"role": "assistant",
                                    "content": [{"type": "tool_use", "name": "Read"}]}})
        records.append({"type": "attachment",
                        "attachment": {"type": "file", "names": ["CLAUDE.md"]}})
    return records


def _by_key(results):
    return {r["key"]: r for r in results}


def test_a_healthy_corpus_confirms_every_claude_assumption():
    with _home(_healthy_claude()):
        out = vf.verify()
    checks = _by_key(out["checks"])
    for key in ("usage_per_message_id", "replay_records_carry_tool_use",
                "latency_links_through_requestRef", "attachment_is_a_record_type"):
        assert checks[key]["status"] == vf.HOLD, (key, checks[key])
    assert checks["usage_per_message_id"]["instances"] >= vf.MIN_INSTANCES


def test_an_empty_machine_reports_no_evidence_and_never_a_pass():
    """The failure that matters: a fresh install must not read as "all assumptions hold"."""
    with _home([], []):
        out = vf.verify()
    assert all(c["status"] == vf.NO_EVIDENCE for c in out["checks"])
    assert out["summary"][vf.HOLD] == 0
    assert out["summary"][vf.VIOLATED] == 0
    # The verdict must say this machine could confirm nothing, not that nothing is wrong.
    assert f"里的 0 条" in out["verdict"], out["verdict"]
    text = vf.verify_text(out)
    assert "这不是通过" in text


def test_a_real_violation_is_reported_with_the_record_that_proves_it():
    """A second machine that writes differing usage per message id would make the collector
    silently wrong, so this must be loud and specific."""
    records = _healthy_claude()
    records.append({"type": "api-request", "id": "reqX"})
    records.append(_assistant("mX", "reqX", {"input_tokens": 10, "output_tokens": 2}))
    records.append(_assistant("mX", "reqX", {"input_tokens": 999, "output_tokens": 2}))
    with _home(records):
        out = vf.verify()
    check = _by_key(out["checks"])["usage_per_message_id"]
    assert check["status"] == vf.VIOLATED
    assert any("mX" in v for v in check["violations"])
    # The record is named by file and id; its contents are never printed.
    assert all("999" not in v for v in check["violations"])
    assert out["summary"][vf.VIOLATED] == 1
    assert "VIOLATED" in out["verdict"]
    assert "不成立" in vf.verify_text(out)


def test_a_single_instance_is_not_enough_to_confirm():
    """One observation is not a format law — the same mistake `reproducible` once made by
    calling a single pair's spread of zero reproducible."""
    records = [{"type": "api-request", "id": "r0"},
               _assistant("m0", "r0"), _assistant("m0", "r0")]
    with _home(records):
        out = vf.verify()
        with _lang("en"):
            english = _by_key(vf.verify()["checks"])["usage_per_message_id"]
    check = _by_key(out["checks"])["usage_per_message_id"]
    assert check["instances"] == 1
    assert check["status"] == vf.NO_EVIDENCE
    # The note must say how many more are needed, not merely that there are too few.
    assert f"需要 {vf.MIN_INSTANCES} 个" in check["note"]
    assert f"{vf.MIN_INSTANCES} needed" in english["note"]


def test_codex_zero_input_bookkeeping_rows_are_not_counted_as_violations():
    """The row that would have refuted the assumption on the machine it came from: input and
    output are zero while `total_tokens` carries a cumulative figure."""
    records = [{"type": "token_usage_record",
                "payload": {"usage": {"input_tokens": 0, "output_tokens": 0,
                                      "total_tokens": 180_230}}}]
    records += [{"type": "token_usage_record",
                 "payload": {"usage": {"input_tokens": 100 + i, "output_tokens": 5,
                                       "total_tokens": 105 + i}}} for i in range(3)]
    with _home(None, records):
        out = vf.verify()
    check = _by_key(out["checks"])["codex_input_includes_cached"]
    assert check["status"] == vf.HOLD
    assert check["instances"] == 3  # the zero-input row is skipped, not failed
    assert not check["violations"]


def test_codex_skill_listing_is_read_from_state_not_world_state():
    """`payload.state.host_skills`. Several of this project's own comments said
    `world_state`, and the verifier inherited the error, reporting no evidence for a record
    that was sitting right there."""
    entries = {f"/h/.agents/skills/s{i}/SKILL.md": {"shown": True} for i in range(4)}
    records = [{"type": "event_msg", "payload": {"state": {"host_skills": {
        "entries": entries, "usage": {"tokens": 2100}}}}}]
    with _home(None, records):
        out = vf.verify()
    assert _by_key(out["checks"])["host_skills_keyed_by_path"]["status"] == vf.HOLD


def test_a_rare_record_is_still_found_when_most_files_lack_it():
    """Codex's skill listing appears in only a small minority of sessions, and not the
    largest ones. A fixed sample of the biggest few files misses it, which is how the row
    read `no_evidence` on the very machine the record was discovered on — so sampling
    continues until a check is settled."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sessions = root / ".codex" / "sessions"
        sessions.mkdir(parents=True)
        # Many large files without the record, one small file with it: exactly the shape
        # that defeats largest-first sampling alone.
        filler = [{"type": "event_msg", "payload": {"msg": "x" * 400}} for _ in range(40)]
        for i in range(12):
            (sessions / f"rollout-big{i}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in filler) + "\n", encoding="utf-8")
        entries = {f"/h/skills/s{i}/SKILL.md": {"shown": True} for i in range(3)}
        (sessions / "rollout-rare.jsonl").write_text(
            json.dumps({"type": "event_msg",
                        "payload": {"state": {"host_skills": {"entries": entries}}}}) + "\n",
            encoding="utf-8")
        saved = os.environ.get("AGENT_PROFILER_CODEX_HOME")
        claude_saved = os.environ.get("AGENT_PROFILER_CLAUDE_HOME")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(root / ".codex")
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(root / ".claude")
        try:
            out = vf.verify()
        finally:
            for key, value in (("AGENT_PROFILER_CODEX_HOME", saved),
                               ("AGENT_PROFILER_CLAUDE_HOME", claude_saved)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    assert _by_key(out["checks"])["host_skills_keyed_by_path"]["status"] == vf.HOLD


def test_unreadable_compressed_sessions_are_not_reported_as_an_empty_corpus():
    """A file this machine has but cannot read is a different answer from a record it does
    not have. Decompression shells out to the `zstd` binary, which macOS does not ship, so
    this is likeliest on exactly the fresh second machine the command exists for — and
    cannot happen where zstd is installed. Collapsing it
    into a bare `no_evidence` turns a one-command fix into an unexplained blank."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sessions = root / ".codex" / "sessions"
        sessions.mkdir(parents=True)
        # Content is irrelevant: it is never reached, because `zstd` is not on PATH.
        (sessions / "rollout-1.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd not-real-zstd")
        saved_path = os.environ.get("PATH")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(root / ".codex")
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(root / ".claude")
        os.environ["PATH"] = str(root / "empty-bin")  # no zstd anywhere on it
        try:
            out = vf.verify()
        finally:
            os.environ["PATH"] = saved_path or ""
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # Still not a pass, and still not a violation — but now it says which.
    codex = [c for c in out["checks"] if c["harness"] == "codex"]
    assert all(c["status"] == vf.NO_EVIDENCE for c in codex)
    assert out["summary"][vf.VIOLATED] == 0
    assert out["unreadable"]["install"] == "zstd"
    assert out["unreadable"]["unread_files"] == 1
    assert any("zstd" in c["note"] for c in codex), [c["note"] for c in codex]

    # And the prose tells the reader the fix rather than leaving a blank row.
    text = vf.verify_text(out)
    assert "读不了，不是没有" in text
    assert "brew install zstd" in text
    assert "别把它们读成" in text
    # The blanket "this corpus holds no instance" is false when files went unread — the
    # instances may be sitting inside them.
    assert "或者有、但没能读取" in text


def test_a_present_decompressor_reports_no_unread_files():
    """The complement: without this, the field above could be set unconditionally and the
    test above would still pass."""
    with _home(None, [{"type": "token_usage_record", "payload": {"usage": {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}]):
        out = vf.verify()
    assert out["unreadable"]["unread_files"] == 0
    assert out["unreadable"]["install"] is None
    assert "brew install" not in vf.verify_text(out)


def test_the_table_explains_the_word_harness_in_plain_terms():
    """A reader took the bare English term for a dependency they had failed to install. It
    names two programs they already have, so the output says so where the column is."""
    with _home(_healthy_claude()):
        text = vf.verify_text(vf.verify())
    assert "| 宿主程序 |" in text
    assert "Claude Code" in text and "Codex" in text
    assert "没有别的东西要装" in text


def test_the_english_rendering_leaves_no_chinese_behind():
    """The failure this guards is a half-translated report: one `tr()` forgotten leaves a
    Chinese fragment in the middle of an English page, which is worse than either language
    alone because it reads as a bug in the tool. Every rendered path is exercised — the
    table, a violation, an untestable row and the unreadable-files section."""
    import re
    records = _healthy_claude()
    records.append({"type": "api-request", "id": "reqX"})
    records.append(_assistant("mX", "reqX", {"input_tokens": 10, "output_tokens": 2}))
    records.append(_assistant("mX", "reqX", {"input_tokens": 999, "output_tokens": 2}))
    with _home(records), _lang("en"):
        out = vf.verify()
        out["unreadable"] = {"unread_files": 3, "install": "zstd"}   # force that section
        text = vf.verify_text(out)
    leaked = re.findall(r"[一-鿿]+", text)
    assert not leaked, leaked
    # And it is actually the English text, not an empty page.
    assert "# Parser assumptions, checked" in text
    assert "| harness | assumption | result | instances |" in text
    assert "holds" in text and "violated" in text
    assert "brew install zstd" in text


def test_the_chinese_rendering_leaves_no_english_prose_behind():
    """The mirror. Identifiers, record-type names and shell commands stay in English by
    design, so this checks the prose markers rather than the absence of ASCII."""
    with _home(_healthy_claude()):
        text = vf.verify_text(vf.verify())
    assert "# 解析假设核对" in text
    assert "| 宿主程序 | 假设 | 结果 | 实例数 |" in text
    assert "✓ 成立" in text
    for english in ("holds", "untestable here", "Parser assumptions"):
        assert english not in text, english


def test_every_claim_and_why_is_translated_in_both_languages():
    """A `tr()` pair whose two halves are identical is a translation that was never done.
    Checked on the structured fields, because those also reach the JSON output."""
    import re
    with _home(_healthy_claude(), [{"type": "event_msg", "payload": {"state": {
            "host_skills": {"entries": {"/h/s/SKILL.md": {}}}}}}]):
        with _lang("zh"):
            zh = {c["key"]: c for c in vf.verify()["checks"]}
        with _lang("en"):
            en = {c["key"]: c for c in vf.verify()["checks"]}
    assert set(zh) == set(en) and len(zh) == 9
    for key in zh:
        for field in ("claim", "why"):
            assert zh[key][field] != en[key][field], (key, field)
            assert re.search(r"[一-鿿]", zh[key][field]), (key, field, "zh not translated")
            assert not re.search(r"[一-鿿]", en[key][field]), (key, field, "en has Chinese")


def test_verify_spends_nothing_and_says_so():
    with _home(_healthy_claude()):
        out = vf.verify()
    cost = out["profiler_cost"]
    assert cost["llm_input_tokens"] == 0 and cost["llm_output_tokens"] == 0
    assert cost["cost_usd"] == 0.0
    assert out["read_only"] is True
    assert any("不往外发任何东西" in n for n in out["notes"])


def test_every_check_states_why_it_matters():
    """A row a reader cannot act on is noise. `why` is what turns "this differs" into "your
    token counts are wrong"."""
    with _home(_healthy_claude()):
        out = vf.verify()
    for check in out["checks"]:
        assert check["why"], check["key"]
        assert check["claim"], check["key"]
        assert check["harness"] in ("claude_code", "codex")


def test_a_rate_limit_notice_is_not_an_instance_of_the_token_assumption():
    """The first version of this check reported `violated` on the machine that wrote it.

    It was right to fail: the assumption was stated as "every `token_count` carries a
    cumulative total", and 0.167 reuses the event type for rate-limit notices with
    `info: null`. The assumption was too broad, not the records wrong — so it was narrowed
    to the records that actually claim to be token reports.
    """
    key = "token_count_present_every_version"
    notices = [{"type": "event_msg", "payload": {
        "type": "token_count", "info": None,
        "rate_limits": {"limit_id": "codex"}}} for _ in range(6)]
    with _home(_healthy_claude(), notices):
        row = {c["key"]: c for c in vf.verify()["checks"]}[key]
    # Not counted at all, so the row says "this machine cannot tell", not "broken".
    assert row["instances"] == 0
    assert row["status"] == vf.NO_EVIDENCE

    reports = [{"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"total_tokens": 100 * i}}}} for i in range(1, 5)]
    with _home(_healthy_claude(), reports):
        row = {c["key"]: c for c in vf.verify()["checks"]}[key]
    assert row["status"] == vf.HOLD and row["instances"] == 4

    # A record that claims to be a token report but carries no total is a real failure.
    broken = [{"type": "event_msg", "payload": {"type": "token_count",
                                                "info": {"model_context_window": 1}}}
              for _ in range(3)]
    with _home(_healthy_claude(), broken):
        row = {c["key"]: c for c in vf.verify()["checks"]}[key]
    assert row["status"] == vf.VIOLATED


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} parser-assumption tests passed (0 tokens spent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

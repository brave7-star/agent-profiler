"""Regression suite for the Claude Code trace collector.

Each case encodes a property verified against live transcripts, or a defect the collector
would otherwise have. See docs/dogfood/trace-schema.md for the evidence.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from agent_profiler.cli import main as cli_main
from agent_profiler.traces import collect_root, collect_session, discover_sessions

SESSION_ID = "11111111-2222-3333-4444-555555555555"
# The transcript starts here; cost-state billing starts at the same instant, so there is
# no resume gap unless a test asks for one.
FIRST_TS = "2026-11-18T10:00:00.000Z"
BILLING_START_MS = 1794996000000
# 30 minutes later: the harness billed a run that began after the transcript did.
RESUMED_BILLING_START_MS = 1794997800000


def _rec(**kw):
    base = {"sessionId": SESSION_ID, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, block, *, ts, usage=None, block_index=0, request_ref=None, stop="end_turn", model="claude-opus-5-5"):
    message = {"id": msg_id, "role": "assistant", "model": model, "stop_reason": stop, "content": [block]}
    if usage is not None:
        message["usage"] = usage
    return _rec(
        type="assistant",
        message=message,
        apiBlockIndex=block_index,
        requestId="req-logical",
        requestRef=request_ref,
        timestamp=ts,
    )


def _usage(inp=10, read=1000, create=100, out=50):
    return {
        "input_tokens": inp,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
        "output_tokens": out,
    }


def _transcript() -> list[dict]:
    """A transcript exercising every real-format trap at once."""
    tool_block = {"type": "tool_use", "id": "toolu_A", "name": "Bash", "input": {"command": "ls"}}
    return [
        _rec(type="api-request", id="apireq-1", timestamp="2026-11-18T10:00:00.000Z"),
        _rec(
            type="user",
            message={"role": "user", "content": [{"type": "text", "text": "do the thing"}]},
            timestamp="2026-11-18T10:00:00.500Z",
        ),
        # One response split across two records: same message id, usage repeated.
        _assistant("msg_1", {"type": "thinking", "thinking": "..."}, ts="2026-11-18T10:00:04.000Z",
                   usage=_usage(), block_index=0, request_ref="apireq-1"),
        _assistant("msg_1", tool_block, ts="2026-11-18T10:00:04.000Z",
                   usage=_usage(), block_index=1, request_ref="apireq-1"),
        # A request replay: repeats the same tool_use block. Must not be counted again.
        _rec(type="api-request-blob", message={"role": "assistant", "content": [tool_block]}),
        # Tool result fed back as a user record: not a human turn.
        _rec(
            type="user",
            message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_A", "is_error": False,
                                                  "content": "ok"}]},
            timestamp="2026-11-18T10:00:05.000Z",
        ),
        _rec(type="api-request", id="apireq-2", timestamp="2026-11-18T10:00:06.000Z"),
        _assistant("msg_2", {"type": "tool_use", "id": "toolu_B", "name": "Read", "input": {}},
                   ts="2026-11-18T10:00:12.000Z", usage=_usage(out=20), request_ref="apireq-2", stop="tool_use"),
        _rec(
            type="user",
            message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_B", "is_error": True,
                                                  "content": "boom"}]},
            timestamp="2026-11-18T10:00:13.000Z",
        ),
        _rec(type="tools-used", tools=["Bash", "Read"]),
        _rec(type="cost-state", totalCostUSD=0.5, totalAPIDuration=9000, totalToolDuration=500,
             totalLinesAdded=3, totalLinesRemoved=1, startTime=BILLING_START_MS, hasUnknownModelCost=False,
             modelUsage={"claude-opus-5-5": {"inputTokens": 20, "cacheReadInputTokens": 2000,
                                             "cacheCreationInputTokens": 200, "outputTokens": 70}}),
    ]


@contextmanager
def _project(records=None, *, session_id=SESSION_ID, extra_files=None):
    """A temp project root plus a matching ~/.claude/projects transcript directory."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        key = str(root.resolve()).replace("/", "-").replace(".", "-")
        proj = home / ".claude" / "projects" / key
        proj.mkdir(parents=True)
        path = proj / f"{session_id}.jsonl"
        lines = [json.dumps(r) for r in (records if records is not None else _transcript())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for name, text in (extra_files or {}).items():
            (proj / name).write_text(text, encoding="utf-8")
        saved = {k: os.environ.get(k) for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        try:
            yield root, path
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_usage_counted_once_per_message():
    """The core trap: one response is many records, each repeating the same usage."""
    with _project() as (_root, path):
        r = collect_session(str(path))
        obs = r["observed"]
        assert obs["assistant_messages"] == 2
        assert r["integrity"]["duplicate_usage_records_skipped"] == 1
        # Naive per-record summing would give 3 x 1000 = 3000 cache reads.
        assert obs["tokens"]["cache_read_input_tokens"] == 2000
        assert obs["tokens"]["output_tokens"] == 70
        assert obs["tokens"]["effective_input_tokens"] == 20 + 2000 + 200
        assert obs["tokens"]["total_tokens"] == 20 + 2000 + 200 + 70


def test_request_replays_are_not_activity():
    """api-request-blob repeats the assistant's tool_use; counting it doubles tool calls."""
    with _project() as (_root, path):
        r = collect_session(str(path))
        tools = r["observed"]["tools"]
        assert tools["calls_total"] == 2
        assert tools["by_tool"]["Bash"]["calls"] == 1
        assert r["observed"]["record_types"]["api-request-blob"] == 1


def test_tool_results_attributed_to_their_tool():
    with _project() as (_root, path):
        tools = collect_session(str(path))["observed"]["tools"]
        assert tools["by_tool"]["Bash"] == {"calls": 1, "ok": 1, "errors": 0}
        assert tools["by_tool"]["Read"] == {"calls": 1, "ok": 0, "errors": 1}
        assert tools["errors_total"] == 1
        assert tools["error_references"][0]["tool"] == "Read"
        assert tools["unattributed_results"] == 0


def test_tool_result_records_are_not_user_turns():
    """Only a text-bearing user record is a human turn."""
    with _project() as (_root, path):
        assert collect_session(str(path))["observed"]["user_turns"] == 1


def test_latency_derived_from_request_ref():
    """assistant.requestRef links to api-request.id; requestId does not."""
    with _project() as (_root, path):
        lat = collect_session(str(path))["observed"]["latency"]
        assert lat["requests_measured"] == 2
        assert lat["min_seconds"] == 4.0
        assert lat["max_seconds"] == 6.0
        assert lat["total_seconds"] == 10.0


def test_cross_check_matches_harness_totals():
    """Agreement with the harness's own accounting is the evidence the rules are right."""
    with _project() as (_root, path):
        cc = collect_session(str(path))["cross_check"]
        assert cc["available"] is True
        assert cc["verdict"] == "exact_match"
        assert cc["double_counting_suspected"] is False
        assert cc["tool_names_agree"] is True
        assert all(f["exact_match"] for f in cc["tokens"].values())


def test_double_counting_is_detected_when_dedupe_fails():
    """Guard the guard: an extra billed message must trip the alarm."""
    records = _transcript() + [
        _assistant("msg_3", {"type": "text", "text": "extra"}, ts="2026-11-18T10:00:20.000Z", usage=_usage())
    ]
    with _project(records) as (_root, path):
        cc = collect_session(str(path))["cross_check"]
        assert cc["verdict"] == "double_counting_suspected"
        assert cc["double_counting_suspected"] is True
        assert "cache_read" in cc["over_counted_fields"]


def test_interrupted_responses_excluded_from_comparison():
    """Real finding: a stop_reason-less response is recorded but not billed."""
    records = _transcript() + [
        _assistant("msg_int", {"type": "text", "text": "cut off"}, ts="2026-11-18T10:00:30.000Z",
                   usage=_usage(read=89141, out=8), stop=None)
    ]
    with _project(records) as (_root, path):
        r = collect_session(str(path))
        assert r["observed"]["interrupted_messages"] == 1
        # Still counted in observed totals, because the transcript records it.
        assert r["observed"]["tokens"]["cache_read_input_tokens"] == 2000 + 89141
        cc = r["cross_check"]
        assert cc["interrupted_messages_excluded"] == 1
        assert cc["interrupted_tokens_excluded"]["cache_read"] == 89141
        assert cc["verdict"] == "exact_match"


def test_resumed_session_is_classified_not_alarmed():
    """Real finding: a resumed transcript predates the run cost-state billed."""
    records = [r for r in _transcript() if r["type"] != "cost-state"]
    records.append(
        _rec(type="cost-state", totalCostUSD=0.1, startTime=RESUMED_BILLING_START_MS,
             modelUsage={"claude-opus-5-5": {"inputTokens": 1, "cacheReadInputTokens": 1,
                                             "cacheCreationInputTokens": 1, "outputTokens": 1}})
    )
    with _project(records) as (_root, path):
        cc = collect_session(str(path))["cross_check"]
        assert cc["verdict"] == "resumed_session_superset"
        assert cc["double_counting_suspected"] is False
        assert cc["resumed_pre_billing_seconds"] > 0


def test_malformed_and_unknown_records_degrade():
    """A transcript must never crash the collector, and drift must be visible."""
    with _project() as (_root, path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
            fh.write("[1,2,3]\n")
            fh.write(json.dumps(_rec(type="brand-new-record-type")) + "\n")
            fh.write("\n")
            fh.write('{"type":"assistant","message":{"id":"trunc"')  # truncated final line
        r = collect_session(str(path))
        integrity = r["integrity"]
        assert integrity["parse_errors"] == 3
        assert integrity["unknown_record_types"] == {"brand-new-record-type": 1}
        assert len(integrity["parse_error_references"]) == 3
        # Real content is still collected despite the damage.
        assert r["observed"]["tools"]["calls_total"] == 2


def test_missing_usage_and_odd_shapes_are_survivable():
    records = [
        _rec(type="assistant", message={"id": "m", "role": "assistant", "content": "a string"}),
        _rec(type="assistant", message=None),
        _rec(type="assistant"),
        _rec(type="user", message={"role": "user", "content": "plain string prompt"}),
        _rec(type="user", message={"role": "user", "content": [None, 7, {"type": "text", "text": "hi"}]}),
        _rec(),
    ]
    with _project(records) as (_root, path):
        r = collect_session(str(path))
        assert r["observed"]["user_turns"] == 2
        assert r["observed"]["assistant_messages"] == 0
        assert r["cross_check"]["available"] is False
        assert r["observed"]["record_types"]["__missing_type__"] == 1


def test_unmatched_tool_results_are_reported():
    """Compaction can remove the tool_use while leaving its result."""
    records = [
        _rec(type="user", message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "gone",
                                                                "is_error": True}]}),
    ]
    with _project(records) as (_root, path):
        tools = collect_session(str(path))["observed"]["tools"]
        assert tools["unattributed_results"] == 1
        assert "__unattributed__" not in tools["by_tool"]
        assert tools["errors_total"] == 1


def test_session_discovery_is_scoped_to_the_root():
    with _project() as (root, path):
        idx = discover_sessions(str(root))
        assert idx["observed"]["sessions_found"] == 1
        assert idx["sessions"][0]["session_id"] == SESSION_ID
        # Compare resolved paths: on macOS /var is a symlink to /private/var.
        assert Path(idx["sessions"][0]["path"]) == path.resolve()
        assert idx["project_dir_exists"] is True
        # A different root must not pick up this project's transcripts.
        with tempfile.TemporaryDirectory() as other:
            empty = discover_sessions(other)
            assert empty["observed"]["sessions_found"] == 0
            assert empty["project_dir_exists"] is False


def test_collect_root_aggregates_and_limits():
    with _project() as (root, _path):
        agg = collect_root(str(root))
        obs = agg["observed"]
        assert obs["sessions_collected"] == 1
        assert obs["tool_calls"] == 2
        assert obs["calls_by_tool"] == {"Bash": 1, "Read": 1}
        assert obs["errors_by_tool"] == {"Read": 1}
        assert obs["tokens"]["total_tokens"] == 2290
        assert agg["failures"] == []
        assert collect_root(str(root), 0)["observed"]["sessions_collected"] == 1
        assert collect_root(str(root), 1)["observed"]["sessions_collected"] == 1


def test_profiler_reports_its_own_trace_cost():
    with _project() as (root, path):
        cost = collect_session(str(path))["profiler_cost"]
        assert cost["llm_input_tokens"] == 0 and cost["llm_output_tokens"] == 0
        assert cost["bytes_read"] > 0 and cost["lines_parsed"] > 0
        assert collect_root(str(root))["profiler_cost"]["llm_input_tokens"] == 0


def test_cli_surfaces_bad_input_as_exit_2():
    with _project() as (root, path):
        assert cli_main(["sessions", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["collect", "--session", str(path), "--output", os.devnull]) == 0
        assert cli_main(["collect", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["sessions", "--root", ""]) == 2
        # An empty --root must be validated as a root, not treated as a missing session.
        assert cli_main(["collect", "--root", ""]) == 2
        assert cli_main(["sessions", "--root", str(root / "nope")]) == 2
        assert cli_main(["collect", "--session", str(root / "missing.jsonl")]) == 2
        # A directory passed where a transcript is expected.
        assert cli_main(["collect", "--session", str(root)]) == 2



def test_skill_invocations_are_attributed_to_the_skill_they_name():
    """Real FN: a skill was assumed to leave no record. An explicit invocation is a `Skill`
    tool call carrying the skill's name, which is the one link from a declared entity to a
    real use. The name comes from a published list, so it is kept rather than digested."""
    records = [
        _rec(type="api-request", id="apireq-1", timestamp="2026-11-18T10:00:00.000Z"),
        _assistant("msg_s1", {"type": "tool_use", "id": "toolu_S1", "name": "Skill",
                              "input": {"skill": "second-skill"}},
                   ts="2026-11-18T10:00:01.000Z", usage=_usage(), request_ref="apireq-1"),
        _assistant("msg_s2", {"type": "tool_use", "id": "toolu_S2", "name": "Skill",
                              "input": {"skill": "second-skill"}},
                   ts="2026-11-18T10:00:02.000Z", usage=_usage(), request_ref="apireq-1"),
        _assistant("msg_s3", {"type": "tool_use", "id": "toolu_S3", "name": "Bash",
                              "input": {"command": "ls"}},
                   ts="2026-11-18T10:00:03.000Z", usage=_usage(), request_ref="apireq-1"),
    ]
    with _project(records) as (_root, path):
        observed = collect_session(str(path))["observed"]
    assert observed["tools"]["components_invoked"] == {"second-skill": 2}
    # A non-skill call contributes no component, and tool counting is unchanged.
    assert observed["tools"]["by_tool"]["Bash"]["calls"] == 1



def test_a_moved_project_surfaces_its_orphaned_history():
    """Real case: a project tree moved and its whole history stayed filed under the old
    path. Reporting a clean zero for the new location is a wrong answer that looks right."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        home = Path(h)
        new_root = Path(d) / "project-a"
        new_root.mkdir()
        old_cwd = "/somewhere/else/project-a"
        old_key = old_cwd.replace("/", "-").replace(".", "-")
        old_dir = home / ".claude" / "projects" / old_key
        old_dir.mkdir(parents=True)
        (old_dir / f"{SESSION_ID}.jsonl").write_text(
            json.dumps({"type": "user", "cwd": old_cwd, "sessionId": SESSION_ID,
                        "timestamp": "2026-11-18T10:00:00.000Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": "x"}]}}) + "\n",
            encoding="utf-8")
        saved = {k: os.environ.get(k) for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        try:
            r = discover_sessions(str(new_root))
            assert r["observed"]["sessions_found"] == 0
            candidates = r["relocation_candidates"]
            assert len(candidates) == 1
            assert candidates[0]["recorded_cwd"] == old_cwd
            assert candidates[0]["sessions"] == 1
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_a_differently_named_project_is_not_claimed_as_a_relocation():
    """Surfacing is a prompt to ask, so the match must be narrow enough to mean something."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        home = Path(h)
        new_root = Path(d) / "project-a"
        new_root.mkdir()
        other = home / ".claude" / "projects" / "-other-project"
        other.mkdir(parents=True)
        (other / f"{SESSION_ID}.jsonl").write_text(
            json.dumps({"type": "user", "cwd": "/other/project", "sessionId": SESSION_ID,
                        "timestamp": "2026-11-18T10:00:00.000Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": "x"}]}}) + "\n",
            encoding="utf-8")
        saved = {k: os.environ.get(k) for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        try:
            assert discover_sessions(str(new_root))["relocation_candidates"] == []
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_context_loads_record_what_entered_the_window():
    """Real FN: `loaded` was called unprovable without ever reading `attachment` records."""
    records = [
        _rec(type="attachment", timestamp="2026-11-18T10:00:00.000Z", attachment={
            "type": "skill_listing", "isInitial": True, "skillCount": 2,
            "names": ["second-skill", "demo-skill"], "content": "x" * 7000}),
        _rec(type="attachment", timestamp="2026-11-18T10:00:01.000Z", attachment={
            "type": "instructions", "files": [
                {"path": "/proj/CLAUDE.md", "type": "Project", "content": "y" * 1200}]}),
        _rec(type="attachment", timestamp="2026-11-18T10:00:02.000Z", attachment={
            "type": "mcp_instructions_delta", "addedNames": ["demo-server"],
            "addedBlocks": ["z" * 300]}),
    ]
    with _project(records) as (_root, path):
        tools = collect_session(str(path))["observed"]["tools"]
    assert tools["components_loaded"] == {"second-skill": 1, "demo-skill": 1, "demo-server": 1}
    assert tools["instructions_loaded"]["/proj/CLAUDE.md"] == {
        "type": "Project", "loads": 1, "chars": 1200}
    # A listing names many skills in one block, so its size stays unsplit.
    assert tools["shared_context_chars"] == {"skill_listing": 7000}



def test_root_rollup_survives_instruction_attachments():
    """A latent crash: the instructions rollup reused the loop variable holding the session
    index record, so `collect_root` blew up on any transcript carrying an instruction
    attachment — which is to say, on essentially every real project."""
    records = [
        _rec(type="attachment", timestamp="2026-11-19T10:00:00.000Z", attachment={
            "type": "instructions", "files": [
                {"path": "/proj/CLAUDE.md", "type": "Project", "content": "y" * 900}]}),
        _assistant("msg_r1", {"type": "text", "text": "ok"},
                   ts="2026-11-19T10:00:01.000Z", usage=_usage()),
    ]
    with _project(records) as (root, _path):
        observed = collect_root(str(root))["observed"]
    assert observed["sessions_collected"] == 1
    assert observed["instructions_loaded"]["/proj/CLAUDE.md"]["loads"] == 1


def test_bundled_agents_are_not_counted_as_loaded():
    """An agent listing names harness built-ins alongside anything declared. Counting the
    built-ins would repeat the Stage 1 false positive of bundled entities inflating a stack."""
    records = [
        _rec(type="attachment", timestamp="2026-11-18T10:00:00.000Z", attachment={
            "type": "agent_listing_delta", "isInitial": True,
            "addedTypes": ["Explore", "Plan", "my-reviewer"],
            "builtInTypes": ["Explore", "Plan"], "builtInPluginTypes": [],
            "addedLines": "x" * 400}),
    ]
    with _project(records) as (_root, path):
        tools = collect_session(str(path))["observed"]["tools"]
    assert tools["components_loaded"] == {"my-reviewer": 1}


def test_a_listing_records_description_depth_not_full_bodies():
    """Measured on real listings: a few thousand characters for a dozen-odd skills,
    formatted `- name: desc`.
    So being listed proves the skill was offered, not that its body entered context."""
    records = [
        _rec(type="attachment", timestamp="2026-11-18T10:00:00.000Z", attachment={
            "type": "skill_listing", "isInitial": True, "names": ["a", "b"],
            "content": "- a: does a thing\n- b: does another\n"}),
    ]
    with _project(records) as (_root, path):
        collect_session(str(path))
    from agent_profiler.traces import _context_loads
    load = next(iter(_context_loads(records[0])))
    assert load["depth"] == "description_only"
    assert load["shared"] is True


# -- whole-environment scope and the profiler's own exhaust ------------------------------


@contextmanager
def _multi_project(projects):
    """Several project keys under one ~/.claude/projects, as a real machine has."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        base = home / ".claude" / "projects"
        for key, sessions in projects.items():
            directory = base / key
            directory.mkdir(parents=True)
            for sid, records in sessions.items():
                (directory / f"{sid}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        saved = {k: os.environ.get(k) for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        try:
            yield root
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _probe_transcript():
    """What Stage 6 leaves behind: one trivial prompt, one response, no tool call."""
    return [
        _rec(type="user", timestamp=FIRST_TS,
             message={"role": "user", "content": [{"type": "text", "text": "Say: X"}]}),
        _assistant("probe-1", {"type": "text", "text": "X"}, ts=FIRST_TS,
                   usage={"input_tokens": 2, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0, "output_tokens": 1}),
    ]


def test_whole_environment_scope_covers_every_project():
    """A root-scoped look answers "what happened in this directory". The question the tool
    exists for is about the environment, and on a real machine a root-scoped corpus is a
    small fraction of it — most of the work happened under other paths."""
    with _multi_project({
        "-a-one": {"s1": _transcript()},
        "-a-two": {"s2": _transcript(), "s3": _transcript()},
    }) as root:
        scoped = discover_sessions(str(root))
        whole = discover_sessions(str(root), whole_environment=True)

    assert scoped["observed"]["sessions_found"] == 0
    assert whole["scope"] == "whole_environment"
    assert whole["observed"]["sessions_found"] == 3
    assert whole["observed"]["projects_found"] == 2
    assert whole["observed"]["sessions_by_project"] == {"-a-one": 1, "-a-two": 2}
    # Each session must say where it came from, or an aggregate cannot be broken down.
    assert all(s["project_key"] and "recorded_cwd" in s for s in whole["sessions"])


def test_the_profilers_own_probe_sessions_are_not_the_users_usage():
    """Real defect, at global scope: a large share of sessions were this tool's own `claude -p`
    probes. Counting them inflates the corpus and bills the profiler's spend to the person."""
    with _multi_project({
        "-a-real": {"s1": _transcript()},
        "-a-probes": {"p1": _probe_transcript(), "p2": _probe_transcript()},
    }) as root:
        whole = collect_root(str(root), whole_environment=True)

    obs = whole["observed"]
    assert obs["sessions_available"] == 3
    assert obs["sessions_collected"] == 1
    assert obs["self_generated"]["sessions"] == 2
    assert obs["self_generated"]["by"] == {"agent_profiler_probe": 2}
    assert obs["self_generated"]["tokens"] > 0
    # Separated, not hidden: the count and the reason both have to be in the output.
    assert "profiler" in obs["self_generated"]["note"]
    # And the probe project must not appear as if the user had worked there.
    assert "-a-probes" not in obs["by_project"]


def test_a_single_session_states_its_own_origin():
    with _multi_project({"-a-probes": {"p1": _probe_transcript()}}) as root:
        path = Path(os.environ["AGENT_PROFILER_CLAUDE_HOME"]) / "projects" / "-a-probes" / "p1.jsonl"
        one = collect_session(str(path))
        assert one["observed"]["origin"]["self_generated"] is True
        assert one["observed"]["origin"]["produced_by"] == "agent_profiler_probe"
        assert "Say: X" in one["observed"]["origin"]["evidence"]

    # A real session must never be labelled as the profiler's.
    with _project() as (_root, real):
        assert "origin" not in collect_session(str(real))["observed"]


def test_a_skill_listed_twice_in_one_session_counts_as_one_session():
    """Real defect: the report said a skill was listed into context in **more sessions than
    the corpus holds**. A session that resumes or compacts receives the listing again, and
    summing occurrences made a numerator its own denominator disproves.

    The aggregation trap again, in a fourth unit: occurrences are not sessions.
    """
    listing = {"type": "attachment",
               "attachment": {"type": "skill_listing", "names": ["alpha"],
                              "content": "- alpha: does a thing"}}
    twice = [_rec(type="user", timestamp=FIRST_TS,
                  message={"role": "user", "content": [{"type": "text", "text": "go"}]}),
             _rec(**listing, timestamp=FIRST_TS),
             _rec(**listing, timestamp=FIRST_TS),
             _assistant("m1", {"type": "text", "text": "ok"}, ts=FIRST_TS, usage=_usage())]
    with _multi_project({"-a-one": {"s1": twice}}) as root:
        observed = collect_root(str(root), whole_environment=True)["observed"]
    assert observed["sessions_collected"] == 1
    assert observed["components_loaded"]["alpha"] == 1


def test_self_generated_signatures_match_what_the_probes_actually_send():
    """The signature table is a copy of prompts that live in other modules, so it can go
    stale silently and quietly hand the profiler's own sessions back to the user. This
    builds each prompt from its real source and requires the collector to name it.

    Real defect: only the paired-probe prompt was recognised, so every session the other
    probes wrote counted as the user's own usage.
    """
    from agent_profiler.evaluate import PROBE_PROMPT
    from agent_profiler.judge import _build_prompt
    from agent_profiler.traces import _signature_match
    from agent_profiler.useful_probe import NEGATIVE_CONTROL, PROBE_SCENARIOS

    assert _signature_match(PROBE_PROMPT)[0] == "agent_profiler_probe"

    judged = _build_prompt([{"id": "a", "tool": "Read", "target": "x.py",
                             "between_commands": ["ls"]}])
    assert _signature_match(judged)[0] == "agent_profiler_judge"

    for scenario in (*PROBE_SCENARIOS, NEGATIVE_CONTROL):
        named = _signature_match(scenario["prompt"])
        assert named and named[0] == "agent_profiler_useful_probe", scenario["id"]


def test_a_real_prompt_is_not_claimed_as_the_profilers_own():
    """The cost of a false positive is silently deleting the user's real activity, so the
    matcher is anchored and a session with tool calls is never claimed however it opens."""
    from agent_profiler.traces import _signature_match

    assert _signature_match("Say: X please") is None
    assert _signature_match("Why did you ask '3 candidates. For each, was the second "
                            "identical call necessary?'") is None
    assert _signature_match("Write a git commit message for this change: auth refactor") is None

    # A second prompt is a person typing, so the session is theirs however it opened.
    kept_going = _probe_transcript() + [
        _rec(type="user", timestamp=FIRST_TS,
             message={"role": "user", "content": [{"type": "text", "text": "now explain it"}]}),
        _assistant("probe-2", {"type": "text", "text": "ok"}, ts=FIRST_TS, usage=_usage()),
    ]
    with _multi_project({"-a-mixed": {"s1": kept_going}}) as root:
        observed = collect_root(str(root), whole_environment=True)["observed"]
    assert observed["self_generated"]["sessions"] == 0
    assert observed["sessions_collected"] == 1


def test_a_probe_that_used_a_tool_is_still_the_profilers_own():
    """Real defect: requiring zero tool calls handed 7 usefulness-probe sessions back to
    the user. The probe asks for a commit message and the model runs `git log` on its own
    to check local style — tool use inside one `-p` run says nothing about who started it.
    An interrupted probe is likewise still a probe: a killed subprocess writes that marker.
    """
    from agent_profiler.useful_probe import PROBE_SCENARIOS

    def opened_with(text, tail):
        return [_rec(type="user", timestamp=FIRST_TS,
                     message={"role": "user", "content": [{"type": "text", "text": text}]}),
                *tail]

    used_a_tool = opened_with(PROBE_SCENARIOS[0]["prompt"], [
        _assistant("p1", {"type": "tool_use", "id": "toolu_X", "name": "Bash",
                          "input": {"command": "git log --oneline -15"}},
                   ts=FIRST_TS, usage=_usage()),
        _rec(type="user", timestamp=FIRST_TS,
             message={"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "toolu_X", "content": "abc fix"}]}),
        _assistant("p2", {"type": "text", "text": "fix: x"}, ts=FIRST_TS, usage=_usage()),
    ])
    interrupted = opened_with("Say: X", [
        _rec(type="user", timestamp=FIRST_TS,
             message={"role": "user",
                      "content": [{"type": "text", "text": "[Request interrupted by user]"}]}),
    ])

    with _multi_project({"-a-probes": {"s1": used_a_tool, "s2": interrupted}}) as root:
        observed = collect_root(str(root), whole_environment=True)["observed"]
    assert observed["self_generated"]["sessions"] == 2
    assert observed["self_generated"]["by"]["agent_profiler_useful_probe"] == 1
    assert observed["sessions_collected"] == 0


# -- the substring prefilters on the two hot patterns ------------------------------------


# Text chosen to attack the prefilters rather than the patterns: a Codex `exec` program
# carrying a whole file, near-misses that contain the pattern's other pieces but not its
# literal, and the real matches the prefilter must still let through.
_PREFILTER_CORPUS = (
    "",
    "ls",
    "cat /home/u/.claude/skills/fourth-skill/SKILL.md",
    "cat /home/u/.claude/skills/fourth-skill/skill.md",          # case matters to both
    "cat /home/u/.claude/skills/fourth-skill/SKILL.markdown",
    "cat /home/u/.agents/skills/.system/pdf/SKILL.md",
    "cat skills/fourth-skill/SKILL.md",                           # relative: dropped later
    "grep -r 'skills/' . && echo SKILL && echo .md",         # pieces, not the literal
    "echo 'SKILL.md'",                                       # the literal without a path
    "const r = await tools.exec_command({cmd:\"cat /home/u/.claude/skills/a/SKILL.md\"})",
    "const out = await tools.mcp__demo_server__list_tasks({limit: 5})",
    "const out = await tools.mcp__demo-server__list_tasks ({limit: 5})",
    "ALL_TOOLS.filter(x => /mcp__demo_server/.test(x.name))",   # names it, never calls it
    "tools.mcp__a__b",                                       # no open paren: not a call
    "mcp__a__b(1)",                                          # no tools. prefix
    # One long non-whitespace run is what made the unanchored pattern quadratic.
    "x" * 20000,
    "a/" * 8000 + "skills/fourth-skill/SKILL.md",
    "const program = `" + ("q" * 12000) + "`; await tools.exec_command({cmd:\"ls\"})",
)


def _without_short_circuit(pattern, text, keep):
    """What the function computed before the prefilter existed."""
    return keep(pattern, text) if text else ()


def test_the_substring_prefilters_change_speed_and_nothing_else():
    """`SKILL_BODY_PATH` was 5.8 of 28 seconds of a whole-environment `normalize`, almost
    all of it backtracking its unanchored `[^\\s"'`]*` across Codex programs that could
    never match. The fix is a literal check before the pattern runs, which is only sound
    while the literal stays *necessary* to the pattern — an edit to the pattern that drops
    it would silently stop finding real skill reads instead of failing loudly.

    So this compares the shipped function against the same work with the prefilter removed,
    and separately asserts the invariant the prefilter rests on.
    """
    from agent_profiler.traces import (
        MCP_CALL_IN_PROGRAM, MCP_CALL_MARKER, SKILL_BODY_MARKER, SKILL_BODY_PATH,
        _mcp_servers_called, _skill_body_paths,
    )

    def skills_the_long_way(text):
        found = []
        for match in SKILL_BODY_PATH.finditer(text):
            hit = match.group(0)
            if hit.startswith("/") and str(Path(hit)) not in found:
                found.append(str(Path(hit)))
        return tuple(found)

    def servers_the_long_way(text):
        return tuple(sorted({m.replace("-", "_") for m in MCP_CALL_IN_PROGRAM.findall(text)}))

    for text in _PREFILTER_CORPUS:
        assert _skill_body_paths("Bash", {"command": text}) == skills_the_long_way(text), text[:60]
        assert _mcp_servers_called("exec", text) == servers_the_long_way(text), text[:60]
        # The invariant: no match can exist in text the prefilter rejects.
        if SKILL_BODY_MARKER not in text:
            assert not SKILL_BODY_PATH.search(text), text[:60]
        if MCP_CALL_MARKER not in text:
            assert not MCP_CALL_IN_PROGRAM.search(text), text[:60]

    # And the prefilters must not have made the functions blind: the corpus has real hits.
    assert _skill_body_paths("Bash", {"command": _PREFILTER_CORPUS[2]}) == (
        "/home/u/.claude/skills/fourth-skill/SKILL.md",)
    assert _mcp_servers_called("exec", _PREFILTER_CORPUS[10]) == ("demo_server",)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} trace collector regression tests passed")


if __name__ == "__main__":
    main()

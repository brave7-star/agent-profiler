"""Regression suite for counterfactual evaluation.

No test here spends a token: `_probe` is replaced with a stub. What is tested is the gate
that refuses bad trades, the projection arithmetic, the rejection of degenerate probes,
and — above all — that measuring a *cost* is never reported as having found *waste*.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import evaluate as ev
from agent_profiler.cli import main as cli_main
from agent_profiler.evaluate import (
    MIN_EIG_RATIO,
    eig_gate,
    evaluate_root,
    measure_schema_cost,
    project_corpus_cost,
)

SESSION = "11111111-2222-3333-4444-555555555555"


def _rec(**kw):
    base = {"sessionId": SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, blocks, *, ts="2026-11-18T10:00:01.000Z"):
    return _rec(type="assistant", timestamp=ts, message={
        "id": msg_id, "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0, "output_tokens": 5},
        "content": blocks})


def _call(call_id, tool, **inp):
    return {"type": "tool_use", "id": call_id, "name": tool, "input": inp}


def _result(call_id):
    return _rec(type="user", timestamp="2026-11-18T10:00:02.000Z", message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "is_error": False, "content": "x"}]})


def _trace_with(mcp_calls):
    """A session that calls each named MCP tool once."""
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for i, tool in enumerate(mcp_calls):
        records += [_assistant(f"m{i}", [_call(f"t{i}", tool, q="x")]), _result(f"t{i}")]
    return records


@contextmanager
def _env(records, servers, *, probe=None):
    """A project root with a trace, an `.mcp.json`, and `_probe` stubbed out."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        (root / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
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
        original = ev._probe
        calls = []

        def fake_probe(config_path, timeout):
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
            loaded = bool(payload.get("mcpServers"))
            calls.append(loaded)
            return (probe or _default_probe)(loaded)

        ev._probe = fake_probe
        try:
            yield root, calls
        finally:
            ev._probe = original
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _default_probe(loaded):
    return {"ok": True, "total_context_tokens": 21795 if loaded else 18459,
            "cache_read": 10253, "cost_usd": 0.075}


# -- the gate ----------------------------------------------------------------------------


def test_gate_approves_a_large_recurring_quantity():
    """The Stage 6 case: a large recurring cost resolved by a measurement costing cents."""
    gate = eig_gate(name="mcp_schema:x", dollars_at_stake=31.46, evaluation_cost_usd=0.30)
    assert gate["decision"] == "approved"
    assert gate["eig_ratio"] > MIN_EIG_RATIO


def test_gate_refuses_the_stage_5_trade():
    """Stage 5 spent more than it found. That trade must not pass this gate."""
    gate = eig_gate(name="judge:screenshots", dollars_at_stake=0.55, evaluation_cost_usd=0.98)
    assert gate["decision"] == "refused"
    assert gate["eig_ratio"] < MIN_EIG_RATIO
    assert "below the" in gate["reason"]


def test_gate_refuses_a_behavioural_ab_on_a_real_corpus():
    """Re-running sessions costs orders of magnitude more than the schema it would test."""
    reference = ev._behavioural_ab_estimate(sessions=94, responses=5116, tokens=120_000_000)
    gate = eig_gate(name="behavioural_ab:x", dollars_at_stake=31.46,
                    evaluation_cost_usd=reference["estimated_usd"])
    assert gate["decision"] == "refused"
    assert reference["limits"]
    # The refusal must not depend on having assumed the expensive rate.
    low, high = reference["estimated_usd_range"]
    assert low < high and reference["estimated_usd"] == low


def test_gate_shows_its_arithmetic_either_way():
    for stake, cost in ((31.46, 0.30), (0.55, 0.98)):
        gate = eig_gate(name="x", dollars_at_stake=stake, evaluation_cost_usd=cost)
        assert gate["dollars_at_stake"] == stake
        assert gate["evaluation_cost_usd"] == cost
        assert gate["threshold"] == MIN_EIG_RATIO
        assert gate["reason"]


# -- measurement -------------------------------------------------------------------------


def test_paired_probes_difference_to_the_schema_size():
    with _env([], {"demo-server": {"command": "x"}}) as (_root, _calls):
        m = measure_schema_cost({"demo-server": {"command": "x"}}, repeats=2)
        assert m["ok"]
        assert m["schema_tokens"] == 21795 - 18459
        assert len(m["samples"]["with"]) == 2 and len(m["samples"]["without"]) == 2


def test_a_degenerate_probe_is_rejected_rather_than_averaged():
    """Real anomaly: one probe returned a context total of 2. Averaging it would have
    reported a negative schema size as if it were a measurement."""
    def probe(loaded):
        if loaded:
            return {"ok": False, "error": "implausible context total (2); probe did not load a prefix"}
        return _default_probe(loaded)

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert not m["ok"]
        assert m["errors"]


def test_probe_guards_against_an_empty_prefix():
    """The guard lives in `_probe` itself, so it holds for every caller."""
    import subprocess

    class _Done:
        returncode = 0
        stdout = json.dumps({"usage": {"input_tokens": 2, "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 0}, "total_cost_usd": 0.0})
        stderr = ""

    original_run, original_which = subprocess.run, ev.shutil.which
    subprocess.run = lambda *a, **k: _Done()
    ev.shutil.which = lambda _name: "/usr/bin/claude"
    try:
        out = ev._probe(Path(os.devnull), 10)
        assert not out["ok"] and "implausible" in out["error"]
    finally:
        subprocess.run = original_run
        ev.shutil.which = original_which


def test_a_flaky_probe_is_retried_within_budget():
    """Measured at 44% across 34 real probes, in either arm. The failure is detected, so a
    bounded retry recovers a replicated sample."""
    state = {"n": 0}

    def probe(loaded):
        state["n"] += 1
        if state["n"] == 2:
            return {"ok": False, "error": "implausible context total (2); probe did not load a prefix"}
        return _default_probe(loaded)

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert m["ok"] and m["retries_used"] == 1
        assert len(m["samples"]["with"]) == 2 and len(m["samples"]["without"]) == 2


def test_retrying_stops_at_the_budget():
    """A probe that keeps failing must not be retried indefinitely: repeated failure with
    no new information is the thing the retry budget exists to stop."""
    with _env([], {"s": {"command": "x"}},
              probe=lambda loaded: {"ok": False, "error": "timeout"}) as (_root, calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert not m["ok"]
        assert m["errors"]
        # One first attempt per arm, plus that arm's own retry budget and nothing more.
        assert len(calls) <= 2 * (1 + ev.PROBE_RETRY_PER_ARM)


def test_one_arms_bad_streak_does_not_starve_the_other():
    """The budget is per arm because an arm with no sample makes the whole pairing useless.
    A bad streak in the first arm used to consume the retries the second one needed, and at
    a 44% failure rate that is a likely run, not a corner case."""
    state = {"n": 0}

    def probe(loaded):
        state["n"] += 1
        # Three failures in the first arm, then one in the second: survivable per arm,
        # fatal under a single shared budget of three.
        if state["n"] in (1, 2, 3, 6):
            return {"ok": False, "error": "timeout"}
        return _default_probe(loaded)

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert m["ok"]
        assert len(m["samples"]["with"]) == 2 and len(m["samples"]["without"]) == 2
        assert m["probes_failed"] == 4


def test_failed_probe_spend_is_reported_separately_from_the_measurement():
    """Money that bought a measurement and money that bought nothing are different facts,
    and only the second one is avoidable. One total would hide it."""
    def probe(loaded):
        if not loaded:
            return {"ok": False, "error": "timeout", "cost_usd": 0.07}
        return _default_probe(loaded)

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert m["probes_failed"] == 4 and m["probes_attempted"] == 6
        assert m["observed_failure_rate"] == round(4 / 6, 3)
        assert abs(m["failed_probe_usd"] - 4 * 0.07) < 1e-6
        assert m["failed_probe_usd"] < m["cost_usd"]
        assert "bought no measurement" in m["failed_probe_note"]


def test_a_failed_probe_with_no_billed_figure_is_not_counted_as_free():
    """A probe that returns no cost was not free; falling back to the measured per-probe
    figure keeps the avoidable spend from rounding itself away. That it is an assumption
    and not a reading has to be said, because it is an upper bound."""
    with _env([], {"s": {"command": "x"}},
              probe=lambda loaded: {"ok": False, "error": "timeout"}) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert m["failed_probe_usd"] == round(m["probes_failed"] * ev.PROBE_COST_USD, 6)
        assert m["failed_probe_usd"] > 0
        assert m["failed_probe_usd_assumed"] == m["probes_failed"]
        assert "is assumed" in m["failed_probe_note"]


def test_the_common_failure_mode_is_not_inflated_into_dollars():
    """The degenerate probe reports two input tokens, so it is billed next to nothing. The
    honest figure is what the envelope says, and the note has to keep the flake from
    reading as money it did not cost."""
    def probe(loaded):
        if not loaded:
            return {"ok": False, "error": "implausible context total (2)", "cost_usd": 0.0004}
        return _default_probe(loaded)

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert m["failed_probe_usd_assumed"] == 0
        assert m["failed_probe_usd"] < 0.01
        assert "latency and a lost sample" in m["failed_probe_note"]


def test_the_run_reports_its_own_failed_probe_spend():
    """`profiler_cost` is where the profiler accounts for itself, so the avoidable part of
    its spend has to surface there and not only inside one measurement."""
    state = {"n": 0}

    def probe(loaded):
        state["n"] += 1
        if state["n"] == 1:
            return {"ok": False, "error": "timeout", "cost_usd": 0.07}
        return _default_probe(loaded)

    with _env(_trace_with(["Read"]), {"s": {"command": "x"}}, probe=probe) as (root, calls):
        r = evaluate_root(str(root), measure=True)
        cost = r["profiler_cost"]
        assert cost["llm_calls"] == len(calls)
        assert cost["failed_probes"] == 1
        assert abs(cost["failed_probe_usd"] - 0.07) < 1e-6


def test_two_indistinguishable_arms_are_not_a_measurement():
    """The real case: the server's token had expired, so it connected to nothing and
    registered no tools. Both arms came back at the same size, the difference was a handful
    of tokens, and the
    stage reported `ok: true, reproducible: true` and projected a negative cost. A reproducible zero
    reads as more trustworthy than a noisy real figure, so it has to be refused loudly."""
    def probe(loaded):
        base = _default_probe(loaded)
        # Identical prefix in both arms, differing only by probe noise.
        base["total_context_tokens"] = 19106 if loaded else 19109
        return base

    with _env(_trace_with(["Read"]), {"s": {"command": "x"}}, probe=probe) as (root, _calls):
        m = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert not m["ok"] and m["inconclusive"]
        assert "noise band" in m["reason"] and "fails to connect" in m["reason"]
        # The evidence stays visible; only the conclusion is withheld.
        assert m["samples"]["with"] and m["samples"]["without"]

        r = evaluate_root(str(root), measure=True)
        entry = r["evaluations"][0]
        assert entry["status"] == "measurement_inconclusive"
        assert entry["reason"]
        # Above all: no dollar projection, because there is nothing to project.
        assert "projection" not in entry


def test_reproducibility_is_reported_not_assumed():
    spread = {"n": 0}

    def probe(loaded):
        base = _default_probe(loaded)
        if loaded:
            spread["n"] += 1
            base["total_context_tokens"] += spread["n"] * 2000
        return base

    with _env([], {"s": {"command": "x"}}, probe=probe) as (_root, _calls):
        wobbly = measure_schema_cost({"s": {"command": "x"}}, repeats=2)
        assert wobbly["ok"] and wobbly["reproducible"] is False
        steady = measure_schema_cost({"s": {"command": "x"}}, repeats=1)
        # A single sample cannot establish reproducibility either.
        assert steady["reproducible"] is False


# -- projection --------------------------------------------------------------------------


def test_projection_arithmetic():
    p = project_corpus_cost(3334, sessions=94, responses=5116)
    assert p["cache_creation_tokens"] == 94 * 3334
    assert p["cache_read_tokens"] == 5116 * 3334
    assert p["total_tokens"] == 94 * 3334 + 5116 * 3334
    assert p["estimated_usd"] > 0


def test_projection_declares_its_assumptions():
    """A projection is Estimated, and must say what it rests on — including the rates."""
    p = project_corpus_cost(3334, sessions=94, responses=5116)
    assert len(p["assumptions"]) >= 3
    assert any("cached prefix" in a for a in p["assumptions"])
    assert any("rates" in a.lower() for a in p["assumptions"])


# -- honesty -----------------------------------------------------------------------------


def test_a_used_server_is_never_called_waste():
    """The core Stage 6 restraint: demo-server costs tokens *and* did 60 real calls."""
    with _env(_trace_with(["mcp__demo-server__get_task"] * 3), {"demo-server": {"command": "x"}}) as (root, _c):
        r = evaluate_root(str(root), measure=True)
        entry = next(e for e in r["evaluations"] if e["server"] == "demo-server")
        assert entry["status"] == "measured"
        assert entry["calls_observed"] == 3
        assert "Cost is not waste" in entry["interpretation"]
        blob = json.dumps(r).lower()
        for banned in ("should be removed", "remove this", "wasted", "useless", "would save"):
            assert banned not in blob, banned


def test_an_empty_corpus_is_not_reported_as_a_measurement_of_zero():
    """Real defect, on a root whose history is filed under its old working directory:
    the stage reported `calls_observed: 0` and nothing at stake, which reads like a clean
    result. Nothing was observed, which is a different fact from nothing being used."""
    with _env([], {"s": {"command": "x"}}) as (root, _calls):
        # No transcript at all, which is what a relocated project looks like from here.
        for recorded in (Path(os.environ["AGENT_PROFILER_CLAUDE_HOME"]) / "projects").rglob("*.jsonl"):
            recorded.unlink()
        r = evaluate_root(str(root))
        assert r["corpus"]["sessions"] == 0
        assert r["corpus"]["blocker"] == "NO_TRACES_COLLECTED"
        entry = r["evaluations"][0]
        assert entry["status"] == "no_corpus"
        assert entry["blocker"] == "NO_TRACES_COLLECTED"
        # A proven zero would be a number; an unobserved quantity must not be one.
        assert entry["calls_observed"] is None
        assert "not a measurement of zero" in entry["reason"]
        assert "gate" not in entry


def test_an_uncalled_server_is_not_turned_into_a_removal_recommendation():
    with _env(_trace_with(["Read"]), {"ghost": {"command": "x"}}) as (root, _calls):
        r = evaluate_root(str(root), measure=True)
        entry = next(e for e in r["evaluations"] if e["server"] == "ghost")
        assert entry["calls_observed"] == 0
        assert "no call to it was observed" in entry["interpretation"]
        # Never called is not evidence of low value; the text must say so, not conclude.
        assert "depends on work these sessions did not cover" in entry["interpretation"]


def test_the_artifact_records_what_it_was_told_to_leave_alone():
    """This is the one stage that connects to a third party, so the instruction not to has
    to be in its record — including when no instruction was found.

    Written after a standing exclusion silently stopped applying and the artifact looked
    identical either way: the server simply read `not_measured`, exactly as it would for a
    user who never asked for one.
    """
    from agent_profiler.core import load_exclusions

    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}) as (root, calls):
        bare = evaluate_root(str(root), measure=False)
        assert bare["exclusions"]["source"] is None
        assert bare["exclusions"]["mcp"] == []
        # Where it looked, so silence is attributable rather than ambiguous.
        assert bare["exclusions"]["searched"]

        (Path(root) / "agent-profiler-exclude.json").write_text(
            json.dumps({"mcp": ["s"], "reason": "private"}), encoding="utf-8")
        held = evaluate_root(str(root), measure=True,
                             exclusions=load_exclusions(None, Path(root)))
    assert held["exclusions"]["mcp"] == ["s"]
    assert held["exclusions"]["reason"] == "private"
    entry = next(e for e in held["evaluations"] if e["server"] == "s")
    assert entry["status"] == "excluded_by_request"
    # And the guarantee itself: measure=True, yet nothing was launched for that server.
    assert calls == [], calls


def test_omitting_the_exclusions_argument_means_find_them_not_ignore_them():
    """A caller who forgets the argument must not start sending traffic the user withheld.

    `exclusions=None` used to mean "nothing is excluded", which is the dangerous reading of
    a missing argument on the one stage that contacts a third party.
    """
    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}) as (root, calls):
        (Path(root) / "agent-profiler-exclude.json").write_text(
            json.dumps({"mcp": ["s"], "reason": "private"}), encoding="utf-8")
        r = evaluate_root(str(root), measure=True)  # deliberately no exclusions passed
    assert r["exclusions"]["mcp"] == ["s"]
    assert next(e for e in r["evaluations"] if e["server"] == "s")["status"] \
        == "excluded_by_request"
    assert calls == [], calls


def test_dry_mode_spends_nothing():
    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}) as (root, calls):
        r = evaluate_root(str(root), measure=False)
        assert calls == []
        assert r["profiler_cost"]["cost_usd"] == 0.0
        assert r["profiler_cost"]["llm_calls"] == 0
        assert all(e["status"] == "not_measured" for e in r["evaluations"])
        assert all("gate" in e for e in r["evaluations"])


def test_self_cost_is_reported():
    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}) as (root, calls):
        r = evaluate_root(str(root), measure=True)
        assert len(calls) == 2 * ev.PROBE_REPEATS
        assert r["profiler_cost"]["llm_calls"] == 2 * ev.PROBE_REPEATS
        assert r["profiler_cost"]["cost_usd"] > 0
        assert r["profiler_cost"]["elapsed_ms"] > 0


def test_failed_probes_still_count_against_self_cost():
    """A probe that failed still cost money; omitting it would understate the spend."""
    state = {"n": 0}

    def probe(loaded):
        state["n"] += 1
        if state["n"] == 2:
            return {"ok": False, "error": "implausible context total (2)"}
        return _default_probe(loaded)

    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}, probe=probe) as (root, calls):
        r = evaluate_root(str(root), measure=True)
        assert r["profiler_cost"]["llm_calls"] == len(calls)


def test_a_measurement_failure_degrades_without_a_projection():
    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}},
              probe=lambda loaded: {"ok": False, "error": "timeout"}) as (root, _calls):
        r = evaluate_root(str(root), measure=True)
        entry = r["evaluations"][0]
        assert entry["status"] == "measurement_failed"
        assert "projection" not in entry


def test_an_unreconstructable_server_is_not_guessed_at():
    """The npx/search-server case: if the launch config cannot be recovered, say so."""
    with _env(_trace_with(["Read"]), {"s": {"command": "x"}}) as (root, _calls):
        (root / ".mcp.json").write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        original = ev._server_config
        ev._server_config = lambda base, name: None
        try:
            r = evaluate_root(str(root), measure=True)
            for entry in r["evaluations"]:
                assert entry["status"] == "unmeasurable"
                assert "could not be reconstructed" in entry["reason"]
        finally:
            ev._server_config = original


def test_report_is_marked_read_only():
    with _env(_trace_with(["mcp__s__go"]), {"s": {"command": "x"}}) as (root, _calls):
        r = evaluate_root(str(root), measure=False)
        assert r["read_only"] is True
        assert r["schema_version"] == ev.EVALUATE_SCHEMA_VERSION


def test_cli_errors():
    with _env([], {"s": {"command": "x"}}) as (root, _calls):
        assert cli_main(["evaluate", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["evaluate", "--root", ""]) == 2
        assert cli_main(["evaluate", "--root", str(root / "nope")]) == 2


def test_probes_and_judge_calls_close_stdin():
    """Real defect, found by a runaway probe. `capture_output=True` redirects stdout and
    stderr but leaves stdin inherited, and `claude -p` appends piped stdin to its prompt.
    Two probes launched from a shell heredoc received the parent's unconsumed script text
    as prompt, ran 27 and 41 agentic turns, and cost 19x a controlled probe.

    Asserted at the call site rather than behaviourally: the alternative is spawning a
    real subprocess, and this is exactly the kind of guard that must hold without spending
    anything to check it."""
    import inspect as _inspect

    from agent_profiler import judge as ju
    from agent_profiler import useful_probe as up

    for module, func in ((ev, ev._probe), (ju, ju._invoke), (up, up._probe)):
        src = _inspect.getsource(func)
        assert "subprocess.run" in src, func.__name__
        assert "stdin=subprocess.DEVNULL" in src, (
            f"{module.__name__}.{func.__name__} spawns the CLI without closing stdin; "
            "its prompt would depend on the caller's unconsumed input")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} evaluate regression tests passed (0 tokens spent)")


if __name__ == "__main__":
    main()

"""Invocation accounting through real measurement commands, with no live subprocesses."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_profiler import codex_probe as cp, useful_probe as up
from test_measurement_batch import _environment, _command
from test_useful_probe import FULL_RULES, COMPLIANT, _codex_probe_returning


def test_failed_instruction_pairs_and_mixed_harnesses_keep_every_call():
    with _environment(["claude_md", "agents_md"]) as (root, files):
        answers = [
            {"ok": True, "ctx": 2000, "cost_usd": 0.02},
            {"ok": False, "error": "failed second arm", "cost_usd": 0.03},
            {"ok": False, "error": "failed first arm", "cost_usd": 0.04},
        ]
        with patch.object(up, "_probe", side_effect=answers) as claude, \
                patch.object(cp, "_controlled_home", side_effect=lambda p: p), \
                patch.object(cp, "_codex_probe", return_value={"ok": False, "error": "timeout"}) as codex:
            result = _command(root, "measure-instructions")
        cost = result["profiler_cost"]
        assert cost["llm_calls"] == claude.call_count + codex.call_count == 5
        assert (cost["claude_runs"], cost["codex_runs"]) == (3, 2)
        assert cost["failed_calls"] == 4
        assert cost["known_cost_usd"] == 0.09 and cost["unknown_cost_calls"] == 2
        assert cost["cost_usd"] is None
        assert len(result["skipped"]) == 2 and not result["instruction_measurements"]
        assert result["skipped"][0]["cost_usd"] == 0.09
        assert result["harnesses"] == ["claude_code", "codex"]


def test_skill_cost_counts_retries_even_when_no_pair_survives():
    with _environment(["skills"]) as (root, files):
        with patch.object(up, "_probe", return_value={
                "ok": False, "error": "rejected", "cost_usd": 0.01}) as probe, \
                patch.object(cp.time, "sleep"):
            result = _command(root, "measure-skills")
        calls = 2 * (cp.ARM_RETRIES + 1)
        cost = result["profiler_cost"]
        assert cost["claude_runs"] == cost["llm_calls"] == probe.call_count == calls
        assert cost["cost_usd"] == cost["spend_usd"] == round(calls * 0.01, 6)
        assert cost["failed_calls"] == calls and cost["unknown_cost_calls"] == 0
        assert result["skipped"][0]["cost_usd"] == cost["cost_usd"]


def test_unknown_retry_cost_survives_a_successful_instruction_measurement_and_resume():
    with _environment(["claude_md"]) as (root, files):
        path = Path(files[0]["path"])
        path.write_text(FULL_RULES)
        seen = []

        def probe(prompt, workdir, instruction, timeout):
            seen.append(prompt)
            if len(seen) == 1:
                return {"ok": False, "error": "timeout", "cost_usd": None}
            return {"ok": True, "text": "391" if "17 times 23" in prompt else COMPLIANT,
                    "ctx": 2000, "cost_usd": 0.02}

        with patch.object(up, "_probe", side_effect=probe), patch.object(up, "RETRY_BACKOFF_SECONDS", 0):
            result = _command(root, "measure-useful", "--repeats", "1")
        row = result["useful_measurements"][0]
        assert row["blocker"] == "NO_SEPARATION"
        assert row["llm_calls"] == len(seen) == row["probes_attempted"] + 1
        for cost in (row, result["profiler_cost"]):
            assert cost["cost_usd"] is None and cost["unknown_cost_calls"] == 1
            assert cost["known_cost_usd"] == round((len(seen) - 1) * 0.02, 6)
            assert cost["failed_calls"] == 1
        with patch.object(up, "_probe", side_effect=AssertionError("resume must not call")):
            resumed = _command(root, "measure-useful", "--repeats", "1", "--resume")
        assert resumed["profiler_cost"]["llm_calls"] == 0
        assert resumed["profiler_cost"]["cost_usd"] == 0


def test_skill_presence_cost_is_included_when_it_blocks_or_allows_compliance():
    for present in (False, True):
        for presence_cost in (0.07, 0.0, None):
            with _environment(["skills"]) as (root, files):
                path = Path(files[0]["path"])
                path.write_text(FULL_RULES)

                def probe(prompt, *args, **kwargs):
                    if "List the names of the skills" in prompt:
                        return {"ok": True, "text": path.parent.name if present else "absent",
                                "ctx": 2000, "cost_usd": presence_cost}
                    return {"ok": True, "text": "391" if "17 times 23" in prompt else COMPLIANT,
                            "ctx": 2000, "cost_usd": 0.01}

                with patch.object(up, "_probe", side_effect=probe) as called:
                    result = _command(root, "measure-useful", "--repeats", "1")
                row = (result["useful_measurements"] if present else result["skipped"])[0]
                for cost in (row, result["profiler_cost"]):
                    assert cost["llm_calls"] == called.call_count
                    assert cost["unknown_cost_calls"] == int(presence_cost is None)
                    known = round((presence_cost or 0) + (called.call_count - 1) * 0.01, 6)
                    assert cost["known_cost_usd"] == known
                    assert cost["cost_usd"] == (None if presence_cost is None else known)
                if not present:
                    assert called.call_count == 1
                    assert row["blocker"] == "COMPONENT_NOT_LOADED_BY_ARM"


def test_codex_usefulness_has_unknown_dollars_at_batch_level_too():
    with _environment(["agents_md"]) as (root, files):
        Path(files[0]["path"]).write_text(FULL_RULES)
        with _codex_probe_returning(), patch.object(up, "_controlled_codex_home", side_effect=lambda p: p):
            result = _command(root, "measure-useful", "--repeats", "1")
        cost = result["profiler_cost"]
        assert cost["llm_calls"] == cost["unknown_cost_calls"] == 6
        assert cost["cost_usd"] is None and cost["known_cost_usd"] == 0


def test_failed_presence_and_abandoned_compliance_keep_billed_retries():
    with _environment(["skills"]) as (root, files):
        Path(files[0]["path"]).write_text(FULL_RULES)
        with patch.object(up, "_probe", return_value={
                "ok": False, "error": "failed", "cost_usd": 0.03}) as probe, \
                patch.object(up, "RETRY_BACKOFF_SECONDS", 0):
            result = _command(root, "measure-useful", "--repeats", "1")
        row = result["useful_measurements"][0]
        calls = 1 + up.CONSECUTIVE_FAILURE_LIMIT * up.RETRY_BUDGET
        assert row["abandoned"] and not row["skill_presence"]["checked"]
        for cost in (row, result["profiler_cost"]):
            assert cost["llm_calls"] == cost["failed_calls"] == probe.call_count == calls
            assert cost["cost_usd"] == round(calls * 0.03, 6)


def test_successful_cost_measurements_keep_unknown_cost_and_exact_call_count():
    for kind in ("claude_md", "agents_md", "skills"):
        with _environment([kind]) as (root, files):
            def probe(*args, **kwargs):
                return {"ok": True, "ctx": 2000, "input_tokens": 2000,
                        "text": "baseline\n" + ("0" if "--plugin-dir" in kwargs.get("extra_args", ()) else ""),
                        "cost_usd": None}

            with patch.object(up, "_probe", side_effect=probe), \
                    patch.object(cp, "_codex_probe", side_effect=probe), \
                    patch.object(cp, "_controlled_home", side_effect=lambda p: p):
                command = "measure-skills" if kind == "skills" else "measure-instructions"
                result = _command(root, command)
            key = "skill_measurements" if kind == "skills" else "instruction_measurements"
            row = result[key][0]
            assert row["inconclusive"]
            for cost in (row, result["profiler_cost"]):
                assert cost["llm_calls"] == cost["unknown_cost_calls"] == 4
                assert cost["failed_calls"] == 0 and cost["cost_usd"] is None


def test_transport_timeout_or_missing_usage_never_claims_free():
    with _environment([]) as (root, files):
        with patch.object(up.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 1)), \
                patch.object(up.shutil, "which", return_value="fixture"):
            assert up._probe("x", root, None, 1)["cost_usd"] is None
            assert up._codex_probe("x", root, root, 1)["cost_usd"] is None
        envelope = {"usage": {"input_tokens": 2000}, "result": "x"}
        with patch.object(up.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(envelope), stderr="")):
            assert up._probe("x", root, None, 1)["cost_usd"] is None
        with patch.object(up.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="failed")), \
                patch.object(up.shutil, "which", return_value="fixture"):
            assert up._codex_probe("x", root, root, 1)["cost_usd"] is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} probe-cost regression tests passed (0 tokens spent)")

"""CLI failures must retain the envelope instead of masquerading as empty stderr."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent_profiler import evaluate, judge, useful_probe
from agent_profiler.claude_result import read_result


def _consumers(done):
    with tempfile.TemporaryDirectory() as raw, \
            patch.object(subprocess, "run", return_value=done), \
            patch.object(evaluate.shutil, "which", return_value="fixture-claude"):
        root = Path(raw)
        return [useful_probe._probe("fixture", root, None, 1),
                evaluate._probe(root / "empty.json", 1), judge._invoke("fixture", 1)]


def test_nonzero_exits_keep_stdout_error_and_billed_cost_in_every_consumer():
    done = subprocess.CompletedProcess([], 1, stderr="", stdout=json.dumps({
        "type": "result", "subtype": "success", "is_error": True,
        "result": "API Error: service is overloaded", "api_error_status": 529,
        "total_cost_usd": 0.007,
    }))
    for result in _consumers(done):
        assert result["ok"] is False
        assert "overloaded" in result["error"]
        assert result["api_error_status"] == 529
        assert result["cost_usd"] == 0.007


def test_a_zero_exit_and_success_subtype_cannot_override_the_error_flag():
    envelope = {"subtype": "success", "is_error": True, "result": "request rejected",
                "total_cost_usd": 0.0, "usage": {"input_tokens": 2500}}
    for result in _consumers(subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(envelope), stderr="")):
        assert result["ok"] is False and "request rejected" in result["error"]
    envelope["is_error"] = False
    envelope["result"] = "[]"
    for result in _consumers(subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(envelope), stderr="")):
        assert result["ok"] is True


def test_missing_cost_stays_unknown_and_non_json_diagnostics_survive():
    for stdout, stderr, expected in [("", "Error: bad flag", "bad flag"),
                                     ("gateway unavailable", "", "gateway unavailable")]:
        _, failure = read_result(subprocess.CompletedProcess([], 1, stdout, stderr))
        assert expected in failure["error"]
        assert failure["cost_usd"] is None
    for stdout in ("not-json", "[]", "null"):
        _, failure = read_result(subprocess.CompletedProcess([], 0, stdout, ""))
        assert failure["ok"] is False


def test_failure_diagnostics_are_bounded_and_use_the_existing_redactor():
    secret = "fixture-value"
    _, failure = read_result(subprocess.CompletedProcess([], 1, json.dumps({
        "is_error": True, "result": "API_KEY=" + secret + " " + "detail " * 200,
        "errors": ["additional diagnostic"], "unrelated": "private field",
    }), ""))
    assert secret not in failure["error"] and "[REDACTED]" in failure["error"]
    assert "private field" not in failure["error"]
    assert len(failure["error"]) <= 520


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} Claude-result tests passed (0 tokens spent)")

"""Exercise interrupted and resumed commands without launching a paid probe."""

import io
import json
import os
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_profiler import cli, core, codex_probe as cp, useful_probe as up, store


@contextmanager
def _environment(kinds):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        files = []
        for index, kind in enumerate(kinds):
            name = {"agents_md": "AGENTS.md", "claude_md": "CLAUDE.md", "skills": "SKILL.md"}[kind]
            path = root / ("skills" if kind == "skills" else "projects") / str(index) / name
            path.parent.mkdir(parents=True)
            path.write_text(f"fixture {index}: End the title with the issue reference.\n")
            files.append({"path": str(path), "kind": kind, "origin": "declared",
                          "scope": "project", "harness": "codex" if kind == "agents_md" else "claude_code"})
        with patch.dict(os.environ, {"AGENT_PROFILER_HOME": str(root),
                                     "AGENT_PROFILER_STORE": str(root / "store")}), \
                patch.object(core, "inspect", return_value={"files": files}), \
                patch("subprocess.run", side_effect=AssertionError("no real subprocess allowed")):
            yield root, files


def _cost(path, **kwargs):
    return {"ok": True, "file": str(path / "SKILL.md" if path.is_dir() else path),
            "mean_tokens": 180, "differences": [180, 180], "pair_spread_tokens": 0,
            "reproducible": True, "pairs": [{"difference": 180}] * 2, "cost_usd": 0.01}


def _useful(path, **kwargs):
    return {"ok": False, "file": str(path / "SKILL.md" if path.is_dir() else path),
            "useful": "unproven", "blocker": "NO_SEPARATION",
            "negative_control": {"held": True}, "probes_attempted": 12,
            "probes_failed": 0, "records": [{"valid": True}] * 12}


def _command(root, command, *flags):
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        assert cli.main([command, "--root", str(root), "--lang", "en", *flags]) == 0
    return json.loads(output.getvalue())


def test_completed_subjects_survive_interruption_and_resume_without_duplicate_spend():
    for command, kind, module, function, result, key in (
            ("measure-instructions", "claude_md", cp, "measure_claude_instruction_cost", _cost, "instruction_measurements"),
            ("measure-skills", "skills", cp, "measure_skill_cost", _cost, "skill_measurements"),
            ("measure-useful", "agents_md", up, "measure_instruction_usefulness", _useful, "useful_measurements")):
        with _environment([kind, kind]) as (root, files):
            calls = []

            def interrupted(path, **kwargs):
                calls.append(path)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return result(path, **kwargs)

            with patch.object(module, function, side_effect=interrupted):
                try:
                    _command(root, command)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("fixture did not interrupt")
            saved = store.stored(keys=(key,))
            assert saved and len(saved[key]) == 1, "first result must exist before the batch returns"
            with patch.object(module, function, side_effect=result) as probe:
                resumed = _command(root, command, "--resume")
                assert probe.call_count == 1
            assert any(s["reason"] == "already_measured_same_content" for s in resumed["skipped"])
            assert len(store.stored(keys=(key,))[key]) == 2, "final CLI save must not duplicate checkpoints"
            with patch.object(module, function, side_effect=AssertionError("already measured")):
                _command(root, command, "--resume")


def test_resume_reads_content_not_size_or_mtime_and_does_not_certify_legacy_results():
    with _environment(["claude_md"]) as (root, files):
        path = Path(files[0]["path"])
        store.remember({"instruction_measurements": [_cost(path)]}, stage="legacy")
        with patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost) as probe:
            _command(root, "measure-instructions", "--resume")
            assert probe.call_count == 1, "legacy rows have no proof of content equality"
            previous = path.stat()
            path.write_text(path.read_text().replace("fixture", "changed"))
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            _command(root, "measure-instructions", "--resume")
            assert probe.call_count == 2
            _command(root, "measure-instructions", "--resume", "--pairs", "3")
            assert probe.call_count == 3, "a request for more pairs must not reuse fewer"
            _command(root, "measure-instructions")
            assert probe.call_count == 4, "omitting resume requests a new reading"


def test_resume_keeps_harness_and_measurement_kind_separate_even_for_symlinked_files():
    with _environment(["agents_md", "claude_md"]) as (root, files):
        agents, claude = (Path(f["path"]) for f in files)
        agents.unlink()
        agents.symlink_to(claude)
        with patch.object(cp, "measure_instruction_cost", side_effect=_cost) as codex, \
                patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost) as anthropic:
            with patch.object(core, "inspect", return_value={"files": files[1:]}):
                _command(root, "measure-instructions")
            _command(root, "measure-instructions", "--resume")
            assert codex.call_count == anthropic.call_count == 1
            _command(root, "measure-instructions", "--resume")
            assert codex.call_count == anthropic.call_count == 1
        with patch.object(up, "measure_instruction_usefulness", side_effect=_useful) as useful:
            _command(root, "measure-useful", "--resume")
            assert useful.call_count == 2, "cost evidence cannot skip usefulness"


def test_identical_instruction_files_at_different_paths_do_not_share_resume_state():
    with _environment(["claude_md", "claude_md"]) as (root, files):
        first, second = (Path(f["path"]) for f in files)
        second.write_bytes(first.read_bytes())
        with patch.object(core, "inspect", return_value={"files": files[:1]}), \
                patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost):
            _command(root, "measure-instructions")
        with patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost) as probe:
            _command(root, "measure-instructions", "--resume")
            assert probe.call_count == 1 and probe.call_args.args[0] == second


def test_skill_resume_includes_supporting_files_and_both_measurement_commands():
    with _environment(["skills"]) as (root, files):
        reference = Path(files[0]["path"]).parent / "reference.txt"
        reference.write_text("old rule")
        with patch.object(cp, "measure_skill_cost", side_effect=_cost) as cost, \
                patch.object(up, "measure_skill_usefulness", side_effect=_useful) as useful:
            for command in ("measure-skills", "measure-useful"):
                _command(root, command, "--resume")
                _command(root, command, "--resume")
            assert cost.call_count == useful.call_count == 1
            reference.write_text("new rule")
            for command in ("measure-skills", "measure-useful"):
                _command(root, command, "--resume")
            assert cost.call_count == useful.call_count == 2


def test_two_failed_components_stop_that_harness_but_leave_the_other_available():
    with _environment(["claude_md", "claude_md", "agents_md", "claude_md"]) as (root, files):
        def dead(path, **kwargs):
            return {"ok": False, "reason": "no pair completed", "pairs": [{"error": "gateway unavailable"}]}

        with patch.object(cp, "measure_claude_instruction_cost", side_effect=dead) as claude, \
                patch.object(cp, "measure_instruction_cost", side_effect=_cost) as codex:
            result = _command(root, "measure-instructions")
        assert claude.call_count == 2 and codex.call_count == 1
        assert "claude_code" in result["stopped_harnesses"]
        assert any(s["reason"].startswith("batch_stopped") for s in result["skipped"])
        assert len(store.stored()["instruction_measurements"]) == 1


def test_an_abandoned_usefulness_run_stops_later_instructions_and_skills():
    with _environment(["claude_md", "claude_md", "skills", "agents_md"]) as (root, files):
        def interrupted(path, **kwargs):
            if path.name == "AGENTS.md":
                return _useful(path)
            return {"ok": False, "useful": "unproven", "blocker": "NO_VALID_RUNS",
                    "probes_attempted": 6, "probes_failed": 6,
                    "abandoned": {"last_error": "gateway unavailable"}}

        with patch.object(up, "measure_instruction_usefulness", side_effect=interrupted) as instructions, \
                patch.object(up, "measure_skill_usefulness", side_effect=AssertionError("harness already stopped")):
            result = _command(root, "measure-useful")
        assert instructions.call_count == 2
        assert "claude_code" in result["stopped_harnesses"]
        assert len(store.stored(keys=store.USEFUL_KEYS)["useful_measurements"]) == 1


def test_mixed_routes_stop_cost_batch_and_unstable_stored_readings_are_not_retried():
    with _environment(["agents_md", "agents_md"]) as (root, files):
        with patch.object(cp, "measure_instruction_cost", return_value={
                "ok": False, "blocker": "ARMS_NOT_SERVED_ALIKE", "reason": "different routes"}) as probe:
            result = _command(root, "measure-instructions")
        assert probe.call_count == 1 and "codex" in result["stopped_harnesses"]
        assert store.stored() is None
        with patch.object(cp, "measure_instruction_cost", side_effect=lambda path, **kw: {
                **_cost(path), "reproducible": False}) as probe:
            _command(root, "measure-instructions")
        with patch.object(cp, "measure_instruction_cost", side_effect=AssertionError("diagnose first")):
            resumed = _command(root, "measure-instructions", "--resume")
        assert all("unstable" in s["reason"] for s in resumed["skipped"])


def test_the_route_alone_marks_a_stored_reading_unstable_on_resume():
    """Each instability marker has to be reached on its own.

    The test above trips `reproducible: False`, which short-circuits the condition before
    the route is ever consulted — so renaming the route field left this branch dead and the
    suite still green. A reading can be reproducible and still have been answered off the
    route the harness asked for, and that one is the dearest to repeat blindly: it looks
    settled.
    """
    with _environment(["agents_md"]) as (root, files):
        with patch.object(cp, "measure_instruction_cost", side_effect=lambda path, **kw: {
                **_cost(path), "reproducible": True,
                "measured_under": {"cli_version": "0.150.0", "model": "m-a",
                                   "arms_on_own_route": False}}):
            _command(root, "measure-instructions")
        with patch.object(cp, "measure_instruction_cost",
                          side_effect=AssertionError("diagnose before repeating")):
            resumed = _command(root, "measure-instructions", "--resume")
        assert all("unstable" in s["reason"] for s in resumed["skipped"]), resumed["skipped"]

        # And the sound case still resumes, or the check above would be refusing everything.
        store_dir = store.store_dir()
        for entry in store_dir.glob("*.json"):
            entry.unlink()
        with patch.object(cp, "measure_instruction_cost", side_effect=lambda path, **kw: {
                **_cost(path), "measured_under": {"cli_version": "0.150.0", "model": "m-a",
                                                  "arms_on_own_route": True}}):
            _command(root, "measure-instructions")
        with patch.object(cp, "measure_instruction_cost", side_effect=_cost) as probe:
            again = _command(root, "measure-instructions", "--resume")
            assert probe.call_count == 0, "an on-route reading must not be paid for twice"
        assert all("already_measured" in s["reason"] for s in again["skipped"]), again["skipped"]


def test_checkpoint_failure_stops_spending_and_keeps_result_in_command_output():
    with _environment(["claude_md", "claude_md"]) as (root, files):
        blocked = root / "blocked"
        blocked.write_text("not a directory")
        with patch.dict(os.environ, {"AGENT_PROFILER_STORE": str(blocked / "store")}), \
                patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost) as probe:
            result = _command(root, "measure-instructions")
        assert probe.call_count == 1
        assert result["checkpoint_errors"]
        assert result["instruction_measurements"][0]["mean_tokens"] == 180
        assert result["skipped"][0]["reason"].startswith("checkpoint_failed")


def test_content_edited_during_measurement_is_not_marked_safe_to_resume():
    with _environment(["claude_md"]) as (root, files):
        def changed(path, **kwargs):
            path.write_text("changed during probe")
            return _cost(path)

        with patch.object(cp, "measure_claude_instruction_cost", side_effect=changed):
            result = _command(root, "measure-instructions")
        row = result["instruction_measurements"][0]
        assert row.get("input_changed_during_measurement") and "input_digest" not in row
        with patch.object(cp, "measure_claude_instruction_cost", side_effect=_cost) as probe:
            _command(root, "measure-instructions", "--resume")
            assert probe.call_count == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} measurement-batch tests passed (0 tokens spent)")

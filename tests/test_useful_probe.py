"""Regression suite for the `useful` link.

No test here spends a token: `_probe` is replaced with a stub. What is tested is the set of
controls that keep a difference between two arms from becoming a finding it has not earned —
above all the ones guarding against *flattering* errors, because a broken `without` arm makes
every component look indispensable and nothing in the output would look wrong.
"""

import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import useful_probe as up
from agent_profiler.useful_probe import (
    RULE_KINDS,
    measure_instruction_usefulness,
    scorable_checks,
)
from agent_profiler import i18n

# Pinned so a developer's own locale cannot change what these assert; the zh strings
# are the ones this suite was written against. English rendering is covered separately.
i18n.set_lang("zh")

# States every rule the check library can verify, so all four are scorable.
FULL_RULES = """# Commit conventions

- Put the issue reference at the end of the subject line.
- The subject stays within 50 characters.
- Write the body as one point per line, each line within 40 characters.
"""

# The same four rules, worded in another language with another reference spelling. Extraction
# must not depend on either, which is the property that a literal-marker library lacked.
FULL_RULES_OTHER_WORDING = """# 提交约定

- 标题末尾带议题编号。
- 标题 50 字以内。
- 正文分点写，每行 40 字以内。
"""

# States only the task-id rule. This is the shape of the real `~/AGENTS.md`, and the file
# that proved the checks respond to specific rules rather than to instructions existing.
# States only the trailing-reference rule, which is the shape of a real file that proved
# the checks respond to specific rules rather than to instructions existing at all.
REFERENCE_ONLY = """# Commit conventions

- The commit message ends with the issue reference, e.g. `feat(api): add limit (#1474)`.
"""

WITH_EXAMPLE = FULL_RULES + """
```
fix(auth): roles misread for one account #1551
```
"""

COMPLIANT = "fix: drop the off-by-one in export #2033\n\n- the page loop ended early\n- the last row is now included"
SLOPPY = "Fix off-by-one in the timesheet export page loop so the final row is included"


def _instruction(tmp: Path, content: str, name: str = "CLAUDE.md") -> Path:
    path = tmp / name
    path.write_text(content, encoding="utf-8")
    return path


@contextmanager
def _probe_returning(*, with_text=COMPLIANT, without_text=SLOPPY, control_without="391",
                     control_with="391", seen=None, probe=None):
    """Swap in a probe whose answer depends only on which arm it was called for."""
    def stub(prompt, workdir, instruction, timeout):
        armed = instruction is not None
        if seen is not None:
            seen.append({"arm": "with" if armed else "without", "prompt": prompt})
        if "17 times 23" in prompt:
            text = control_with if armed else control_without
        else:
            text = with_text if armed else without_text
        return {"ok": True, "text": text, "ctx": 20_000, "cost_usd": 0.01}

    original, backoff = up._probe, up.RETRY_BACKOFF_SECONDS
    up._probe = probe or stub
    # The budget stays real, so a retry path is exercised exactly as it ships. Only the wait
    # between attempts goes, because no test should pay for wall clock: with the literal in
    # place, four failure-path tests alone sat through 135 seconds of sleep.
    up.RETRY_BACKOFF_SECONDS = 0
    try:
        yield
    finally:
        up._probe, up.RETRY_BACKOFF_SECONDS = original, backoff


@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as raw:
        yield Path(raw)


def test_clean_separation_on_both_tasks_proves_useful():
    with _probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    assert out["useful"] == "proven", out
    assert out["negative_control"]["held"] is True
    # The claim must stay inside what the method supports: the file defines the standard.
    assert "worth following" in out["claim"]
    assert set(out["rules_proven"]) == {k["name"] for k in RULE_KINDS}


def test_a_broken_without_arm_is_refused_not_celebrated():
    """The flattering error: if the `without` arm cannot do arithmetic either, it is dead."""
    with _probe_returning(control_without="I cannot help with that"), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    assert out["useful"] == "unproven"
    assert out["blocker"] == "CONTROL_FAILED"
    # The separation on the real checks was perfect; it still must not be reported.
    assert "artifact" in out["reason"]


def test_reciting_the_files_example_voids_the_run():
    recited = "fix(auth): roles misread for one account #1551\n\n- the caller was dropped"
    with _probe_returning(with_text=recited), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, WITH_EXAMPLE), repeats=2)
    assert out["useful"] == "unproven"
    assert out["blocker"] == "RECITED_EXAMPLE"
    assert out["recited_example_runs"] > 0


def test_identical_arms_report_no_separation():
    with _probe_returning(without_text=COMPLIANT), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    assert out["useful"] == "unproven"
    assert out["blocker"] == "NO_SEPARATION"


def test_a_file_stating_no_verifiable_rule_is_not_probed():
    """No quotable rule means no scorable claim, so nothing should be spent."""
    calls = []
    with _probe_returning(seen=calls), _tmp() as tmp:
        out = measure_instruction_usefulness(
            _instruction(tmp, "# Notes\n\nBe careful and write good code.\n"), repeats=2)
    assert out["blocker"] == "NO_STATED_RULES"
    assert calls == []


def test_an_empty_file_is_refused():
    with _probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, "   \n"), repeats=2)
    assert out["ok"] is False
    assert "empty" in out["reason"]


def test_only_rules_the_file_states_are_scorable():
    """The lesson that cost a retraction: a check with no quotable rule cannot be scored."""
    stated, not_stated = scorable_checks(REFERENCE_ONLY)
    assert [c["name"] for c in stated] == ["subject_ends_with_reference"]
    assert "body_line_length_limit" in [c["name"] for c in not_stated]

    stated_full, not_stated_full = scorable_checks(FULL_RULES)
    assert len(stated_full) == len(RULE_KINDS)
    assert not_stated_full == []


def test_a_file_stating_one_rule_is_scored_only_on_that_rule():
    with _probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(
            _instruction(tmp, REFERENCE_ONLY, name="AGENTS.md"), repeats=2)
    assert out["rules_proven"] == ["subject_ends_with_reference"]
    assert "body_line_length_limit" in out["rules_not_stated"]
    # Unmeasured content means the verdict is a floor, and the payload has to say so.
    assert any("floor" in c for c in out["caveats"])


def test_arms_alternate_within_each_task():
    """Three failures in this project came from comparing unpaired measurements."""
    calls = []
    with _probe_returning(seen=calls), _tmp() as tmp:
        measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    arms = [c["arm"] for c in calls]
    pairs = [tuple(arms[i:i + 2]) for i in range(0, len(arms), 2)]
    assert all(set(p) == {"with", "without"} for p in pairs), pairs
    # Consecutive repeats lead with opposite arms, so monotonic drift cancels.
    assert pairs[0] != pairs[1]


def test_auth_failures_are_invalid_runs_not_task_failures():
    """A harness that never started must not look like a component that was needed."""
    def probe(prompt, workdir, instruction, timeout):
        if instruction is None:
            return {"ok": False, "error": "auth/init failure (ctx=2)", "cost_usd": 0.0001}
        return {"ok": True, "text": COMPLIANT, "ctx": 20_000, "cost_usd": 0.01}

    with _probe_returning(probe=probe), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    assert out["probes_failed"] > 0
    # Zero valid `without` runs cannot support a separation, so the control gate catches it.
    assert out["useful"] == "unproven"
    assert out["blocker"] == "CONTROL_FAILED"


def test_the_context_floor_accepts_bare_mode():
    """A real `--bare` run sits well under a thousand tokens, and the old guard treated
    anything below that as a failed probe. The figure is a property of the harness, not of
    anyone's configuration — bare mode drops the user's settings entirely."""
    assert up.MIN_VALID_CTX < 864


def test_probes_close_stdin():
    """`claude -p` appends piped stdin to its prompt; inheriting it cost many times a
    controlled probe once.

    Asserted at the call site, like the equivalent guard in test_evaluate.py: the
    alternative is spawning a real subprocess to check it.
    """
    import inspect as _inspect

    source = _inspect.getsource(up._probe)
    assert "subprocess.run" in source
    assert "stdin=subprocess.DEVNULL" in source, (
        "useful_probe._probe spawns the CLI without closing stdin; its prompt would "
        "depend on the caller's unconsumed input")


def test_instruction_and_skill_probes_cannot_inherit_mcp_servers():
    from unittest.mock import patch
    from types import SimpleNamespace

    calls = []

    def launch(argv, **kwargs):
        calls.append(argv)
        # Fail before parsing an envelope; only the actual process boundary matters.
        return SimpleNamespace(returncode=1, stderr="fixture stop")

    with _tmp() as tmp, patch.object(up.subprocess, "run", side_effect=launch):
        up._probe("fixture", tmp, None, 1)
        up._probe("fixture", tmp, None, 1, bare=False,
                  extra_args=("--setting-sources", "", "--plugin-dir", str(tmp)))
    assert len(calls) == 2
    for argv in calls:
        assert "--strict-mcp-config" in argv
        assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--bare" not in calls[1] and "--plugin-dir" in calls[1]


def test_no_result_overstates_what_was_measured():
    """External validity is weak and the payload must not let a reader forget it."""
    with _probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    blob = json.dumps(out, ensure_ascii=False)
    for banned in ("saves ", "wasted", "proves the file is worth"):
        assert banned not in blob, banned
    assert any("not effect across the user's real sessions" in c for c in out["caveats"])


# -- the Codex arm -----------------------------------------------------------------------


@contextmanager
def _codex_probe_returning(*, with_text=COMPLIANT, without_text=SLOPPY,
                           control="391", seen=None):
    """Swap in a Codex probe. Codex reports tokens and no dollar figure."""
    def stub(prompt, home, workdir, timeout):
        armed = (workdir / "AGENTS.md").is_file()
        if seen is not None:
            seen.append({"arm": "with" if armed else "without", "prompt": prompt})
        if "17 times 23" in prompt:
            text = control
        else:
            text = with_text if armed else without_text
        return {"ok": True, "text": text, "ctx": 13_700, "cost_usd": None}

    original, backoff = up._codex_probe, up.RETRY_BACKOFF_SECONDS
    up._codex_probe = stub
    up.RETRY_BACKOFF_SECONDS = 0
    try:
        yield
    finally:
        up._codex_probe, up.RETRY_BACKOFF_SECONDS = original, backoff


def test_each_instruction_file_is_measured_in_the_harness_that_loads_it():
    """A Claude result attributed to a Codex file would be the wrong component."""
    assert up.INSTRUCTION_HARNESS["claude_md"] == "claude_code"
    assert up.INSTRUCTION_HARNESS["agents_md"] == "codex"
    assert set(up.INSTRUCTION_HARNESS.values()) <= set(up.HARNESS_ARMS)


def test_an_unknown_harness_is_rejected():
    with _tmp() as tmp:
        try:
            measure_instruction_usefulness(_instruction(tmp, REFERENCE_ONLY), harness="nope")
        except ValueError as exc:
            assert "unknown harness" in str(exc)
        else:
            raise AssertionError("an unknown harness must not silently pick a default")


def test_the_codex_arm_proves_usefulness_without_a_channel_caveat():
    """Codex finds the injected copy the same way it finds the real file: walk-up."""
    with _codex_probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(
            _instruction(tmp, REFERENCE_ONLY, name="AGENTS.md"), repeats=2, harness="codex")
    assert out["useful"] == "proven"
    assert out["harness"] == "codex"
    assert out["rules_proven"] == ["subject_ends_with_reference"]
    # The Claude arm needs a caveat about the delivery channel; this one does not, and the
    # payload must not carry a caveat that does not apply to it.
    assert not any("appended system prompt" in c for c in out["caveats"])
    # The caveats that do still apply are present.
    assert any("not effect across the user's real sessions" in c for c in out["caveats"])
    assert any("floor" in c for c in out["caveats"])


def test_a_contaminated_codex_baseline_is_detected():
    """The walk-up hazard: an ancestor AGENTS.md puts instructions in *both* arms.

    Reachable in practice when the temp root sits inside HOME — a `TMPDIR` under the home
    directory is enough, and then the walk-up reaches the user's real `~/AGENTS.md`.
    Running anyway would yield a near-zero difference, which reads as "this file does
    nothing": a false negative indistinguishable from a real result.
    """
    with _tmp() as root:
        (root / "AGENTS.md").write_text(
            "Put the issue reference at the end of the subject.", encoding="utf-8")
        nested = root / "nested"
        nested.mkdir()
        built = up._codex_arms(nested, REFERENCE_ONLY, "AGENTS.md")
    assert "error" in built
    assert "above it" in built["error"]
    assert "run" not in built


def test_a_refusing_arm_builder_stops_the_measurement_before_spending():
    """Whatever the arms refuse for, no probe may run and no verdict may be reported."""
    calls = []

    def refusing(tmp, content, filename):
        return {"error": "the baseline arm has an instruction file above it (/x/AGENTS.md)"}

    original = dict(up.HARNESS_ARMS)
    up.HARNESS_ARMS["codex"] = refusing
    try:
        with _codex_probe_returning(seen=calls), _tmp() as tmp:
            out = measure_instruction_usefulness(
                _instruction(tmp, REFERENCE_ONLY, name="AGENTS.md"), repeats=2,
                harness="codex")
    finally:
        up.HARNESS_ARMS.clear()
        up.HARNESS_ARMS.update(original)
    assert out["useful"] == "unproven"
    assert out["blocker"] == "CONTAMINATED_BASELINE"
    assert calls == []


def test_an_empty_ancestor_file_does_not_count_as_contamination():
    """`~/.codex/AGENTS.md` is 0 bytes here; an empty file instructs nothing."""
    with _tmp() as root:
        (root / "AGENTS.md").write_text("   \n", encoding="utf-8")
        nested = root / "nested"
        nested.mkdir()
        assert up._ancestor_instruction(nested) is None


def test_codex_results_do_not_claim_a_dollar_cost():
    """`0.0` would read as "this was free"; the absence is reported as an absence."""
    with _codex_probe_returning(), _tmp() as tmp:
        out = measure_instruction_usefulness(
            _instruction(tmp, REFERENCE_ONLY, name="AGENTS.md"), repeats=2, harness="codex")
    assert out["cost_usd"] is None
    assert "no dollar figure" in out["cost_note"]
    assert out["probes_attempted"] > 0


def test_the_codex_arm_alternates_too():
    calls = []
    with _codex_probe_returning(seen=calls), _tmp() as tmp:
        measure_instruction_usefulness(
            _instruction(tmp, REFERENCE_ONLY, name="AGENTS.md"), repeats=2, harness="codex")
    arms = [c["arm"] for c in calls]
    pairs = [tuple(arms[i:i + 2]) for i in range(0, len(arms), 2)]
    assert all(set(p) == {"with", "without"} for p in pairs), pairs
    assert pairs[0] != pairs[1]


def test_the_codex_probe_closes_stdin():
    """`codex exec` appends piped stdin to its prompt, exactly like `claude -p`."""
    import inspect as _inspect

    source = _inspect.getsource(up._codex_probe)
    assert "subprocess.run" in source
    assert "stdin=subprocess.DEVNULL" in source


def test_a_runaway_codex_run_is_rejected_not_recorded():
    """A run far above the fixed prefix did work instead of answering the task."""
    assert up.PROBE_CONTEXT_CEILING > 13_700
    source = Path(up.__file__).read_text(encoding="utf-8")
    assert "PROBE_CONTEXT_CEILING" in source.split("def _codex_probe", 1)[1].split("def ", 1)[0]


# -- the skill arm ------------------------------------------------------------------------
#
# Same checks and same controls as an instruction file; only the arm differs. These tests
# cover the three things that are specific to it and easy to get silently wrong: the flags,
# bare mode, and leaving the real directory alone.


@contextmanager
def _skill_probe(*, with_text=COMPLIANT, without_text=SLOPPY, control="391", seen=None,
                 listed="demo-skill", probe=None):
    """A stub for the skill arm, which tells its arms apart by flags rather than by an
    appended file: both arms pass `instruction=None`."""
    def stub(prompt, workdir, instruction, timeout, *, bare=True, extra_args=()):
        armed = "--plugin-dir" in extra_args
        # The presence check runs first and asks the harness to list its skills. Answering
        # it here keeps these tests about the arms; `listed` is what that reply contains.
        if "List the names of the skills" in prompt:
            return {"ok": True, "text": listed, "ctx": 20_000, "cost_usd": 0.01}
        if seen is not None:
            seen.append({"arm": "with" if armed else "without", "bare": bare,
                         "args": tuple(extra_args)})
        if "17 times 23" in prompt:
            return {"ok": True, "text": control, "ctx": 20_000, "cost_usd": 0.01}
        return {"ok": True, "text": with_text if armed else without_text,
                "ctx": 20_000, "cost_usd": 0.01}

    original, budget = up._probe, up.RETRY_BUDGET
    backoff = up.RETRY_BACKOFF_SECONDS
    up._probe = probe or stub
    # The retry budget stays real so a failure path is exercised as it ships; only the wait
    # between attempts is removed, because no test should sit through wall clock.
    up.RETRY_BACKOFF_SECONDS = 0
    try:
        yield
    finally:
        up._probe, up.RETRY_BUDGET = original, budget
        up.RETRY_BACKOFF_SECONDS = backoff


def _skill(tmp: Path, content: str, name: str = "demo-skill") -> Path:
    directory = tmp / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(content, encoding="utf-8")
    return directory


def test_a_skill_stating_a_rule_is_measured_like_an_instruction_file():
    with _skill_probe(), _tmp() as tmp:
        out = up.measure_skill_usefulness(_skill(tmp, FULL_RULES), repeats=2)
    assert out["useful"] == "proven", out
    assert out["negative_control"]["held"] is True
    assert set(out["rules_proven"]) == {k["name"] for k in RULE_KINDS}


def test_the_skill_arm_never_runs_bare():
    """Bare mode skips plugin sync, so `--plugin-dir` would load nothing and the two arms
    would be identical while appearing not to be — a false "does nothing"."""
    calls = []
    with _skill_probe(seen=calls), _tmp() as tmp:
        up.measure_skill_usefulness(_skill(tmp, FULL_RULES), repeats=1)
    assert calls and all(call["bare"] is False for call in calls)


def test_both_skill_arms_clear_the_users_own_instruction_files():
    """The confound the instruction route has to argue about is removed by the route here:
    a rule the user's own files state as well is absent from both arms."""
    calls = []
    with _skill_probe(seen=calls), _tmp() as tmp:
        up.measure_skill_usefulness(_skill(tmp, FULL_RULES), repeats=1)
    for call in calls:
        assert "--setting-sources" in call["args"]
        assert call["args"][call["args"].index("--setting-sources") + 1] == ""
    assert any("--plugin-dir" in call["args"] for call in calls)
    assert any("--plugin-dir" not in call["args"] for call in calls)


def test_the_real_skill_directory_is_untouched_by_its_own_measurement():
    """The directory is the object under measurement. A probe that edited it would be
    measuring its own edit, and the configuration under test is read-only."""
    with _skill_probe(), _tmp() as tmp:
        directory = _skill(tmp, FULL_RULES)
        (directory / "references").mkdir()
        (directory / "references" / "extra.md").write_text("detail\n", encoding="utf-8")
        before = {str(p.relative_to(directory)): p.read_bytes()
                  for p in sorted(directory.rglob("*")) if p.is_file()}
        up.measure_skill_usefulness(directory, repeats=1)
        after = {str(p.relative_to(directory)): p.read_bytes()
                 for p in sorted(directory.rglob("*")) if p.is_file()}
    assert before == after


def test_a_skill_stating_no_rule_is_not_probed():
    calls = []
    with _skill_probe(seen=calls), _tmp() as tmp:
        out = up.measure_skill_usefulness(
            _skill(tmp, "# Helper\n\nUse good judgement.\n"), repeats=2)
    assert out["blocker"] == "NO_STATED_RULES"
    assert calls == []


def test_a_skills_own_fixture_skills_are_not_treated_as_installs():
    """A skill that ships fixture skills puts them under `evals/fixtures/skills/<name>/`,
    which passes the "child of a skills/ directory" test exactly. They are data belonging to
    the outer skill, and probing them would invent installs that nothing loads."""
    with _tmp() as tmp:
        root = tmp / ".claude" / "skills"
        real = root / "outer"
        (real / "evals" / "fixtures" / "skills" / "inner").mkdir(parents=True)
        (real / "SKILL.md").write_text(FULL_RULES, encoding="utf-8")
        (real / "evals" / "fixtures" / "skills" / "inner" / "SKILL.md").write_text(
            FULL_RULES, encoding="utf-8")
        scan = {"files": [
            {"kind": "skills", "origin": "declared", "harness": "claude_code",
             "path": str(real / "SKILL.md")},
            {"kind": "skills", "origin": "declared", "harness": "claude_code",
             "path": str(real / "evals" / "fixtures" / "skills" / "inner" / "SKILL.md")},
        ]}
        assert up._declared_skill_dirs(scan) == {real}


def test_a_codex_side_skill_has_no_arm_and_is_left_alone():
    """`--plugin-dir` is a Claude Code flag. Measuring a Codex skill through it would
    attribute the result to a route that does not load it there."""
    with _tmp() as tmp:
        path = tmp / ".codex" / "skills" / "k" / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(FULL_RULES, encoding="utf-8")
        scan = {"files": [{"kind": "skills", "origin": "declared", "harness": "codex",
                           "path": str(path)}]}
        assert up._declared_skill_dirs(scan) == set()


def test_every_probe_failing_is_not_reported_as_a_failed_control():
    """From a real run: 72 of 90 probes died and four components reported CONTROL_FAILED,
    which reads as a finding about those components. Nothing was measured at all."""
    def dead(prompt, workdir, instruction, timeout, *, bare=True, extra_args=()):
        return {"ok": False, "error": "exit 1: not logged in", "cost_usd": 0.0}

    with _probe_returning(probe=dead), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=2)
    assert out["blocker"] == "NO_VALID_RUNS"
    # The harness's own words have to survive into the artifact, or the run can only be
    # diagnosed by reproducing it by hand.
    assert "not logged in" in out["reason"]


def test_a_failing_probe_keeps_the_harnesss_own_error_text():
    """`exit 1` alone was indistinguishable from the known auth/init flake."""
    import subprocess

    original = up.subprocess.run

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Error: bad flag\n")

    up.subprocess.run = fake_run
    try:
        with _tmp() as tmp:
            work = tmp / "w"
            work.mkdir()
            result = up._probe("hi", work, None, 5)
    finally:
        up.subprocess.run = original
    assert result["ok"] is False
    assert "bad flag" in result["error"]


def test_harness_health_marks_an_arm_that_answered_nothing():
    """One harness dead and one healthy, in a single run: the per-component verdicts cannot
    carry this, because each one looks like an ordinary unproven."""
    health = up._harness_health([
        {"harness": "codex", "records": [{"valid": True}, {"valid": True}]},
        {"harness": "claude_code", "records": [{"valid": False, "error": "exit 1"}] * 3},
    ])
    assert health["codex"]["usable"] is True
    assert health["claude_code"]["usable"] is False
    assert health["claude_code"]["errors"] == {"exit 1": 3}


def test_the_rule_library_holds_no_authors_conventions():
    """The error this replaced: the library was one person's commit rules copied in as
    literal strings. It then only matched files worded their way, and the components it
    selected for measurement were the ones restating those same conventions — so every
    measurement stayed inside one setup. A tool holds mechanism; specifics come from
    whatever it is pointed at."""
    import inspect as _inspect

    source = _inspect.getsource(up)
    library = source[source.index("RULE_KINDS = ("):source.index("def _stated_limit")]
    # No limit may be compared against a constant. Regex quantifiers like `{0,40}` and
    # `{2,3}` contain digits too, so the test is about *comparisons*, not about digits
    # appearing anywhere — checking for digits flagged the patterns themselves.
    assert not re.search(r"(?:<=|>=|<|>|==)\s*\d", library), library
    # Nor may a reference spelling be a literal marker. The pattern describes shapes.
    assert "taskId" not in source

    # And the mechanism has to work on wordings the author never saw.
    for text in ("- Subject must end with the ticket id, max 72 characters.\n",
                 "- 标题末尾带编号，72 字以内。\n"):
        stated, _ = scorable_checks(text)
        names = {c["name"] for c in stated}
        assert "subject_ends_with_reference" in names, text
        limit = next(c["parameter"] for c in stated
                     if c["name"] == "subject_length_limit")
        assert limit == 72, text


def test_a_limit_is_read_from_the_file_not_assumed():
    """Two files stating different limits must be held to their own."""
    strict, _ = scorable_checks("- The subject stays within 50 characters.\n")
    loose, _ = scorable_checks("- The subject stays within 72 characters.\n")
    strict_check = next(c for c in strict if c["name"] == "subject_length_limit")
    loose_check = next(c for c in loose if c["name"] == "subject_length_limit")
    subject = "x" * 60
    assert strict_check["predicate"](subject, []) is False
    assert loose_check["predicate"](subject, []) is True


def test_a_bare_number_is_not_read_as_a_limit():
    """A step number or a year beside the word "title" would otherwise become a length
    rule, and the probe would hold a file to a limit it never stated."""
    stated, not_stated = scorable_checks(
        "3. In 2026 the title convention changed.\n")
    assert "subject_length_limit" in {c["name"] for c in not_stated}


def test_a_skill_the_arm_did_not_load_is_refused_before_spending():
    """From a real run: three copies came back `no separation`, which reads as "this skill
    changes nothing". A skill loads on a description match, and the probe's tasks had
    nothing to do with this one's description, so most likely it never entered context —
    "it did nothing", "it never loaded" and "the task did not need it" are not
    distinguishable in that number."""
    calls = []
    with _skill_probe(seen=calls, listed="something-else"), _tmp() as tmp:
        out = up.measure_skill_usefulness(_skill(tmp, FULL_RULES), repeats=2)
    assert out["blocker"] == "COMPONENT_NOT_LOADED_BY_ARM"
    # Refused *before* the arms run, so nothing is spent on two identical arms.
    assert calls == []


def test_a_presence_check_that_could_not_run_does_not_block_the_measurement():
    """A failure to check is not a failure to load. Refusing on the strength of a probe
    that never answered would turn one flake into a verdict."""
    def flaky(prompt, workdir, instruction, timeout, *, bare=True, extra_args=()):
        if "List the names of the skills" in prompt:
            return {"ok": False, "error": "exit 1 (no stderr)", "cost_usd": 0.0}
        if "17 times 23" in prompt:
            return {"ok": True, "text": "391", "ctx": 20_000, "cost_usd": 0.01}
        armed = "--plugin-dir" in extra_args
        return {"ok": True, "text": COMPLIANT if armed else SLOPPY,
                "ctx": 20_000, "cost_usd": 0.01}

    # Installed through the helper, not assigned inside the `with` block: assigning there
    # leaves the real probe in place when the block exits, and every later test in the file
    # then calls the harness for real. That cost a test run that spent money.
    with _skill_probe(probe=flaky), _tmp() as tmp:
        out = up.measure_skill_usefulness(_skill(tmp, FULL_RULES), repeats=2)
    assert out.get("useful") == "proven", out
    assert out["skill_presence"]["checked"] is False


def test_a_batch_is_abandoned_once_the_harness_stops_answering():
    """54 probes were spent against a condition that was not recovering, and the retry
    budget cannot help: it is sized for an independent failure rate, so every attempt
    against a persistent condition fails."""
    def dead(prompt, workdir, instruction, timeout):
        return {"ok": False, "error": "exit 1 (no stderr)", "cost_usd": 0.0}

    with _probe_returning(probe=dead), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=3)
    assert out["blocker"] == "NO_VALID_RUNS"
    assert out["abandoned"]["consecutive_failures"] == up.CONSECUTIVE_FAILURE_LIMIT
    # The whole point: stopped early rather than running all 18.
    assert out["probes_attempted"] == up.CONSECUTIVE_FAILURE_LIMIT
    assert "abandoned" in out["reason"]


def test_scattered_failures_do_not_abandon_a_healthy_batch():
    """The counter has to reset on success, or ordinary intermittent failures spread across
    tasks would accumulate into a false exhaustion and cut a good run short."""
    state = {"n": 0}

    def every_other(prompt, workdir, instruction, timeout):
        state["n"] += 1
        if state["n"] % 2 == 0:
            return {"ok": False, "error": "auth/init failure (ctx=2)", "cost_usd": 0.0}
        return {"ok": True, "text": COMPLIANT if instruction is not None else SLOPPY,
                "ctx": 20_000, "cost_usd": 0.01}

    with _probe_returning(probe=every_other), _tmp() as tmp:
        out = measure_instruction_usefulness(_instruction(tmp, FULL_RULES), repeats=3)
    assert out["abandoned"] is None
    assert out["probes_attempted"] == 18


def test_a_directory_without_a_skill_md_is_refused():
    with _skill_probe(), _tmp() as tmp:
        directory = tmp / "empty-skill"
        directory.mkdir()
        out = up.measure_skill_usefulness(directory, repeats=1)
    assert out["ok"] is False and "SKILL.md" in out["reason"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    pristine = up._probe
    for t in tests:
        t()
        # No test may leave a live probe installed. One did, by assigning `up._probe`
        # inside a `with` block that restored the real one on exit, and every later test
        # then called the harness for real and spent money.
        assert up._probe is pristine, f"{t.__name__} left a probe installed"
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} useful-probe regression tests passed")


if __name__ == "__main__":
    main()

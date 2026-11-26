"""Regression suite for prefix differencing, and mostly for the skill arm.

No test here spends a token: `_probe` is replaced with a stub.

What is tested is the thing that nearly stayed un-tested forever. "No CLI flag disables one
skill, so there is no single-component arm" was written into this project five times and
believed, and it was false — the arm is built by addition, like every other arm here. The
lesson is not that the blocker was wrong. It is that a *treatment arm which silently loads
nothing* reports a real component as free, and nothing about that output looks broken:
`--plugin-dir` under `--bare` is accepted, loads no skill, and would have returned "this
skill costs 0 tokens" with a clean spread of 0.

So the guard that matters is the one where each arm lists its own skills, and most of this
file is about pairs being *rejected*.
"""

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import codex_probe as cp
from agent_profiler import useful_probe as up
from agent_profiler import i18n

# Pinned so a developer's own locale cannot change what these assert; the zh strings
# are the ones this suite was written against. English rendering is covered separately.
i18n.set_lang("zh")

BUILT_INS = ["dataviz", "code-review", "simplify", "loop", "run", "init"]


def _skill(tmp: Path, name: str = "demo-skill") -> Path:
    directory = tmp / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Interrogate the user with hard questions.\n---\n"
        "# Grill me\n", encoding="utf-8")
    (directory / "references").mkdir()
    (directory / "references" / "notes.md").write_text("extra file\n", encoding="utf-8")
    return directory


@contextmanager
def _probe_returning(*, with_extra="arm:demo-skill", with_ctx=17_259, without_ctx=17_096,
                     without_names=BUILT_INS, seen=None):
    """A stub keyed on whether the arm carried `--plugin-dir`, which is the real arm's mark."""
    def stub(prompt, workdir, instruction, timeout, *, bare=True, extra_args=()):
        armed = "--plugin-dir" in extra_args
        if seen is not None:
            seen.append({"arm": "with" if armed else "without", "bare": bare,
                         "extra_args": list(extra_args)})
        names = list(without_names)
        if armed and with_extra:
            names = [with_extra] + names
        return {"ok": True, "text": "\n".join(names),
                "ctx": with_ctx if armed else without_ctx, "cost_usd": 0.05}

    original = up._probe
    up._probe = stub
    try:
        yield
    finally:
        up._probe = original


@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as raw:
        yield Path(raw)


def test_a_verified_pair_reports_the_marginal_token_cost_of_one_skill():
    """The measurement this project was told could not exist."""
    with _probe_returning(), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=2)

    assert result["ok"] is True
    assert result["skill"] == "demo-skill"
    assert result["mean_tokens"] == 163
    assert result["pair_spread_tokens"] == 0 and result["reproducible"] is True
    assert result["differences"] == [163, 163]
    # A figure measured under one harness must say which, or it gets added to the wrong floor.
    assert result["harness"] == "claude_code"
    # Both caveats travel with the number: the plugin channel, and the 11-skill baseline.
    assert "plugin" in result["channel_caveat"] and "marginal" in result["channel_caveat"]


def test_arms_alternate_within_the_window_and_the_baseline_drops_user_skills():
    """Interleaving is the project's central lesson; the flags are what make the arms differ."""
    seen = []
    with _probe_returning(seen=seen), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=2)

    assert [p["order"] for p in result["pairs"]] == [["without", "with"], ["with", "without"]]
    # Never bare: bare mode skips plugin sync, so the treatment arm would load nothing.
    assert all(run["bare"] is False for run in seen)
    assert all(run["extra_args"][:2] == ["--setting-sources", ""] for run in seen)
    assert [run["arm"] for run in seen] == ["without", "with", "with", "without"]


def test_a_treatment_arm_that_loaded_nothing_is_rejected_not_averaged():
    """The failure that would have published "this skill is free".

    `--plugin-dir` under `--bare` is accepted and loads no skill. With identical listings and
    a believable prefix, only the arm's own listing distinguishes that from a free skill.
    """
    with _probe_returning(with_extra=None, with_ctx=17_096), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=1)

    assert result["ok"] is False and "no pair completed" in result["reason"]
    assert "missing from the treatment arm" in result["pairs"][0]["error"]
    assert "difference" not in result["pairs"][0]
    assert "mean_tokens" not in result


def test_a_baseline_that_lists_nothing_makes_the_absence_check_vacuous():
    """An empty listing proves nothing about absence, so it cannot stand in for it."""
    with _probe_returning(without_names=[]), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=1)

    assert result["ok"] is False
    assert "vacuous" in result["pairs"][0]["error"]


def test_a_baseline_that_already_has_the_skill_is_rejected():
    """If both arms carry it, the difference is not about this skill."""
    with _probe_returning(without_names=["demo-skill"] + BUILT_INS), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=1)

    assert result["ok"] is False
    assert "already in the baseline" in result["pairs"][0]["error"]


def test_an_arm_that_loaded_a_fraction_of_the_prefix_is_rejected():
    """The same relative guard the CLAUDE.md measurement needed, for the same reason: a
    partially-initialized arm passes any absolute floor and inverts the result."""
    with _probe_returning(with_ctx=1_340, without_ctx=23_632), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=1)

    assert result["ok"] is False
    assert "arms disagree" in result["pairs"][0]["error"]


def test_a_difference_inside_the_noise_band_is_inconclusive_rather_than_a_finding():
    with _probe_returning(with_ctx=17_110, without_ctx=17_096), _tmp() as tmp:
        result = cp.measure_skill_cost(_skill(tmp), pairs=1)

    assert result["ok"] is False and result["inconclusive"] is True
    assert str(cp.MIN_DETECTABLE_TOKENS) in result["reason"]


def test_the_real_skill_is_copied_and_never_modified():
    """The environment is the object under measurement — the standing rule for this project."""
    with _tmp() as tmp:
        directory = _skill(tmp)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in directory.rglob("*") if p.is_file()}
        with _probe_returning():
            cp.measure_skill_cost(directory, pairs=1)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in directory.rglob("*") if p.is_file()}
        assert after == before
        # The plugin is built in a temp directory, so nothing is left beside the skill.
        assert sorted(p.name for p in directory.parent.iterdir()) == ["demo-skill"]


def test_the_arm_plugin_carries_exactly_one_skill_with_its_whole_directory():
    """A skill is a directory, not just its SKILL.md; injecting only the entry point would
    measure something the installed skill is not."""
    with _tmp() as tmp:
        plugin = cp._plugin_with_one_skill(tmp, _skill(tmp))
        manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["name"] == cp.ARM_PLUGIN_NAME
        assert [p.name for p in (plugin / "skills").iterdir()] == ["demo-skill"]
        assert (plugin / "skills" / "demo-skill" / "references" / "notes.md").is_file()


def test_listed_skills_strips_the_plugin_prefix_the_channel_adds():
    """A plugin skill is listed as `plugin:name`, so a literal match would never find it."""
    names = cp._listed_skills("arm:demo-skill\ndataviz\n- code-review\n`simplify`\n")
    assert names == {"demo-skill", "dataviz", "code-review", "simplify"}
    assert cp._listed_skills("NONE") == {"NONE"}


def _inventory(*paths: str) -> dict:
    """A scan whose file list has the real shape: every file inside every skill."""
    files = []
    for path in paths:
        files.append({"kind": "skills", "origin": "declared", "path": path})
        files.append({"kind": "skills", "origin": "declared",
                      "path": str(Path(path).parent / "references" / "notes.md")})
    return {"files": files}


def test_only_skills_the_harness_would_list_are_measured():
    """Three filters, each of which a stubbed probe cannot see, and all three were added
    after a real run contradicted the inventory.

    The inventory offered 12 candidates under 6 names. The harness's own listing names 4.
    Every extra candidate would have been measured at a real cost and reported beside the real
    figures: a `references/` directory, two eval fixtures nested inside `nested-skill`, and
    four second copies.
    """
    with _tmp() as tmp:
        claude = tmp / ".claude" / "skills"
        agents = tmp / ".agents" / "skills"
        for root in (claude, agents):
            root.mkdir(parents=True)
            for name in ("demo-skill", "second-skill"):
                directory = root / name
                directory.mkdir()
                body = "same\n" if name == "demo-skill" else f"differs at {root}\n"
                (directory / "SKILL.md").write_text(body, encoding="utf-8")
                (directory / "references").mkdir()
                (directory / "references" / "notes.md").write_text("x\n", encoding="utf-8")
            # An eval fixture with the shape of a skill, nested inside a real one.
            fixture = root / "demo-skill" / "evals" / "fixtures" / "skills" / "sample"
            fixture.mkdir(parents=True)
            (fixture / "SKILL.md").write_text("fixture\n", encoding="utf-8")

        scan = _inventory(*[str(p) for p in tmp.rglob("SKILL.md")])
        scan["files"].append({"kind": "skills", "origin": "declared",
                              "path": str(tmp / ".codex" / "skills" / "eighth-skill" / "SKILL.md")})
        targets, dropped = cp._installed_skills(scan)

    assert sorted(p.name for p in targets) == ["demo-skill", "second-skill", "second-skill"]
    reasons = {d["skill"]: d["reason"] for d in dropped}
    assert "nested inside another skill" in reasons["sample"]
    assert "byte-identical" in reasons["demo-skill"]
    # Codex's own skills are not measurable here: `--plugin-dir` is a Claude Code flag.
    assert not any("eighth-skill" in str(p) for p in targets)


def test_a_name_installed_twice_with_different_content_is_marked_ambiguous():
    """Both copies are measurable, the harness lists the name once, and which one it loads
    is a blind spot this project already confirmed is closed. So neither figure may be
    quietly attributed to the listed skill."""
    paths = ["/root-a/skills/second-skill", "/root-b/skills/second-skill",
             "/root-a/skills/demo-skill"]
    ambiguous = cp._ambiguous_names([Path(p) for p in paths])

    assert set(ambiguous) == {"second-skill"}
    assert sorted(ambiguous["second-skill"]) == sorted(paths[:2])


REFUTATIONS = ("false", "beside the point", "disproved", "turned out", "was wrong",
               "it was wrong", "不相干", "误判")

# Every wording of the disproved blocker found in this repo. The list grew *after* the first
# version of this guard passed while three live sites still asserted the claim, and each
# miss taught something different about why a phrase blacklist is weak:
#
#   README.md                     not scanned at all — the guard looked only in src/ and docs/
#   feasibility-2026-11-19.md     scanned, but the phrase was line-wrapped, so no substring hit
#   useful_probe.py (notes)       scanned, but worded "toggled" where the guard said "disabled"
#
# The last was the worst of the three: it is an *emitted* note, so the shipped artifact
# `useful-instructions.json` was still telling its reader that no skill can be
# isolated, days after the measurement that disproved it. Hence: normalize whitespace, scan
# every text file in the repo, and keep the variants together where they can be extended.
BLOCKER_WORDINGS = (
    "no CLI flag disables one skill",
    "a skill cannot be disabled individually",
    "none of them can be toggled individually",
    "none of which can be toggled individually",
    "cannot be toggled individually",
    "没法单独关掉",
    "不能单独关掉",
)


def test_the_blocker_this_module_disproved_is_never_stated_without_its_refutation():
    """Five sites said no single-skill arm exists. A disproved blocker left standing is the
    tool telling its user it cannot answer something it just answered.

    The phrases themselves are not banned — this module's own docstring quotes one in order
    to refute it, and deleting that history is how a blocker gets re-adopted. What is banned
    is a paragraph stating one with nothing nearby marking it as dead.

    Generated artifacts under `docs/dogfood/` are excluded because they are outputs: a stale
    one is fixed by re-running. That exclusion is also why the emitted *note* matters more
    than the docstring — the note is what lands in the artifact.
    """
    root = Path(__file__).resolve().parent.parent
    sources = [p for p in root.rglob("*")
               if p.suffix in (".py", ".md") and p.is_file()
               and "dogfood" not in p.parts and ".git" not in p.parts]
    assert any(p.name == "README.md" for p in sources), "the entry point must be scanned"
    stale = []
    for path in sources:
        text = path.read_text(errors="replace")
        for block in text.split("\n\n"):
            # Whitespace-normalized: the claim survived once purely by being line-wrapped.
            flat = " ".join(block.split())
            if not any(phrase in flat for phrase in BLOCKER_WORDINGS):
                continue
            if not any(marker in flat for marker in REFUTATIONS):
                stale.append(f"{path.relative_to(root)}: {flat[:90]}")
    assert not stale, stale


def _emitted_notes() -> list[tuple[str, str]]:
    """Every string this project puts in an artifact's `notes` list, found structurally.

    A docstring may quote the disproved blocker in order to refute it — several do, and
    deleting that history is how a blocker gets re-adopted. An emitted note may not, because
    it lands in a shipped artifact with no surrounding paragraph to correct it. Telling the
    two apart needs the parse tree, not a substring search over the file.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "src"
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and key.value == "notes"):
                    continue
                for item in getattr(value, "elts", []):
                    text = "".join(
                        part.value for part in ast.walk(item)
                        if isinstance(part, ast.Constant) and isinstance(part.value, str))
                    if text:
                        found.append((path.name, " ".join(text.split())))
    return found


def test_no_emitted_note_states_the_disproved_blocker():
    """The miss that mattered: a corrected docstring sitting above a note that still shipped
    the claim, so `useful-instructions.json` kept telling its reader no skill
    can be isolated for days after the measurement that disproved it."""
    notes = _emitted_notes()
    assert len(notes) > 10, "the extractor found almost nothing, so it is not checking"
    stale = [f"{where}: {text[:80]}" for where, text in notes
             if any(w in text for w in BLOCKER_WORDINGS)]
    assert not stale, stale
    # And the replacement really is in an emitted note, not only in the prose around it.
    # The two arms are named in different notes now that they are at different stages: the
    # skill arm is what the compliance probe runs on, while the server arm exists and still
    # has no machine-checkable task. Requiring one sentence to carry both would force the
    # output to keep describing a solved thing and an unsolved thing as one blocker.
    assert any("plugin-dir" in text for _, text in notes)
    assert any("strict-mcp-config" in text for _, text in notes)


def test_one_good_pair_does_not_claim_reproducibility():
    """A single pair has a spread of 0 by arithmetic, not by repeating.

    Four of the first seven real skill measurements lost a pair to the auth/init flake and
    every one of them reported `reproducible: true` off one observation — the strongest
    claim in the result, manufactured.
    """
    with _probe_returning(), _tmp() as tmp:
        one = cp.measure_skill_cost(_skill(tmp), pairs=1)
        two = cp.measure_skill_cost(_skill(tmp / "second"), pairs=2)

    assert one["ok"] is True and one["pair_spread_tokens"] == 0
    assert one["reproducible"] is None
    assert two["reproducible"] is True


def test_a_flaked_arm_is_retried_rather_than_losing_the_pair():
    """The flake runs near 40% and the surviving arm is billed either way, so a dead cell
    costs money and returns nothing."""
    attempts = []

    def flaky(prompt, workdir, instruction, timeout, *, bare=True, extra_args=()):
        armed = "--plugin-dir" in extra_args
        attempts.append("with" if armed else "without")
        # The first call fails the way the real harness does, with a 2-token context.
        if len(attempts) == 1:
            return {"ok": False, "error": "auth/init failure (ctx=2)", "cost_usd": 0.02}
        names = (["arm:demo-skill"] if armed else []) + BUILT_INS
        return {"ok": True, "text": "\n".join(names),
                "ctx": 17_259 if armed else 17_096, "cost_usd": 0.05}

    original, delay = up._probe, cp.ARM_RETRY_DELAY
    up._probe, cp.ARM_RETRY_DELAY = flaky, 0
    try:
        with _tmp() as tmp:
            result = cp.measure_skill_cost(_skill(tmp), pairs=1)
    finally:
        up._probe, cp.ARM_RETRY_DELAY = original, delay

    assert result["ok"] is True and result["differences"] == [163]
    assert len(attempts) == 3  # the failed arm, its retry, then the other arm
    # The failed attempt is still billed, which is the whole reason to retry it.
    assert result["cost_usd"] > 0.1


def test_a_skipped_instruction_file_keeps_why_each_arm_failed():
    """Found on a real run: three files came back `no pair completed` and nothing else.

    That one phrase covers a harness that never answered, a timeout, and a pair that ran and
    was rejected as unsound — three situations needing three different responses. The
    per-attempt errors were already being collected and then dropped when the skipped entry
    was built, so the artifact was less informative than the data behind it.
    `measure_skills` carried the detail; this path did not.
    """
    with _tmp() as tmp:
        home = tmp / "home"
        home.mkdir()
        target = tmp / "AGENTS.md"
        target.write_text("project rules\n", encoding="utf-8")

        calls = []

        def dead(home_dir, workdir, timeout):
            calls.append(str(workdir))
            return {"ok": False, "error": "exit 1: credentials could not be read"}

        original = cp._codex_probe
        cp._codex_probe = dead
        try:
            result = cp.measure_instruction_cost(target, pairs=1)
        finally:
            cp._codex_probe = original

    assert calls, "the probe must actually have been attempted"
    assert result["ok"] is False
    assert result["reason"] == "no pair completed"
    # The diagnosis has to survive into what a reader sees.
    errors = [p.get("error") for p in result["pairs"] if p.get("error")]
    assert errors and all("credentials" in e for e in errors), result["pairs"]


def _instruction_probe(*, versions=("0.150.0", "0.150.0"), models=("m-a", "m-a"),
                       tokens=(1000, 1400)):
    """A stub for the Codex arm probe, cycling through the given builds and models."""
    state = {"n": 0}

    def stub(home, workdir, timeout):
        i = state["n"] % len(versions)
        armed = (workdir / "AGENTS.md").is_file()
        state["n"] += 1
        return {"ok": True, "input_tokens": tokens[1] if armed else tokens[0],
                "cli_version": versions[i], "model": models[i]}
    return stub


def test_a_token_figure_records_the_build_and_model_that_produced_it():
    """Found by re-running a measurement two weeks after the first one.

    The same file, byte for byte, measured 25% higher — while both runs reported a pair
    spread of 0 and `reproducible: true`. Neither artifact named a build or a model, so the
    two figures could not be reconciled at all, and the earlier one had meanwhile been
    quoted as settled. A token count belongs to a tokenizer; the build picks the model.
    """
    with _tmp() as tmp:
        target = tmp / "AGENTS.md"
        target.write_text("project rules\n", encoding="utf-8")
        original = cp._codex_probe
        cp._codex_probe = _instruction_probe()
        try:
            result = cp.measure_instruction_cost(target, pairs=2)
        finally:
            cp._codex_probe = original

    assert result["ok"] is True
    assert result["measured_under"] == {"cli_version": "0.150.0", "model": "m-a"}
    # And the caveat has to travel with the figure, because a spread of 0 reads as
    # stability and is only agreement inside one window.
    assert "not stability across runs" in result["comparability"]


def test_arms_that_ran_under_different_builds_are_flagged_not_averaged():
    """The failure the field exists to catch: a build changing mid-measurement makes the
    two arms incomparable, so keeping the first value seen would hide it."""
    with _tmp() as tmp:
        target = tmp / "AGENTS.md"
        target.write_text("project rules\n", encoding="utf-8")
        original = cp._codex_probe
        cp._codex_probe = _instruction_probe(versions=("0.150.0", "0.151.0"),
                                             models=("m-a", "m-b"))
        try:
            result = cp.measure_instruction_cost(target, pairs=2)
        finally:
            cp._codex_probe = original

    under = result["measured_under"]
    assert under["cli_version"] is None and under["model"] is None
    assert under["mixed"]["cli_versions"] == ["0.150.0", "0.151.0"]
    assert "not a clean measurement" in under["warning"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} prefix-differencing regression tests passed (0 tokens spent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

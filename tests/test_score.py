"""Regression suite for preference-aware scoring.

No test here spends a token: scoring is deterministic. What is tested is that links are
never interpolated, that a score is withheld when the evidence a verdict depends on is
missing, and that a "next measurement" is never suggested for a gap it could not close.
"""

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import score as sc
from agent_profiler.cli import main as cli_main
from agent_profiler.score import (
    NO_EVENT_EMITTED,
    PREFERENCE_PROFILES,
    REQUIRED_FOR_SCORE,
    build_chain,
    score_component,
    score_root,
    sensitivity,
)
from agent_profiler import i18n

# Pinned so a developer's own locale cannot change what these assert; the zh strings
# are the ones this suite was written against. English rendering is covered separately.
i18n.set_lang("zh")

SESSION = "77777777-6666-5555-4444-333333333333"


def _rec(**kw):
    base = {"sessionId": SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, blocks):
    return _rec(type="assistant", timestamp="2026-11-18T10:00:01.000Z", message={
        "id": msg_id, "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0, "output_tokens": 5},
        "content": blocks})


def _trace_calling(tools):
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for i, tool in enumerate(tools):
        # A plain name, or (name, input) when the call's arguments matter.
        name, tool_input = tool if isinstance(tool, tuple) else (tool, {"q": "x"})
        records.append(_assistant(f"m{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": name, "input": tool_input}]))
        records.append(_rec(type="user", timestamp="2026-11-18T10:00:02.000Z",
                            message={"role": "user", "content": [
                                {"type": "tool_result", "tool_use_id": f"t{i}",
                                 "is_error": False, "content": "x"}]}))
    return records


@contextmanager
def _env(records, servers):
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
        try:
            yield root
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _fully_evidenced(name="full", *, cost_usd=1.0, calls=10, useful=0.9):
    """A component with every scored link present — the case real data never supplies."""
    return {
        "kind": "mcp", "name": name, "harness": "claude_code", "harnesses": ["claude_code"],
        "scope": "project",
        "links": {
            "installed": sc._link(sc.PROVEN, evidence="x", strength="observed"),
            "loaded": sc._link(sc.PROVEN, evidence="x", strength="observed"),
            "used": sc._link(sc.PROVEN, evidence="x", strength="observed", value=calls),
            "useful": sc._link(sc.PROVEN, evidence="x", strength="observed", value=useful),
            "cost": sc._link(sc.PARTIAL, evidence="x", strength="estimated", value=cost_usd),
            "recommendation": sc._unproven("n/a", sc.REQUIRES_EARLIER_LINK),
        },
    }


# -- links are never interpolated --------------------------------------------------------


def test_installed_is_not_evidence_of_used():
    with _env(_trace_calling(["Read"]), {"ghost": {"command": "x"}}) as root:
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        ghost = next(c for c in chain if c["name"] == "ghost")
        assert ghost["links"]["installed"]["status"] == sc.PROVEN
        assert ghost["links"]["useful"]["status"] == sc.UNPROVEN


def test_used_is_not_evidence_of_useful():
    with _env(_trace_calling(["mcp__s__go"] * 5), {"s": {"command": "x"}}) as root:
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        server = next(c for c in chain if c["name"] == "s")
        assert server["links"]["used"]["status"] == sc.PROVEN
        assert server["links"]["used"]["value"] == 5
        assert server["links"]["useful"]["status"] == sc.UNPROVEN


def _scan(root):
    from agent_profiler.core import inspect
    return inspect(str(root))


def _traces(root):
    from agent_profiler.traces import collect_root
    return collect_root(str(root))


# -- withholding ---------------------------------------------------------------------------


def test_a_component_missing_useful_is_not_scored():
    """The real-world case: everything measurable is present and the verdict is still withheld."""
    component = _fully_evidenced()
    component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
    result = score_component(component, "balanced")
    assert result["scored"] is False
    assert result["blocking_links"] == ["useful"]


def test_a_fully_evidenced_component_is_scored():
    """The machinery must actually work when the evidence exists, or withholding is vacuous."""
    result = score_component(_fully_evidenced(), "balanced")
    assert result["scored"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert result["strength"] == "estimated"
    assert "not preferences supplied by anyone" in result["caveat"]


def test_every_required_link_blocks_on_its_own():
    for link in REQUIRED_FOR_SCORE:
        component = _fully_evidenced()
        component["links"][link] = sc._unproven("gone", sc.NOT_MEASURED)
        assert score_component(component, "balanced")["scored"] is False, link


def test_stack_score_is_withheld_when_nothing_is_scorable():
    with _env(_trace_calling(["mcp__s__go"]), {"s": {"command": "x"}}) as root:
        r = score_root(str(root))
        assert r["stack_score"]["withheld"] is True
        assert r["stack_score"]["value"] is None
        assert "coverage" in r["stack_score"]["reason"]


def test_stack_score_is_withheld_when_a_minority_is_scorable():
    scorable = [_fully_evidenced(f"s{i}") for i in range(2)]
    withheld = [_fully_evidenced(f"w{i}") for i in range(8)]
    for c in withheld:
        c["links"]["useful"] = sc._unproven("gone", sc.NOT_MEASURED)
    out = sc._stack_score(scorable, scorable + withheld, "balanced")
    assert out["withheld"] is True
    assert "minority" in out["reason"]


def test_stack_score_is_produced_when_coverage_is_sufficient():
    scorable = [_fully_evidenced(f"s{i}") for i in range(8)]
    withheld = [_fully_evidenced(f"w{i}") for i in range(2)]
    for c in withheld:
        c["links"]["useful"] = sc._unproven("gone", sc.NOT_MEASURED)
    out = sc._stack_score(scorable, scorable + withheld, "balanced")
    assert out["withheld"] is False
    assert out["value"] is not None
    assert out["strength"] == "estimated"


# -- preferences ---------------------------------------------------------------------------


def test_the_preference_source_is_declared_as_assumed():
    """A score whose preferences the tool invented must never imply they are the user's."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        r = score_root(str(root))
        assert r["preference_source"] == "assumed_by_tool_not_elicited"
        assert any("Nobody" in n or "nobody" in n for n in r["notes"])


def test_profiles_disagree_when_cost_and_usage_point_opposite_ways():
    """A cheap unused component vs an expensive used one: the profile decides, so say so."""
    expensive_and_used = _fully_evidenced("busy", cost_usd=30.0, calls=60, useful=0.9)
    out = sensitivity(expensive_and_used)
    assert out["profiles"]["cost_sensitive"]["score"] != out["profiles"]["capability_first"]["score"]
    assert out["spread"] > 0
    if not out["stable"]:
        assert "changes the answer" in out["note"]


def test_sensitivity_is_undefined_when_nothing_is_scorable():
    component = _fully_evidenced()
    component["links"]["useful"] = sc._unproven("gone", sc.NOT_MEASURED)
    out = sensitivity(component)
    assert out["stable"] is None
    assert "undefined" in out["note"]


def test_unknown_profile_is_rejected():
    for call in (lambda: score_component(_fully_evidenced(), "vibes"),
                 lambda: score_root("/tmp", profile="vibes")):
        try:
            call()
        except ValueError as exc:
            assert "unknown preference profile" in str(exc)
        else:
            raise AssertionError("an unknown profile must be rejected")


def test_profile_weights_are_declared_and_normalized():
    for name, profile in PREFERENCE_PROFILES.items():
        assert abs(sum(profile["weights"].values()) - 1.0) < 1e-9, name
        assert profile["description"]


# -- next measurement ----------------------------------------------------------------------


def _with_skill(root, name="demo"):
    skills = root / ".claude" / "skills" / name
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "SKILL.md").write_text(f"---\nname: {name}\n---\nx\n", encoding="utf-8")


def test_an_unrecordable_link_is_called_unsolved_not_collectable():
    """Real FP: a component was told to 'collect traces' when no trace can record it.

    Commands are the honest case: nothing in a transcript names one entering context."""
    with _env(_trace_calling(["Read"]), {}) as root:
        commands = root / ".claude" / "commands"
        commands.mkdir(parents=True)
        (commands / "deploy.md").write_text("do a deploy\n", encoding="utf-8")
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        command = next((c for c in chain if c["kind"] == "commands"), None)
        assert command is not None, "the fixture command must be discovered"
        assert command["links"]["loaded"]["blocker"] == NO_EVENT_EMITTED
        nxt = sc._candidate(command, "loaded")
        assert nxt["achievable"] is False and nxt["method"] == "unsolved"
        assert "collect" not in nxt["note"].lower()
        assert "instrumentation" in nxt["note"]


def test_a_skill_listing_does_not_decide_other_kinds():
    """The listing names skills. Reading a command's absence from it as proof of never
    being loaded would manufacture evidence from a record that was never about commands."""
    with _env(_trace_calling(["Read"]), {}) as root:
        commands = root / ".claude" / "commands"
        commands.mkdir(parents=True)
        (commands / "deploy.md").write_text("x\n", encoding="utf-8")
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        command = next(c for c in chain if c["kind"] == "commands")
        assert command["links"]["loaded"]["status"] == sc.UNPROVEN
        assert "does not cover this kind" in command["links"]["loaded"]["evidence"]


def test_an_explicitly_invoked_skill_proves_its_used_link():
    """Real FN: Stage 7 first claimed a skill leaves no record. An explicit invocation is a
    `Skill` tool call that names it, and 20 such calls existed in the real corpus."""
    with _env(_trace_calling([("Skill", {"skill": "demo"})] * 3), {}) as root:
        _with_skill(root)
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        skill = next(c for c in chain if c["kind"] == "skills")
        assert skill["links"]["used"]["status"] == sc.PROVEN
        assert skill["links"]["used"]["value"] == 3


def test_a_never_invoked_skill_is_partial_not_a_proven_zero():
    """The asymmetry is the finding: a positive is solid, a negative is not, because the
    same skill can enter context by description match and leave no record."""
    with _env(_trace_calling(["Read"]), {}) as root:
        _with_skill(root)
        chain = build_chain(scan=_scan(root), traces=_traces(root))
        skill = next(c for c in chain if c["kind"] == "skills")
        used = skill["links"]["used"]
        assert used["status"] == sc.PARTIAL
        assert used["value"] == 0
        assert "not evidence that it went unused" in used["evidence"]


def test_the_cheapest_achievable_step_wins_over_an_earlier_unsolved_one():
    """Chain order must not pick: `useful` precedes `cost` in the narrative but is unsolved,
    while `cost` is a cheap probe that already works."""
    component = _fully_evidenced()
    component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
    component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    nxt = sc._next_measurement(component)
    assert nxt["link"] == "cost" and nxt["achievable"] is True
    assert nxt["estimated_usd"] == 0.30
    # The unsolved one is still reported, just not recommended.
    assert any(a["link"] == "useful" for a in nxt["alternatives"])


def test_free_beats_cheap_when_both_are_achievable():
    component = _fully_evidenced()
    component["links"]["used"] = sc._unproven("harness not traced", sc.HARNESS_NOT_TRACED)
    component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    assert sc._next_measurement(component)["link"] == "used"


def test_a_probe_is_not_suggested_for_a_harness_it_cannot_probe():
    """Real FP: Codex-declared servers were offered a Claude Code probe."""
    component = _fully_evidenced()
    component["harness"] = "codex"
    component["harnesses"] = ["codex"]
    component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
    cost = sc._candidate(component, "cost")
    assert cost["achievable"] is False
    assert cost["estimated_usd"] is None
    assert "Claude Code only" in cost["note"]
    # With nothing achievable, the recommendation must not invent a price.
    nxt = sc._next_measurement(component)
    assert nxt["achievable"] is False and nxt["estimated_usd"] is None


def test_the_useful_gap_names_the_blocker_that_actually_applies_to_each_kind():
    """`useful` is unreachable for MCP servers, skills, agents and commands — but no longer
    for the same reason, and one shared sentence hid that.

    Two blockers recorded here were never tested and both turned out false. The MCP note
    said an isolating arm "also hides credentials": `--strict-mcp-config --mcp-config <one
    server>` loads exactly that server, verified by asking each arm to list its own tools.
    The skill note said no flag disables one skill, which was true and beside the point —
    the arm works by *addition*: `--setting-sources ""` leaves the built-ins and
    `--plugin-dir` adds exactly one back, a reproducible positive difference with spread 0.

    So both notes must now name the half that is actually missing (a machine-checkable
    task), while a kind with no arm at all must not inherit their "an arm exists" wording.
    """
    for kind, flag in (("mcp", "strict-mcp-config"), ("skills", "plugin-dir")):
        component = _fully_evidenced()
        component["kind"] = kind
        component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
        nxt = sc._next_measurement(component)
        assert nxt["link"] == "useful" and nxt["achievable"] is False
        assert "0.014x" in nxt["note"]
        assert "an arm exists" in nxt["note"] and flag in nxt["note"]
        assert "machine can check" in nxt["note"]
        # The disproved blockers must not survive anywhere in the sentence.
        assert "hides credentials" not in nxt["note"]
        assert "cannot be disabled individually" not in nxt["note"]

    agent = _fully_evidenced()
    agent["kind"] = "agents"
    agent["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
    note = sc._next_measurement(agent)["note"]
    assert "no arm has been found" in note
    assert "an arm exists" not in note and "plugin-dir" not in note


def test_useful_is_achievable_for_an_instruction_file_stating_a_checkable_rule(tmp=None):
    """The narrower design Stage 6 left open, now implemented for instruction files."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "CLAUDE.md"
        path.write_text("Put the issue reference at the end of the subject, within 50 characters.", encoding="utf-8")
        component = _fully_evidenced()
        component["kind"] = "claude_md"
        component["path"] = str(path)
        component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
        nxt = sc._next_measurement(component)
    assert nxt["link"] == "useful" and nxt["achievable"] is True
    assert "compliance probe" in nxt["method"]
    assert nxt["estimated_usd"] > 0


def test_a_cost_in_tokens_is_scored_on_its_own_curve_not_as_dollars():
    """Real defect, found on the first component ever to satisfy every link.

    A Codex instruction file costing a few hundred tokens per request reached `scorable`,
    and the cost signal `1/(1+value)` read that token count as dollars: a signal near zero,
    i.e. ruinously expensive, for something cheap. The first fix blocked every non-dollar
    unit, which blocked every
    component in the environment, because no route to a per-component cost reports money.
    The defect was the curve, not the unit: tokens per request are a real cost measure.
    """
    component = _fully_evidenced(name="AGENTS.md")
    component["kind"] = "agents_md"
    component["links"]["cost"] = sc._link(sc.PARTIAL, evidence="450 tokens per request",
                                          strength="observed", value=450,
                                          unit="tokens_per_request")
    coverage = sc._coverage(component)
    assert coverage["scorable"] is True
    result = sc.score_component(component, "balanced")
    assert result["scored"] is True
    # The original symptom: 450 must not read as ruinous.
    assert result["signals"]["cost"] > 0.5
    assert result["cost_unit"] == "tokens_per_request"
    # The unit travels with the score, so no reader can mistake which curve produced it.
    assert result["cost_scale"] == sc.COST_SCALES["tokens_per_request"]


def test_the_cost_scale_is_a_reproducible_derivation_not_a_round_guess():
    """The scale was a round 1,000 with a comment claiming a range it had never been checked
    against. It is the point where the cost curve reads 0.5, so it has to sit near the centre
    of the blocks actually measured or the signal saturates and every component scores alike.

    The measurements it was derived from are one private stack's component sizes and are not
    published, so what is tested is the *derivation* and the property it buys. A reader
    re-derives the constant from their own artifacts.
    """
    import statistics

    cal = sc.COST_SCALE_CALIBRATION["tokens_per_request"]
    # No inventory of someone else's components, in any field of the calibration.
    for value in cal.values():
        assert not re.search(r"\d{3,}", str(value)), value

    # The derivation is the geometric centre, to the nearest 50 — on any sample set.
    for samples in ((100, 1000), (150, 400, 400, 700, 2000, 2300), (650,)):
        expected = int(round(statistics.geometric_mean(samples) / 50) * 50)
        assert sc.scale_from(samples) == expected, samples
    assert sc.scale_from([200, 800]) == 400  # pinned, so a rewrite cannot drift silently

    # And the property the derivation exists for: a scale at the centre of a set spanning an
    # order of magnitude spreads the signal, where a scale above the set bunches it.
    blocks = (150, 400, 700, 2300)
    spread = {}
    for label, scale in (("centred", sc.scale_from(blocks)), ("above", 10_000)):
        signals = sorted(1 / (1 + v / scale) for v in blocks)
        spread[label] = signals[-1] - signals[0]
    assert spread["centred"] > 0.5 > spread["above"], spread

    # The shipped constant is on the same rounding grid the derivation produces.
    assert sc.COST_SCALES["tokens_per_request"] % 50 == 0


def test_the_cost_scale_moves_scores_but_never_their_order():
    """The scale constant is a convention, so the claim that it does not decide rankings
    has to hold rather than be asserted in a comment."""
    cheap = _fully_evidenced(name="cheap")
    dear = _fully_evidenced(name="dear")
    for component, value in ((cheap, 100), (dear, 5000)):
        component["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="observed",
                                              value=value, unit="tokens_per_request")
    order = []
    for scale in (1.0, 1000.0, 1e6):
        saved = sc.COST_SCALES["tokens_per_request"]
        sc.COST_SCALES["tokens_per_request"] = scale
        try:
            a = sc.score_component(cheap, "balanced")["score"]
            b = sc.score_component(dear, "balanced")["score"]
        finally:
            sc.COST_SCALES["tokens_per_request"] = saved
        order.append(a > b)
    assert order == [True, True, True]


def test_a_cost_in_an_unrecognized_unit_still_blocks():
    """A unit with no scale cannot be placed on any curve, so it is evidence and not a
    score — the distinction the dollar rule was reaching for."""
    component = _fully_evidenced(name="odd")
    component["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="observed",
                                          value=6135, unit="chars_per_session")
    coverage = sc._coverage(component)
    assert coverage["scorable"] is False
    assert coverage["blocking"] == ["cost"]
    result = sc.score_component(component, "balanced")
    assert result["scored"] is False
    assert "chars_per_session" in result["reason"]


def test_scores_built_on_different_cost_units_are_never_averaged():
    """A cost signal is a position on a curve. Two curves calibrated for different units
    share an axis only by coincidence, so a stack total spanning both means nothing."""
    tokens = _fully_evidenced(name="a")
    tokens["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="observed",
                                       value=450, unit="tokens_per_request")
    dollars = _fully_evidenced(name="b")
    dollars["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="estimated",
                                        value=0.5, unit="usd")
    both = [tokens, dollars]
    assert sc.comparable_cost_units(both) == {"tokens_per_request", "usd"}
    stack = sc._stack_score(both, both, "balanced")
    assert stack["withheld"] is True and "different units" in stack["reason"]

    # One unit throughout, and the same components total normally.
    dollars["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="observed",
                                        value=90, unit="tokens_per_request")
    assert sc._stack_score(both, both, "balanced")["withheld"] is False


def test_the_useful_signal_reads_status_not_the_rule_count():
    """The value is a count of rules proven; a count in a 0-1 signal breaks the total."""
    component = _fully_evidenced()
    component["links"]["useful"] = sc._link(sc.PROVEN, evidence="x", strength="observed",
                                            value=2, unit="rules_proven")
    result = sc.score_component(component, "balanced")
    assert result["signals"]["useful"] == 1.0
    assert 0.0 <= result["score"] <= 1.0

    component["links"]["useful"] = sc._link(sc.PARTIAL, evidence="x", strength="observed")
    assert sc.score_component(component, "balanced")["signals"]["useful"] == 0.5


def test_no_measurement_is_proposed_for_an_excluded_component():
    """Real hole: the report offered `evaluate --measure` against an excluded MCP server.

    An exclusion exists to stop this tool contacting an endpoint. Suggesting the one method
    that contacts it hands the user a step that violates their own instruction.
    """
    component = _fully_evidenced(name="off-limits")
    component["excluded_by_request"] = True
    component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    cost = sc._candidate(component, "cost")
    assert cost["achievable"] is False
    assert cost["estimated_usd"] is None
    assert "not to contact" in cost["note"]


def test_nested_mcp_inventory_is_not_offered_as_a_runnable_schema_probe():
    from agent_profiler.evaluate import evaluate_root

    with _env(_trace_calling([]), {"direct": {"command": "fixture"}}) as root:
        nested = root / "bundle"
        nested.mkdir()
        (nested / ".mcp.json").write_text(json.dumps({"mcpServers": {
            "nested-only": {"command": "fixture"}, "direct": {"command": "fixture"}
        }}))
        data = score_root(str(root))
        components = {c["name"]: c for c in data["components"] if c["kind"] == "mcp"}
        offered = {name for name, c in components.items()
                   if sc._candidate(c, "cost")["achievable"]}
        runnable = {e["server"] for e in evaluate_root(str(root))["evaluations"]}
    assert "nested-only" in components, "unsupported declarations must remain visible"
    assert offered == runnable == {"direct"}
    assert sc._candidate(components["nested-only"], "cost")["estimated_usd"] is None


def test_a_claude_instruction_cost_is_offered_because_the_route_now_works():
    """It was recorded unsolved on a half-true premise: removal from a copied config does
    hide the auth token, but *additive injection* was abandoned after an unpaired run
    drifted — before interleaved pairs existed. With them it returns the same figure in both
    arm orders, spread 0. A blocker that has been cleared must stop being reported."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "CLAUDE.md"
        path.write_text("rules", encoding="utf-8")
        component = _fully_evidenced(name="CLAUDE.md")
        component["kind"] = "claude_md"
        component["path"] = str(path)
        component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
        nxt = sc._next_measurement(component)

    assert nxt["link"] == "cost" and nxt["achievable"] is True
    assert nxt["estimated_usd"] == 0.15
    assert "unsolved" not in nxt["method"]
    # The channel caveat has to travel with the offer, not be discovered afterwards.
    assert "byte-identical" in nxt["note"]


def test_an_mcp_cost_is_the_measured_per_request_tokens_not_the_corpus_projection():
    """The aggregation trap for the fifth time. The probe measures a per-request token
    figure; the dollar figure is that projected over a whole corpus. Carrying it as the
    link's value priced the server at a whole-corpus projection *per request* on the cost
    curve, and kept the
    one directly measured number out of the stack floor where it belongs."""
    scan = {"observed": {"components": {}}, "files": [],
            "mcp_servers": [{"name": "pw", "source": "user_claude_json", "scope": "user",
                             "declared_in": "/x/.claude.json"}]}
    traces = {"observed": {"sessions_collected": 2, "calls_by_tool": {"mcp__pw__go": 3}}}
    evaluation = {"evaluations": [{
        "server": "pw", "status": "measured",
        "measurement": {"schema_tokens": 470, "harness": "claude_code"},
        "projection": {"schema_tokens": 470, "estimated_usd": 10.62,
                       "total_tokens": 5961110}}]}
    link = sc.build_chain(scan=scan, traces=traces, evaluation=evaluation)[0]["links"]["cost"]
    assert link["value"] == 470 and link["unit"] == "tokens_per_request"
    assert link["harness"] == "claude_code"
    # The projection is not lost — it moves to the evidence, labelled cumulative.
    assert "10.62" in link["evidence"] and "cumulative" in link["evidence"]


def test_a_multi_harness_component_counts_only_toward_the_harness_it_was_measured_in():
    """A server declared for two harnesses was measured under one. Adding its figure to
    both floors asserts a prefix cost where it was never observed — search-server, measured
    through `claude -p`, was being added to Codex's floor, where it is reached through a
    different plugin."""
    component = _fully_evidenced(name="pw")
    component["kind"] = "mcp"
    component["scope"] = "user"
    component["harnesses"] = ["claude_code", "codex"]
    component["links"]["cost"] = sc._link(sc.PARTIAL, evidence="x", strength="observed",
                                          value=470, unit="tokens_per_request")
    component["links"]["cost"]["harness"] = "claude_code"

    out = sc.stack_cost([component])
    assert out["claude_code"]["per_request_tokens_floor"] == 470
    assert "codex" not in out or not out["codex"].get("per_request_tokens_floor")
    reasons = [n["why"] for n in out.get("codex", {}).get("not_summable", [])]
    assert any("measured under claude_code" in r for r in reasons)


def test_one_evaluation_file_can_carry_both_kinds_of_cost_measurement():
    """Real defect: measuring an MCP schema and passing the new artifact dropped
    the instruction-file cost measured earlier, and the only scorable component in the
    environment went back to unscorable. The two measurements live in disjoint keys, so a
    merged file must satisfy both readers rather than one replacing the other.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "AGENTS.md"
        path.write_text("rule", encoding="utf-8")
        scan = {"observed": {"components": {}},
                "files": [{"kind": "agents_md", "origin": "declared", "path": str(path),
                           "scope": "user", "bytes": 4}],
                "mcp_servers": [{"name": "pw", "source": "user_claude_json",
                                 "scope": "user", "declared_in": "/x/.claude.json"}]}
        traces = {"observed": {"sessions_collected": 2, "calls_by_tool": {"mcp__pw__go": 3}}}
        merged = {
            "evaluations": [{"server": "pw", "status": "measured",
                             "measurement": {"schema_tokens": 470},
                             "projection": {"schema_tokens": 470, "estimated_usd": 10.62,
                                            "total_tokens": 5961110}}],
            "instruction_measurements": [{"file": str(path), "ok": True,
                                          "loaded": "proven", "loaded_evidence": "x",
                                          "mean_tokens": 450, "pair_spread_tokens": 0,
                                          "reproducible": True, "differences": [450, 450],
                                          "pairs": [{"pair": 1, "difference": 450},
                                                    {"pair": 2, "difference": 450}]}],
        }
        components = {c["name"]: c for c in
                      sc.build_chain(scan=scan, traces=traces, evaluation=merged)}

    assert components["pw"]["links"]["cost"]["status"] != sc.UNPROVEN
    instruction = next(c for n, c in components.items() if n.startswith("AGENTS"))
    assert instruction["links"]["cost"]["status"] != sc.UNPROVEN
    assert instruction["links"]["cost"]["unit"] == "tokens_per_request"


def _measurement(path, tokens, *, version=None):
    return {"file": str(path), "ok": True, "loaded": "proven", "loaded_evidence": "x",
            "mean_tokens": tokens, "pair_spread_tokens": 0, "reproducible": True,
            "differences": [tokens, tokens],
            "pairs": [{"pair": 1, "difference": tokens}, {"pair": 2, "difference": tokens}],
            **({"measured_under": {"cli_version": version, "model": "m"}} if version else {})}


def test_two_measurements_of_one_file_that_disagree_are_not_silently_resolved():
    """Real case, found while merging two artifacts to refresh a report.

    One instruction file had been measured twice, two weeks apart, and the figures differed
    by a quarter — each run reporting a pair spread of 0. The index was a dict comprehension,
    so the number a reader saw depended on which record came last, and nothing said a second
    one existed.

    Precedence is unchanged (later wins); what is required is that the disagreement is
    visible, because a spread of 0 beside a contested figure reads as settled precision.
    """
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        path = root / "AGENTS.md"
        path.write_text("rules\n", encoding="utf-8")
        scan = {"files": [{"kind": "agents_md", "path": str(path), "origin": "declared",
                           "scope": "project", "bytes": 6}],
                "observed": {"components": {}}, "mcp_servers": []}
        traces = {"observed": {"sessions_collected": 2, "calls_by_tool": {}}}
        evaluation = {"instruction_measurements": [
            _measurement(path, 380, version="0.144.0"),
            _measurement(path, 474, version="0.147.0"),
        ]}
        components = sc.build_chain(scan=scan, traces=traces, evaluation=evaluation)

    instruction = next(c for c in components if c["name"].startswith("AGENTS"))
    cost = instruction["links"]["cost"]
    # The later figure is used, as it always was.
    assert cost["value"] == 474
    # And the reader is told it is contested, with both numbers.
    assert "not settled" in cost["evidence"]
    assert "380" in cost["evidence"] and "474" in cost["evidence"]
    conflict = cost["conflict"]
    assert conflict["used"] == 474
    assert [f["mean_tokens"] for f in conflict["figures"]] == [380, 474]
    # What each was measured under, which is the only way to act on the disagreement.
    assert [f["measured_under"]["cli_version"] for f in conflict["figures"]] \
        == ["0.144.0", "0.147.0"]
    # Reported at the top level too, beside the other join problem it resembles.
    assert sc.measurement_conflicts(evaluation)[0]["file"] == str(path)


def test_one_file_measured_in_both_harnesses_is_two_facts_not_a_disagreement():
    """A real environment symlinks `AGENTS.md` to `CLAUDE.md`, so the same bytes are loaded
    by both harnesses and cost a different amount in each. The first version of the conflict
    check called that a contradiction.

    The deeper defect it exposed: resolving the path — which is what lets an artifact join
    across a `/var` symlink — collapsed the user's own symlink too, so two components and
    two separately paid measurements landed on one key and the Codex figure was silently
    replaced by the Claude one.
    """
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        target = root / "CLAUDE.md"
        target.write_text("rules\n", encoding="utf-8")
        link = root / "AGENTS.md"
        link.symlink_to(target.name)

        evaluation = {"instruction_measurements": [
            dict(_measurement(link, 1723), harness="codex"),
            dict(_measurement(target, 1928), harness="claude_code"),
        ]}
        assert sc.measurement_conflicts(evaluation) == []

        index, _ = sc._index_measurements(
            evaluation["instruction_measurements"], lambda v: str(Path(v).resolve()))
        # Both measurements survive the join, which they did not before.
        assert sorted(m["mean_tokens"] for m in index.values()) == [1723, 1928]
        assert len(index) == 2


def test_an_unlabelled_measurement_is_still_placed_in_its_harness():
    """Artifacts written before the label existed must keep joining. The file's own name is
    a sound fallback, not a guess: a CLAUDE.md is measured through Claude Code and an
    AGENTS.md through Codex by construction."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        agents, claude = root / "AGENTS.md", root / "CLAUDE.md"
        for p in (agents, claude):
            p.write_text("rules\n", encoding="utf-8")
        assert sc._measurement_harness({"file": str(agents)}, None) == "codex"
        assert sc._measurement_harness({"file": str(claude)}, None) == "claude_code"
        # An explicit label wins over the name, and a header wins over nothing.
        assert sc._measurement_harness({"file": str(agents), "harness": "claude_code"},
                                       None) == "claude_code"
        assert sc._measurement_harness({"file": str(root / "other.md")},
                                       "codex") == "codex"


def test_the_same_file_measured_twice_with_the_same_result_is_not_a_conflict():
    """Re-running a measurement and getting the same answer is confirmation. Reporting it
    as a disagreement would make a reader distrust the one case that should reassure them."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "AGENTS.md"
        path.write_text("rules\n", encoding="utf-8")
        evaluation = {"instruction_measurements": [
            _measurement(path, 474), _measurement(path, 474)]}
        assert sc.measurement_conflicts(evaluation) == []


def test_an_empty_instruction_file_is_not_offered_a_cost_measurement():
    """Differencing an empty file once returned -3 tokens and nearly became a finding."""
    component = _fully_evidenced(name="AGENTS.md")
    component["kind"] = "agents_md"
    component["bytes"] = 0
    component["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    cost = sc._candidate(component, "cost")
    assert cost["achievable"] is False
    assert "empty" in cost["note"]

    component["bytes"] = 1134
    assert sc._candidate(component, "cost")["achievable"] is True


def _instruction_scan(path):
    return {"files": [{"kind": "claude_md", "origin": "declared", "path": str(path),
                       "bytes": 100, "scope": "user"}]}


def test_a_useful_measurement_populates_the_chain():
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "CLAUDE.md"
        path.write_text("Put the issue reference at the end of the subject.", encoding="utf-8")
        useful = {"useful_measurements": [{
            "file": str(path), "useful": "proven",
            "useful_evidence": "the arm carrying this file satisfied 2 stated rule(s)",
            "rules_proven": ["subject_ends_with_reference", "body_line_length_limit"],
        }]}
        components = sc._instruction_components(
            _instruction_scan(path), {"observed": {"sessions_collected": 5}}, None, useful)

    assert len(components) == 1
    links = components[0]["links"]
    assert links["useful"]["status"] == sc.PROVEN
    assert links["useful"]["value"] == 2
    # `used` rides on the same experiment, and the evidence must say so outright: two links
    # turning green off one observation would otherwise read as independent corroboration.
    assert links["used"]["status"] == sc.PROVEN
    assert "same observation as the useful link" in links["used"]["evidence"]
    # Cost is still unmeasured for a Claude instruction file, so no recommendation is due,
    # and the reason must name the link that is actually missing.
    assert links["recommendation"]["status"] == sc.UNPROVEN
    assert "the cost link" in links["recommendation"]["evidence"]
    assert "the useful link" not in links["recommendation"]["evidence"]


def test_a_refused_useful_measurement_is_not_silently_unmeasured():
    """A measurement that ran its controls and declined is a different state from no try."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "CLAUDE.md"
        path.write_text("Put the issue reference at the end of the subject.", encoding="utf-8")
        useful = {"useful_measurements": [{
            "file": str(path), "useful": "unproven", "blocker": "CONTROL_FAILED",
            "reason": "the negative control did not pass in both arms",
        }]}
        components = sc._instruction_components(
            _instruction_scan(path), {"observed": {"sessions_collected": 5}}, None, useful)

    links = components[0]["links"]
    assert links["useful"]["status"] == sc.UNPROVEN
    assert links["useful"]["blocker"] == "CONTROL_FAILED"
    assert "negative control" in links["useful"]["evidence"]


def test_useful_is_refused_for_an_instruction_file_with_no_checkable_rule():
    """A probe against a file stating no verifiable rule would refuse; do not offer it."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "CLAUDE.md"
        path.write_text("Be thoughtful and write clean code.", encoding="utf-8")
        component = _fully_evidenced()
        component["kind"] = "claude_md"
        component["path"] = str(path)
        component["links"]["useful"] = sc._unproven("not measured", sc.NOT_MEASURED)
        nxt = sc._next_measurement(component)
    assert nxt["achievable"] is False
    assert "rule a machine can check" in nxt["note"]


def test_a_fully_evidenced_component_needs_nothing():
    assert sc._next_measurement(_fully_evidenced()) is None


# -- real-shape assembly -------------------------------------------------------------------


def test_a_server_declared_twice_is_one_component_with_both_harnesses():
    """Real FP: first-declaration-wins labelled a dual-declared server with one harness."""
    scan = {"mcp_servers": [
        {"name": "dual", "source": "project_root", "declared_in": "/p/.mcp.json", "scope": "project"},
        {"name": "dual", "source": "project_codex_dir", "declared_in": "/p/.codex/config.toml",
         "scope": "project"}],
        "observed": {"components": {}}}
    chain = build_chain(scan=scan, traces={"observed": {"calls_by_tool": {}, "harnesses": ["claude_code"]}})
    assert len(chain) == 1
    assert chain[0]["harnesses"] == ["claude_code", "codex"]
    assert "/p/.mcp.json" in chain[0]["links"]["installed"]["evidence"]


def test_an_untraced_harness_blocks_used_rather_than_proving_zero():
    """Stage 4's lesson: never called is only a fact if the harness was actually traced."""
    scan = {"mcp_servers": [{"name": "cx", "source": "user_codex_dir", "declared_in": "/c",
                             "scope": "user"}], "observed": {"components": {}}}
    # Claude Code really was traced, so the gap is the harness, not missing collection.
    chain = build_chain(scan=scan,
                        traces={"observed": {"calls_by_tool": {}, "harnesses": ["claude_code"],
                                             "sessions_collected": 94}})
    assert chain[0]["links"]["used"]["status"] == sc.UNPROVEN
    assert chain[0]["links"]["used"]["blocker"] == sc.HARNESS_NOT_TRACED


def test_scoring_spends_nothing_and_touches_nothing():
    with _env(_trace_calling(["mcp__s__go"]), {"s": {"command": "x"}}) as root:
        r = score_root(str(root))
        assert r["profiler_cost"]["cost_usd"] == 0.0
        assert r["profiler_cost"]["llm_input_tokens"] == 0
        assert r["read_only"] is True


def test_no_counterfactual_or_removal_claims_in_output():
    with _env(_trace_calling(["mcp__s__go"]), {"s": {"command": "x"}}) as root:
        blob = json.dumps(score_root(str(root))).lower()
        for banned in ("would save", "could save", "should be removed", "useless", "wasted"):
            assert banned not in blob, banned


def test_cli_errors():
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        assert cli_main(["score", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["score", "--root", ""]) == 2
        assert cli_main(["score", "--root", str(root / "nope")]) == 2



def test_an_empty_corpus_proves_nothing_about_usage():
    """Real FP from the report dogfood: a moved project collected 0 sessions, and
    'no call in 0 traced sessions' was being reported as a proven zero."""
    scan = {"mcp_servers": [{"name": "s", "source": "project_root", "declared_in": "/p",
                             "scope": "project"}],
            "observed": {"components": {"skills": {"names": ["user/claude_code/demo"]}}}}
    traces = {"observed": {"calls_by_tool": {}, "harnesses": ["claude_code"],
                           "sessions_collected": 0}}
    chain = build_chain(scan=scan, traces=traces)
    for component in chain:
        for link in ("loaded", "used"):
            info = component["links"][link]
            assert info["status"] == sc.UNPROVEN, (component["name"], link)
            assert info["blocker"] == sc.NO_TRACES_COLLECTED, (component["name"], link)
    # And the recommended fix points at collection, free, mentioning the moved-project case.
    nxt = sc._next_measurement(chain[0])
    assert nxt["achievable"] is True and nxt["estimated_usd"] == 0.0
    assert "relocation_candidates" in nxt["note"]


def test_a_probe_measurement_survives_an_empty_corpus():
    """The Stage 6 probe measures the live prefix, so it holds with no transcripts at all."""
    scan = {"mcp_servers": [{"name": "s", "source": "project_root", "declared_in": "/p",
                             "scope": "project"}], "observed": {"components": {}}}
    traces = {"observed": {"calls_by_tool": {}, "harnesses": ["claude_code"],
                           "sessions_collected": 0}}
    evaluation = {"evaluations": [{"server": "s", "status": "measured",
                                   "projection": {"schema_tokens": 3331, "total_tokens": 1,
                                                  "estimated_usd": 0.1, "assumptions": ["x"]}}]}
    chain = build_chain(scan=scan, traces=traces, evaluation=evaluation)
    assert chain[0]["links"]["loaded"]["status"] == sc.PROVEN


def test_a_listing_does_not_credit_an_untraced_harness_with_the_same_name():
    """Real FP from self-dogfooding: `demo-skill` exists for both harnesses, and the Codex
    one was credited with the Claude listing's loads. A listing names a skill without
    saying which harness it belongs to."""
    scan = {"mcp_servers": [], "observed": {"components": {"skills": {
        "names": ["user/claude_code/demo-skill", "user/codex/demo-skill"]}}}}
    traces = {"observed": {"calls_by_tool": {}, "harnesses": ["claude_code"],
                           "components_loaded": {"demo-skill": 23}, "sessions_collected": 23}}
    chain = build_chain(scan=scan, traces=traces)
    by_harness = {c["harness"]: c for c in chain}
    assert by_harness["claude_code"]["links"]["loaded"]["status"] == sc.PROVEN
    codex = by_harness["codex"]["links"]["loaded"]
    assert codex["status"] == sc.UNPROVEN
    assert codex["blocker"] == sc.HARNESS_NOT_TRACED



def test_a_moved_project_does_not_report_a_clean_zero():
    """Real case: the project tree moved mid-session and the history stayed under the old
    path. A score reading 'no usage evidence' without mentioning that is misleading."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        home = Path(h)
        moved = Path(d) / "demo"
        moved.mkdir()
        (moved / ".mcp.json").write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}),
                                         encoding="utf-8")
        old_cwd = "/old/path/demo"
        old = home / ".claude" / "projects" / old_cwd.replace("/", "-")
        old.mkdir(parents=True)
        (old / f"{SESSION}.jsonl").write_text(json.dumps({
            "type": "user", "cwd": old_cwd, "sessionId": SESSION,
            "timestamp": "2026-11-18T10:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "x"}]}}) + "\n",
            encoding="utf-8")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            r = score_root(str(moved))
            assert r["observed"]["components_scorable"] == 0
            candidates = r["relocation_candidates"]
            assert len(candidates) == 1, "the orphaned history must reach the score output"
            assert candidates[0]["recorded_cwd"] == old_cwd
            assert candidates[0]["sessions"] == 1
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v



# -- real preferences vs invented ones ---------------------------------------------------


def _prefs_file(tmp, profiles):
    path = Path(tmp) / "prefs.json"
    path.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")
    return str(path)


def test_without_a_preferences_file_the_output_says_the_weights_were_assumed():
    """The failure this fixes: "preference-aware" was a claim about weights nobody supplied."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        r = score_root(str(root))
        assert r["preference_source"] == sc.ASSUMED
        assert r["preferences_file"] is None
        assert sc.PREFERENCES_FILENAME in r["preferences_hint"]


def test_the_prose_never_contradicts_the_preference_source():
    """Real defect: `preference_source` said `supplied_by_user` while
    the notes still read "Nobody supplied preferences". On a withheld stack the notes are
    the only place preferences are described, so the artifact's one statement was wrong."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        assumed = json.dumps(score_root(str(root))["notes"], ensure_ascii=False)
        assert "Nobody supplied preferences" in assumed

        with tempfile.TemporaryDirectory() as tmp:
            path = _prefs_file(tmp, {"mine": {"description": "what I care about",
                                              "weights": {"cost": 1, "used": 1, "useful": 1}}})
            supplied = score_root(str(root), preferences_path=path)
        notes = json.dumps(supplied["notes"], ensure_ascii=False)
        assert supplied["preference_source"] == sc.ELICITED
        assert "Nobody supplied preferences" not in notes
        assert supplied["preferences_file"] in notes


def test_supplied_preferences_are_used_and_marked_as_the_users():
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        with tempfile.TemporaryDirectory() as tmp:
            path = _prefs_file(tmp, {"mine": {"description": "what I care about",
                                              "weights": {"cost": 3, "used": 2, "useful": 1}}})
            r = score_root(str(root), preferences_path=path)
        assert r["preference_source"] == sc.ELICITED
        assert r["profile_used"] == "mine"
        assert r["profiles_available"] == ["mine"]
        assert r["preferences_hint"] is None


def test_a_preferences_file_at_the_root_is_found_without_a_flag():
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        (root / sc.PREFERENCES_FILENAME).write_text(json.dumps({"profiles": {
            "mine": {"weights": {"cost": 1, "used": 1, "useful": 1}}}}), encoding="utf-8")
        r = score_root(str(root))
        assert r["preference_source"] == sc.ELICITED
        assert r["preferences_file"].endswith(sc.PREFERENCES_FILENAME)


def test_relative_weights_are_normalized_so_profiles_stay_comparable():
    """3/2/1 is a ratio, not a demand that the user do arithmetic."""
    profiles, source, _origin = sc.load_preferences(None, Path("/nonexistent"))
    assert source == sc.ASSUMED
    with tempfile.TemporaryDirectory() as tmp:
        path = _prefs_file(tmp, {"mine": {"weights": {"cost": 3, "used": 2, "useful": 1}}})
        loaded, source, _ = sc.load_preferences(path, Path(tmp))
    assert source == sc.ELICITED
    weights = loaded["mine"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["cost"] > weights["used"] > weights["useful"]


def test_supplied_weights_actually_change_the_score():
    """If the file is accepted but ignored, the fix is cosmetic."""
    component = _fully_evidenced(cost_usd=30.0, calls=60, useful=0.9)
    cheap = {"p": {"weights": {"cost": 0.9, "used": 0.05, "useful": 0.05}}}
    rich = {"p": {"weights": {"cost": 0.05, "used": 0.05, "useful": 0.9}}}
    with tempfile.TemporaryDirectory() as tmp:
        a, _, _ = sc.load_preferences(_prefs_file(tmp, cheap), Path(tmp))
        b, _, _ = sc.load_preferences(_prefs_file(tmp + "", rich), Path(tmp))
    low = score_component(component, "p", a)["score"]
    high = score_component(component, "p", b)["score"]
    assert low != high
    assert score_component(component, "p", a)["caveat"] == "the weights were supplied by the user"


def test_a_malformed_preferences_file_is_rejected_rather_than_silently_ignored():
    """Falling back to invented weights after being handed real ones would be the worst
    outcome: the output would claim `supplied_by_user` for weights the user never wrote."""
    cases = [
        ({}, "no `profiles` object"),
        ({"mine": {}}, "no `weights` object"),
        ({"mine": {"weights": {"cost": 1}}}, "missing weights for"),
        ({"mine": {"weights": {"cost": 1, "used": 1, "useful": 1, "installed": 1}}},
         "unscorable links"),
        ({"mine": {"weights": {"cost": "x", "used": 1, "useful": 1}}}, "non-numeric"),
        ({"mine": {"weights": {"cost": -1, "used": 1, "useful": 1}}}, "negative"),
        ({"mine": {"weights": {"cost": 0, "used": 0, "useful": 0}}}, "summing to zero"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for profiles, expected in cases:
            path = _prefs_file(tmp, profiles)
            try:
                sc.load_preferences(path, Path(tmp))
            except ValueError as exc:
                assert expected in str(exc), (profiles, str(exc))
            else:
                raise AssertionError(f"should have been rejected: {profiles}")


def test_a_named_preferences_file_that_is_missing_is_an_error():
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        try:
            score_root(str(root), preferences_path=str(root / "nope.json"))
        except FileNotFoundError as exc:
            assert "preferences file does not exist" in str(exc)
        else:
            raise AssertionError("a missing named file must not fall back to invented weights")


def test_an_unknown_profile_names_what_is_available():
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        try:
            score_root(str(root), profile="vibes")
        except ValueError as exc:
            assert "available:" in str(exc)
        else:
            raise AssertionError("an unknown profile must be rejected")


# -- instruction files as components -----------------------------------------------------


def test_instruction_files_are_components_with_their_own_evidence():
    """They were missing from the chain entirely, which is the largest omission the chain
    had: an instruction file sits in the fixed prefix of *every* request."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        (root / "CLAUDE.md").write_text("project instructions\n", encoding="utf-8")
        (root / "AGENTS.md").write_text("codex instructions\n", encoding="utf-8")
        r = score_root(str(root))
    kinds = {c["kind"] for c in r["components"]}
    assert {"claude_md", "agents_md"} <= kinds, kinds

    for c in r["components"]:
        if c["kind"] not in ("claude_md", "agents_md"):
            continue
        # `used` must not be faked either way: a file is not invoked, and loading it is
        # not evidence that it influenced anything.
        used = c["links"]["used"]
        assert used["status"] == sc.UNPROVEN
        assert used["blocker"] == sc.NO_EVENT_EMITTED
        assert "not invoked" in used["evidence"]
        assert c["links"]["useful"]["status"] == sc.UNPROVEN


def test_a_codex_measurement_proves_loading_that_no_record_reports():
    """The audit had recorded Codex instruction loading as CONFIRMED unobservable, on the
    evidence that `base_instructions` is empty in every session examined. It is empty — and
    the instructions still enter the prefix. A measured difference is the evidence."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        agents = root / "AGENTS.md"
        agents.write_text("project rules\n", encoding="utf-8")
        # Deliberately the *unresolved* spelling: on macOS the scan resolves /var to
        # /private/var, and a measurement keyed on the raw path used to be dropped.
        evaluation = {"instruction_measurements": [{
            "file": str(agents), "ok": True, "loaded": "proven",
            "loaded_evidence": "injecting this 14-byte file moved the fixed prefix by 450 "
                               "tokens across 2 interleaved pair(s), spread 0",
            "mean_tokens": 450, "differences": [450, 450], "pair_spread_tokens": 0,
        }]}
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(evaluation, fh)
            path = fh.name
        try:
            r = score_root(str(root), evaluation_path=path)
        finally:
            os.unlink(path)

    entry = next(c for c in r["components"]
                 if c["kind"] == "agents_md"
                 and Path(c["path"]).resolve() == agents.resolve())
    assert entry["links"]["loaded"]["status"] == sc.PROVEN
    assert entry["links"]["loaded"]["strength"] == "observed"
    assert "interleaved" in entry["links"]["loaded"]["evidence"]
    # And the same measurement supplies cost, as `partial` rather than proven: the number
    # is observed, the projection over a corpus is not.
    cost = entry["links"]["cost"]
    assert cost["status"] == sc.PARTIAL and cost["value"] == 450


def test_unstable_prefixes_never_supply_loading_proof_or_a_cost_floor():
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "AGENTS.md"
        path.write_text("project rules\n")
        measurement = {"file": str(path), "harness": "codex", "ok": True,
                       "loaded": "proven", "loaded_evidence": "fixture difference",
                       "mean_tokens": 700, "differences": [400, 1000],
                       "pair_spread_tokens": 600}
        for refusal in ({"reproducible": False},
                        {"measured_under": {"mixed": {"models": ["a", "b"]}}}):
            reading = {**measurement, **refusal}
            scan = _instruction_scan(path)
            scan["files"][0]["kind"] = "agents_md"
            components = sc._instruction_components(
                scan, {}, {"instruction_measurements": [reading]})
            assert len(components) == 1
            for link in ("loaded", "cost"):
                assert components[0]["links"][link]["status"] == sc.UNPROVEN
                assert components[0]["links"][link]["blocker"] == "unstable_measurement"
                assert components[0]["links"][link]["value"] is None
            candidate = sc._candidate(components[0], "cost")
            assert candidate["achievable"] is False and candidate["estimated_usd"] is None
            assert "diagnose" in candidate["method"]
            skill_cost = sc._entity_cost("skills", str(path), {sc._resolved_dir(str(path)): reading})
            assert skill_cost["status"] == sc.UNPROVEN and skill_cost["value"] is None


def test_instruction_components_with_the_same_basename_are_distinguishable():
    """Real output had two rows both reading `AGENTS.md` — a user-scope file and
    `~/.codex/AGENTS.md` — with different evidence and nothing to tell them apart."""
    with _env(_trace_calling(["Read"]), {"s": {"command": "x"}}) as root:
        # Two files with the identical basename, in different scopes.
        (root / "AGENTS.md").write_text("project rules\n", encoding="utf-8")
        nested = root / "packages" / "api"
        nested.mkdir(parents=True)
        (nested / "AGENTS.md").write_text("nested rules\n", encoding="utf-8")
        r = score_root(str(root))
    names = [c["name"] for c in r["components"] if c["kind"] in ("claude_md", "agents_md")]
    assert len(names) >= 2 and len(names) == len(set(names)), names


# -- stack-level cost: answerable without per-component attribution ----------------------


def _costed(name, harness, value, unit="tokens_per_request", status=None, scope="user"):
    c = _fully_evidenced(name=name)
    c["harness"] = harness
    c["harnesses"] = [harness]
    # User scope by default: these fixtures stand for always-on blocks, which is what the
    # floor is about. A project-scoped block is only in the prefix inside its own project,
    # so it is excluded from the floor and tested separately.
    c["scope"] = scope
    c["links"]["cost"] = sc._link(status or sc.PARTIAL, evidence="x", strength="observed",
                                  value=value, unit=unit)
    return c


def test_stack_cost_never_sums_across_harnesses():
    """A Codex request does not carry Claude Code's CLAUDE.md; one total would describe a
    session nobody runs."""
    out = sc.stack_cost([_costed("a", "codex", 450), _costed("b", "claude_code", 500)])
    assert out["codex"]["per_request_tokens_floor"] == 450
    assert out["claude_code"]["per_request_tokens_floor"] == 500
    assert "all" not in out and "total" not in out


def test_the_floor_excludes_project_blocks_because_they_are_alternatives():
    """"Every request carries at least N" has to be true of every request. Adding two
    projects' instruction files made it true of *none*: a request is made in one working
    directory, so it never carries both. On a real environment this overstated the floor by
    a factor of three, and it was the report's headline number."""
    always = _costed("CLAUDE.md (~)", "claude_code", 709)
    one = _costed("CLAUDE.md (proj-a)", "claude_code", 1928, scope="project")
    two = _costed("CLAUDE.md (proj-b)", "claude_code", 1991, scope="project")
    for c, n in ((one, 983), (two, 0)):
        c["links"]["cost"]["requests_observed"] = n
    out = sc.stack_cost([always, one, two])["claude_code"]

    assert out["per_request_tokens_floor"] == 709
    assert [b["tokens"] for b in out["project_blocks"]] == [1991, 1928]
    # The worst case is the heaviest block that was actually carried. `proj-b` is heavier
    # but its project has no recorded request, so naming it would state a cost nobody paid.
    assert out["heaviest_project_block"] == "CLAUDE.md (proj-a)"
    assert out["in_heaviest_project"] == 709 + 1928
    assert any("alternatives" in a or "互斥" in a for a in out["assumptions"])


def test_a_floor_with_no_project_blocks_says_nothing_about_a_heaviest_project():
    """Most environments have none, and inventing a worst case from an empty list is how a
    report grows a number with nothing behind it."""
    out = sc.stack_cost([_costed("CLAUDE.md (~)", "claude_code", 709)])["claude_code"]
    assert out["project_blocks"] == []
    assert out["in_heaviest_project"] is None
    assert out["heaviest_project_block"] is None


def test_stack_cost_counts_a_name_installed_twice_once():
    """The aggregation trap, sixth occurrence — this time in the unit "components vs. what
    the harness lists".

    `second-skill` is installed under `~/.claude/skills` and `~/.agents/skills`. That is two
    components and one listed skill, so only one copy is ever in a prefix. Summing both
    inflated this floor by about 30%, and nothing about the output looked wrong.
    """
    a = _costed("second-skill", "claude_code", 143)
    b = _costed("second-skill", "claude_code", 141)
    a["kind"] = b["kind"] = "skills"
    a["path"] = "/root-a/skills/second-skill/SKILL.md"
    b["path"] = "/root-b/skills/second-skill/SKILL.md"
    out = sc.stack_cost([a, b, _costed("CLAUDE.md", "claude_code", 709)])["claude_code"]

    # The lower of the two, because this is a floor.
    assert out["per_request_tokens_floor"] == 141 + 709
    assert out["components_measured"] == 2
    duplicate = [c for c in out["contributions"] if c.get("duplicate_of")]
    assert len(duplicate) == 1 and duplicate[0]["tokens"] == 141
    assert duplicate[0]["duplicate_of"]["tokens"] == 143
    assert duplicate[0]["duplicate_of"]["path"] == "/root-a/skills/second-skill/SKILL.md"


def test_a_shared_skill_is_costed_under_the_harness_that_measured_it():
    """`shared` marks `~/.agents/skills`, which both harnesses read. It is not a harness, so
    a per-request floor for it describes a session nobody runs — and leaving it in its own
    bucket drops the figure from the one floor where it is really paid."""
    component = _costed("demo-skill", "shared", 160)
    component["kind"] = "skills"
    component["links"]["cost"]["harness"] = "claude_code"
    out = sc.stack_cost([component])

    assert "shared" not in out
    assert out["claude_code"]["per_request_tokens_floor"] == 160


def test_stack_cost_refuses_to_add_characters_to_tokens():
    """The Claude skill listing is recorded in characters because no tokenizer runs."""
    out = sc.stack_cost([
        _costed("tokens", "codex", 450),
        _costed("chars", "codex", 7114, unit="chars"),
    ])
    assert out["codex"]["per_request_tokens_floor"] == 450
    assert [n["name"] for n in out["codex"]["not_summable"]] == ["chars"]


def test_stack_cost_is_a_floor_and_says_what_is_missing():
    components = [_costed("measured", "codex", 450)]
    unmeasured = _fully_evidenced(name="unmeasured")
    unmeasured["harness"], unmeasured["harnesses"] = "codex", ["codex"]
    unmeasured["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    components.append(unmeasured)
    out = sc.stack_cost(components)["codex"]
    assert out["basis"] == "floor"
    assert "unmeasured" in out["unmeasured"]
    # The additivity of separate prefix blocks is an assumption and must be returned as one.
    assert any("假设" in a for a in out["assumptions"])


def test_a_shared_block_counts_toward_the_total_but_not_per_component_coverage():
    """One listing covers twenty skills: it is a block the harness pays for, not a
    component's cost, so it must not inflate the per-component measured fraction."""
    shared = {"harness": "codex", "skill_listing_tokens": [1800, 2100],
              "skills_covered": ["fourth-skill", "eighth-skill"]}
    skill = _fully_evidenced(name="fourth-skill")
    skill["harness"], skill["harnesses"] = "codex", ["codex"]
    skill["links"]["cost"] = sc._unproven("not measured", sc.NOT_MEASURED)
    out = sc.stack_cost([_costed("AGENTS.md", "codex", 450), skill], shared)["codex"]
    assert out["per_request_tokens_floor"] == 2250
    assert out["per_request_tokens_floor_high"] == 2550
    # The skill is covered by the shared block, so it is no longer listed as unmeasured...
    assert "fourth-skill" not in out["unmeasured"]
    # ...but it did not contribute a per-component measurement either.
    assert out["components_measured"] == 1


def test_stack_cost_withholds_when_nothing_is_in_tokens():
    out = sc.stack_cost([_costed("dollars", "claude_code", 1.0, unit=None)])["claude_code"]
    assert out["per_request_tokens_floor"] is None
    assert "tokens per request" in out["blocker"]


def test_a_measurement_that_joins_to_nothing_is_named_rather_than_dropped():
    """Real defect, on an artifact that had cost real money to produce.

    The measurement artifact kept beside this repo had been anonymized for quoting, which
    rewrote its home directory. Every one of its ten measurements — two instruction files,
    seven skills and one MCP server — then joined to nothing, and the report said "not
    measured" for all of them. Both halves matter: the orphan is named, and a measurement
    that does join is not reported as an orphan.
    """
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        here = root / "AGENTS.md"
        here.write_text("real", encoding="utf-8")
        scan = {"root": str(root), "files": [{"path": str(here)}], "mcp_servers": []}
        evaluation = {
            "root": "/elsewhere/project",
            "instruction_measurements": [{"file": "/elsewhere/AGENTS.md", "ok": True},
                                         {"file": str(here), "ok": True}],
            "skill_measurements": [{"file": "/elsewhere/.agents/skills/s/SKILL.md",
                                    "ok": True}],
            "evaluations": [{"server": "absent-server", "status": "measured"}],
        }
        orphans = sc.unmatched_measurements(scan, evaluation)

    subjects = [o["subject"] for o in orphans]
    # The file that is really there joined, so it is not an orphan.
    assert str(here) not in subjects
    assert sorted(subjects) == sorted(["/elsewhere/AGENTS.md",
                                       "/elsewhere/.agents/skills/s/SKILL.md",
                                       "absent-server"])
    kinds = {o["subject"]: o["kind"] for o in orphans}
    assert kinds["/elsewhere/.agents/skills/s/SKILL.md"] == "skill"
    assert kinds["absent-server"] == "mcp_server"
    reasons = {o["subject"]: o["reason"] for o in orphans}
    # Nothing is on disk at that path, and the reason says exactly that and nothing more.
    assert "本机这个路径上没有文件" == reasons["/elsewhere/AGENTS.md"]
    assert "名字" in reasons["absent-server"]


def test_an_orphan_that_is_on_disk_is_told_apart_from_one_that_is_not():
    """The two cases need different actions, so they must not share a reason.

    A path that exists means the measurement is sound and the scan was too narrow, which
    `--all` fixes for free. A path that does not exist means the artifact describes some
    other machine, and no re-scan will help. An earlier version blamed the artifact's
    recorded root for both, which is wrong in the ordinary case: artifacts are recorded
    from a project root while `--all` scans the whole home, so the roots legitimately
    differ almost every time.
    """
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        present = root / "unscanned" / "SKILL.md"
        present.parent.mkdir()
        present.write_text("real but not scanned", encoding="utf-8")
        scan = {"root": str(root), "files": [], "mcp_servers": []}
        evaluation = {"root": str(root), "skill_measurements": [
            {"file": str(present), "ok": True},
            {"file": str(root / "gone" / "SKILL.md"), "ok": True},
        ]}
        reasons = {o["subject"]: o["reason"]
                   for o in sc.unmatched_measurements(scan, evaluation)}

    assert "扩大扫描范围" in reasons[str(present)]
    assert "本机这个路径上没有文件" == reasons[str(root / "gone" / "SKILL.md")]


def test_a_matching_artifact_reports_no_unmatched_measurements():
    """The other half: this diagnostic must not cry wolf on a healthy artifact."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        (root / "AGENTS.md").write_text("real", encoding="utf-8")
        scan = {"root": str(root), "files": [{"path": str(root / "AGENTS.md")}],
                "mcp_servers": [{"name": "present-server"}]}
        evaluation = {
            "root": str(root),
            "instruction_measurements": [{"file": str(root / "AGENTS.md"), "ok": True}],
            "evaluations": [{"server": "present-server", "status": "measured"}],
        }
        useful = {"root": str(root),
                  "useful_measurements": [{"file": str(root / "AGENTS.md"),
                                           "verdict": "proven"}]}
        assert sc.unmatched_measurements(scan, evaluation, useful) == []


def _carried(harnesses, measured_under, rate=100):
    """A component whose cost is a per-request rate, ready for the volume multiply."""
    c = _fully_evidenced(name="demo")
    c["harness"] = "+".join(harnesses)
    c["harnesses"] = list(harnesses)
    c["links"]["cost"] = sc._link(sc.PARTIAL, evidence=f"{rate} tokens per request",
                                  strength="observed", value=rate,
                                  unit="tokens_per_request")
    c["links"]["cost"]["harness"] = measured_under
    return c


def test_a_per_request_rate_becomes_a_total_using_the_observed_request_count():
    """The withheld reason named a request volume as what was missing, and it was free.

    One model response is one request, and both harnesses' counts are already in hand
    where the chain is built, so leaving the rate unmultiplied asked the user to buy a
    measurement that had been taken.
    """
    c = _carried(["claude_code"], "claude_code", rate=709)
    sc._attach_carried_tokens([c], {"observed": {"assistant_messages": 16731}},
                              {"requests": 8377})
    cost = c["links"]["cost"]
    assert cost["requests_observed"] == 16731
    assert cost["tokens_carried"] == 709 * 16731
    assert cost.get("carried_is_lower_bound") is None


def test_a_rate_is_multiplied_by_its_own_harnesss_traffic_only():
    c = _carried(["codex"], "codex", rate=450)
    sc._attach_carried_tokens([c], {"observed": {"assistant_messages": 16731}},
                              {"requests": 8377})
    assert c["links"]["cost"]["tokens_carried"] == 450 * 8377


def test_a_component_on_two_harnesses_measured_on_one_is_marked_a_lower_bound():
    """The real case: `search-server` is loaded by Codex too, at an unmeasured rate.

    Without the mark the product reads as a total, which is the aggregation mistake this
    project keeps making.
    """
    c = _carried(["claude_code", "codex"], "claude_code", rate=470)
    sc._attach_carried_tokens([c], {"observed": {"assistant_messages": 1000}},
                              {"requests": 500})
    cost = c["links"]["cost"]
    assert cost["tokens_carried"] == 470 * 1000
    assert cost["carried_is_lower_bound"] is True
    assert cost["carried_excludes"] == ["codex"]


def _project_carried(path, rate=1928, harness="claude_code"):
    c = _carried([harness], harness, rate=rate)
    c["scope"] = "project"
    c["path"] = path
    return c


def _collect_by_project(by_project, total=20000):
    """A whole-environment collect's shape: a total, and the per-project split under it."""
    return {"observed": {"assistant_messages": total,
                         "by_project": {f"p{i}": {"recorded_cwd": cwd,
                                                  "assistant_messages": n}
                                        for i, (cwd, n) in enumerate(by_project.items())}}}


def test_a_project_scoped_file_is_carried_by_its_own_projects_requests():
    """The environment total is the wrong multiplier for a file only in that project's
    prefix. Using it charged one project's CLAUDE.md to every request in the corpus."""
    c = _project_carried("/w/proj/CLAUDE.md")
    sc._attach_carried_tokens(
        [c], _collect_by_project({"/w/proj": 983, "/w/other": 15000}), {"requests": 8377})
    cost = c["links"]["cost"]
    assert cost["requests_observed"] == 983
    assert cost["tokens_carried"] == 1928 * 983
    assert cost["requests_scope"] == "project"


def test_a_project_file_counts_requests_from_its_subdirectories_too():
    """A session started below the project root records that subdirectory, so an exact
    match would report zero for a project worked in every day."""
    c = _project_carried("/w/proj/CLAUDE.md")
    sc._attach_carried_tokens(
        [c], _collect_by_project({"/w/proj": 10, "/w/proj/src": 90, "/w/project-two": 500}),
        {"requests": 0})
    # `/w/project-two` must not be swallowed: the containment test is on path parts, so a
    # string prefix match would have added its 500.
    assert c["links"]["cost"]["requests_observed"] == 100


def test_a_project_with_no_requests_says_so_instead_of_reporting_a_small_number():
    """The real case that exposed this: the largest `carried` figure in a report belonged to
    a project with no requests at all — a file never loaded once, printed as the heaviest
    thing in the stack. Zero is a finding, so it is stated rather than left as a blank."""
    c = _project_carried("/w/unused/CLAUDE.md", rate=1991)
    sc._attach_carried_tokens([c], _collect_by_project({"/w/proj": 983}), {"requests": 8377})
    cost = c["links"]["cost"]
    assert cost["requests_observed"] == 0
    assert cost["tokens_carried"] == 0
    assert "never" in cost["carried_is_zero_because"] or "没被付过" in cost["carried_is_zero_because"]


def test_a_user_scoped_file_still_uses_the_whole_environments_requests():
    """Scoping is per component: a user-scope file is in every session's prefix, so
    narrowing it to one project would under-report it by as much as the bug over-reported."""
    c = _carried(["claude_code"], "claude_code", rate=709)
    c["scope"] = "user"
    c["path"] = "/home/u/.claude/CLAUDE.md"
    sc._attach_carried_tokens([c], _collect_by_project({"/w/proj": 983}, total=17868), {"requests": 8377})
    assert c["links"]["cost"]["requests_observed"] == 17868


def test_a_same_named_project_elsewhere_is_reported_beside_the_figure_not_inside_it():
    """Transcripts are filed by working directory, so a moved project leaves its history
    under the old path. Attributing by the current path alone reports zero for a file
    thousands of requests carried — but a matching final path segment is a reason to ask,
    not proof two paths are one project, so it is never added in."""
    c = _project_carried("/w/new/proj/CLAUDE.md")
    sc._attach_carried_tokens(
        [c], _collect_by_project({"/w/new/proj": 983, "/w/old/proj": 6724, "/w/unrelated": 99}),
        {"requests": 8377})
    cost = c["links"]["cost"]
    # Not merged: the figure stays at what this path can prove.
    assert cost["tokens_carried"] == 1928 * 983
    assert cost["requests_elsewhere"] == [{"cwd": "/w/old/proj", "requests": 6724}]
    assert "/w/old/proj" in cost["requests_elsewhere_note"]
    # A directory that shares no name is not a candidate, or this would flag everything.
    assert all(e["cwd"] != "/w/unrelated" for e in cost["requests_elsewhere"])


def test_a_projects_own_subdirectories_are_not_reported_as_somewhere_else():
    """They are already inside the figure, so listing them would double the reader's count
    of the same requests."""
    c = _project_carried("/w/proj/CLAUDE.md")
    sc._attach_carried_tokens(
        [c], _collect_by_project({"/w/proj": 10, "/w/proj/nested/proj": 40}), {"requests": 0})
    cost = c["links"]["cost"]
    assert cost["requests_observed"] == 50
    assert cost.get("requests_elsewhere") is None


def test_shared_is_expanded_to_the_harnesses_that_actually_carry_it():
    """`shared` is `~/.agents`, not a harness with traffic of its own.

    Reporting "excludes shared" would be meaningless; what is excluded is Codex's
    requests, and a reader has to be told which traffic the figure leaves out.
    """
    c = _carried(["shared"], "claude_code", rate=160)
    sc._attach_carried_tokens([c], {"observed": {"assistant_messages": 1000}},
                              {"requests": 500})
    assert c["links"]["cost"]["carried_excludes"] == ["codex"]


def test_a_rate_with_no_request_count_for_its_harness_is_withheld_not_guessed():
    c = _carried(["unknown_harness"], "unknown_harness", rate=99)
    sc._attach_carried_tokens([c], {"observed": {"assistant_messages": 1000}},
                              {"requests": 500})
    cost = c["links"]["cost"]
    assert cost["tokens_carried"] is None
    assert "unknown_harness" in cost["carried_blocker"]


def test_the_blocker_asks_for_a_price_once_the_volume_is_counted():
    """Three states, not two. Saying "spend is not measured" after the volume is in hand
    would send someone to buy what they already have."""
    rate_only = sc._link(sc.PARTIAL, evidence="x", value=709, unit="tokens_per_request")
    counted = dict(rate_only, tokens_carried=11_000_000)
    proven_useful = sc._link(sc.PROVEN, evidence="y")
    assert "request volume" not in sc._withheld_because(proven_useful, counted)
    assert "price per token" in sc._withheld_because(proven_useful, counted)
    assert "spend is not" in sc._withheld_because(proven_useful, rate_only)


def _useful_artifact(skill_md, *, verdict="proven", rules=("subject_ends_with_reference",)):
    return {"useful_measurements": [{
        "file": str(skill_md), "useful": verdict, "rules_proven": list(rules),
        "useful_evidence": "with the skill the title carries the id; without it, not",
    }]}


def test_a_compliance_measurement_proves_useful_for_a_skill():
    """The link the probe can close. Keyed through the same index the cost artifact uses, so
    a measurement that was paid for does not read as never attempted."""
    with _env(_trace_calling(["Read"]), {}) as root:
        _with_skill(root, "demo")
        skill_md = root / ".claude" / "skills" / "demo" / "SKILL.md"
        chain = build_chain(scan=_scan(root), traces=_traces(root),
                            useful=_useful_artifact(skill_md))
        skill = next(c for c in chain if c["kind"] == "skills" and c["name"] == "demo")
        assert skill["links"]["useful"]["status"] == sc.PROVEN
        assert skill["links"]["useful"]["rules_proven"] == ["subject_ends_with_reference"]


def test_a_probe_that_did_not_separate_is_not_partial_credit():
    with _env(_trace_calling(["Read"]), {}) as root:
        _with_skill(root, "demo")
        skill_md = root / ".claude" / "skills" / "demo" / "SKILL.md"
        artifact = _useful_artifact(skill_md, verdict="unproven")
        artifact["useful_measurements"][0]["reason"] = "the arms did not separate"
        chain = build_chain(scan=_scan(root), traces=_traces(root), useful=artifact)
        skill = next(c for c in chain if c["kind"] == "skills" and c["name"] == "demo")
        assert skill["links"]["useful"]["status"] == sc.UNPROVEN
        assert "did not separate" in skill["links"]["useful"]["evidence"]


def test_proving_a_skill_useful_does_not_also_claim_the_user_used_it():
    """An instruction file is always-on, so a separation is evidence of use as well. A skill
    enters context on a description match, so the probe proves it *can* change the output.
    Writing the probe's own arm into `used` would turn this tool's run into evidence about
    the user's work."""
    with _env(_trace_calling(["Read"]), {}) as root:
        _with_skill(root, "demo")
        skill_md = root / ".claude" / "skills" / "demo" / "SKILL.md"
        chain = build_chain(scan=_scan(root), traces=_traces(root),
                            useful=_useful_artifact(skill_md))
        skill = next(c for c in chain if c["kind"] == "skills" and c["name"] == "demo")
        assert skill["links"]["useful"]["status"] == sc.PROVEN
        assert skill["links"]["used"]["status"] != sc.PROVEN


# -- can `useful` be proven at all, and if not, why ---------------------------------------
#
# Every assertion here is decided from a file on disk. The point of the split is that
# `unproven` alone reads as unfinished work, and a reader acting on that spends money on a
# component no method here can measure.


def _unproven_component(kind, name, path=None):
    links = {link: sc._unproven("n/a", sc.NO_EVENT_EMITTED)
             for link in ("installed", "loaded", "used", "useful", "cost", "recommendation")}
    out = {"kind": kind, "name": name, "harness": "claude_code",
           "harnesses": ["claude_code"], "scope": "user", "links": links}
    if path is not None:
        out["path"] = str(path)
    return out


def test_a_server_with_no_text_cannot_state_a_rule():
    """A server is declared in config and ships no prose, so there is no sentence to check."""
    m = sc.useful_measurability(_unproven_component("mcp", "s"))
    assert m["status"] == "not_measurable"
    assert m["reason"] == sc.NO_TEXT


def test_an_empty_file_is_told_apart_from_one_that_states_no_rule():
    """Both are unmeasurable, for different reasons, and the reasons suggest different acts."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "AGENTS.md"
        empty.write_text("", encoding="utf-8")
        prose = Path(tmp) / "CLAUDE.md"
        prose.write_text("Prefer small commits. Ask before refactoring.", encoding="utf-8")
        assert sc.useful_measurability(
            _unproven_component("agents_md", "a", empty))["reason"] == sc.EMPTY_TEXT
        assert sc.useful_measurability(
            _unproven_component("claude_md", "c", prose))["reason"] == sc.NO_RULE


def test_a_cli_flag_in_an_example_is_not_a_stated_rule():
    """The exact false positives a hand-rolled pattern set produced before this used the
    probe's own library: a sentence about what a *tool returns*, and a flag inside a
    worked example. Either one would have put a paid probe behind a skill that states
    nothing about its own output."""
    with tempfile.TemporaryDirectory() as tmp:
        skill = Path(tmp) / "SKILL.md"
        skill.write_text("Run `node bin/render.mjs check out.html --json`.\n"
                         "The reply is JSON text (`{code:0, data:...}`).\n", encoding="utf-8")
        m = sc.useful_measurability(_unproven_component("skills", "k", skill))
        assert m["status"] == "not_measurable"
        assert m["reason"] == sc.NO_RULE


def test_a_skill_stating_a_checkable_rule_becomes_measurable_and_affordable():
    """The arm for one skill is verified, so a skill that quotes a rule is in the same
    position as an instruction file — and the suggested step must say so."""
    with tempfile.TemporaryDirectory() as tmp:
        skill = Path(tmp) / "SKILL.md"
        skill.write_text("Put the issue reference at the end of the subject.\n",
                         encoding="utf-8")
        component = _unproven_component("skills", "k", skill)
        m = sc.useful_measurability(component)
        assert m["status"] == "measurable" and m["rules"] == ["subject_ends_with_reference"]
        step = sc._candidate(component, "useful")
        assert step["achievable"] is True
        assert "--plugin-dir" in step["method"]


def test_divergent_copies_make_a_result_about_the_copy_that_was_injected():
    """One name, two installs, different rules. The harness lists the name once and does not
    record which copy it loaded, so the caveat has to travel with the measurability."""
    with tempfile.TemporaryDirectory() as tmp:
        rich = Path(tmp) / "a" / "SKILL.md"
        rich.parent.mkdir()
        rich.write_text("Reference at the end of the subject; subject within 50 characters.\n",
                        encoding="utf-8")
        thin = Path(tmp) / "b" / "SKILL.md"
        thin.parent.mkdir()
        thin.write_text("Put the issue reference at the end of the subject.\n", encoding="utf-8")
        one = _unproven_component("skills", "k", rich)
        two = _unproven_component("skills", "k", thin)
        m = sc.useful_measurability(one, [one, two])
        assert m["status"] == "measurable"
        assert m["result_is_about_injected_copy"] is True and m["copies"] == 2


def test_identical_copies_raise_no_caveat():
    """Otherwise every duplicated install would carry a warning that does not apply."""
    with tempfile.TemporaryDirectory() as tmp:
        rule = "Put the issue reference at the end of the subject.\n"
        first = Path(tmp) / "a" / "SKILL.md"
        first.parent.mkdir()
        first.write_text(rule, encoding="utf-8")
        second = Path(tmp) / "b" / "SKILL.md"
        second.parent.mkdir()
        second.write_text(rule, encoding="utf-8")
        one = _unproven_component("skills", "k", first)
        two = _unproven_component("skills", "k", second)
        m = sc.useful_measurability(one, [one, two])
        assert m["status"] == "measurable"
        assert "result_is_about_injected_copy" not in m


def test_a_proven_link_is_not_relabelled_as_something_still_to_measure():
    m = sc.useful_measurability(_fully_evidenced("done"))
    assert m["status"] == "proven" and m["reason"] is None


def test_measurability_totals_account_for_every_component():
    """A split that loses a component would understate one of the two groups."""
    with tempfile.TemporaryDirectory() as tmp:
        skill = Path(tmp) / "SKILL.md"
        skill.write_text("Put the issue reference at the end of the subject.\n", encoding="utf-8")
        components = [_unproven_component("mcp", "s"),
                      _unproven_component("skills", "k", skill),
                      _fully_evidenced("done")]
        for component in components:
            component["useful_measurability"] = sc.useful_measurability(component, components)
        totals = sc._measurability_totals(components)
        assert totals["proven"] + totals["measurable"] + totals["not_measurable"] == 3
        assert totals["reasons"] == {sc.NO_TEXT: 1}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} score regression tests passed (0 tokens spent)")


if __name__ == "__main__":
    main()

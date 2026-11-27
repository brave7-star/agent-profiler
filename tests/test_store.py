"""Regression suite for the measurement store.

Measurement is the only thing in this project that costs money, and it was the only thing
with nowhere to live: the result went to stdout, and a `report` run that did not name the
artifact showed **no cost column at all**. Producing a report with costs meant hand-merging
four separate files, a step no reader should have to reinvent.

So the properties tested here are about not losing paid work, and about never passing a
remembered figure off as a fresh one.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import store as st


@contextmanager
def _store():
    """A temporary store, so no test reads or writes the developer's own measurements."""
    with tempfile.TemporaryDirectory() as raw:
        saved = os.environ.get("AGENT_PROFILER_STORE")
        os.environ["AGENT_PROFILER_STORE"] = raw
        try:
            yield Path(raw)
        finally:
            if saved is None:
                os.environ.pop("AGENT_PROFILER_STORE", None)
            else:
                os.environ["AGENT_PROFILER_STORE"] = saved


def _measured(path="/w/CLAUDE.md", tokens=709):
    return {"mode": "evaluate", "instruction_measurements": [
        {"ok": True, "file": path, "tokens": tokens, "loaded": "proven"}]}


def test_a_measurement_is_kept_and_read_back():
    with _store():
        kept = st.remember(_measured(), stage="measure-instructions")
        assert kept is not None and kept.is_file()
        out = st.stored()
    assert out["instruction_measurements"][0]["tokens"] == 709
    assert out["from_store"] is True
    assert out["sources"] == [kept.name]


def test_an_empty_store_is_nothing_rather_than_an_empty_measurement():
    """`None` means "nobody has measured anything", which is what makes the report say the
    cost column is unmeasured instead of showing zeros as though they were readings."""
    with _store():
        assert st.stored() is None


def test_a_run_that_measured_nothing_is_not_kept():
    """A failed or skipped probe produces an artifact with no measurements in it. Keeping
    those would fill the store with files that contribute nothing and make `sources` a
    misleading count of what the figures rest on."""
    with _store() as directory:
        assert st.remember({"mode": "evaluate", "skipped": ["x"]},
                           stage="measure-instructions") is None
        assert list(directory.glob("*.json")) == []


def test_two_measurements_of_one_file_are_both_kept():
    """Two readings that disagree are a finding this project already reports. Collapsing
    them here — by overwriting, or by keeping the newest — would hide the disagreement
    before anything had a chance to notice it."""
    with _store():
        st.remember(_measured(tokens=380), stage="measure-instructions")
        st.remember(_measured(tokens=474), stage="measure-instructions")
        out = st.stored()
    assert sorted(m["tokens"] for m in out["instruction_measurements"]) == [380, 474]


def test_every_measurement_kind_survives_the_round_trip():
    """Three stages write three different keys, and an earlier report needed all three
    hand-merged into one file. Dropping any one silently empties part of the cost column."""
    with _store():
        st.remember({"mode": "evaluate", "instruction_measurements": [{"file": "a"}]},
                    stage="measure-instructions")
        st.remember({"mode": "evaluate", "skill_measurements": [{"name": "b"}]},
                    stage="measure-skills")
        st.remember({"mode": "evaluate", "evaluations": [{"name": "c"}]},
                    stage="evaluate")
        out = st.stored()
    assert len(out["instruction_measurements"]) == 1
    assert len(out["skill_measurements"]) == 1
    assert len(out["evaluations"]) == 1
    assert len(out["sources"]) == 3


def test_a_corrupt_entry_is_named_rather_than_failing_the_run():
    """`report` is free and read-only, and must not be taken down by one bad file. The name
    is reported so the entry can be removed rather than silently ignored forever."""
    with _store() as directory:
        st.remember(_measured(), stage="measure-instructions")
        (directory / "99999999T000000-broken-0.json").write_text("{not json",
                                                                 encoding="utf-8")
        out = st.stored()
    assert out["unreadable"] == ["99999999T000000-broken-0.json"]
    assert len(out["instruction_measurements"]) == 1


def test_a_store_holding_only_a_corrupt_entry_still_says_so():
    """Returning `None` here would read as "nothing was ever measured", when the truth is
    that something was and cannot be read."""
    with _store() as directory:
        (directory / "99999999T000000-broken-0.json").write_text("[]", encoding="utf-8")
        out = st.stored()
    assert out is not None and out["sources"] == []
    assert out["unreadable"] == ["99999999T000000-broken-0.json"]


def test_an_unwritable_store_does_not_fail_a_paid_measurement():
    """The measurement has already been paid for and printed. Losing the copy is bad; losing
    the run because the copy could not be written is worse."""
    with tempfile.TemporaryDirectory() as raw:
        blocked = Path(raw) / "file-not-a-directory"
        blocked.write_text("", encoding="utf-8")
        saved = os.environ.get("AGENT_PROFILER_STORE")
        os.environ["AGENT_PROFILER_STORE"] = str(blocked / "measurements")
        try:
            assert st.remember(_measured(), stage="measure-instructions") is None
            assert st.stored() is None
        finally:
            if saved is None:
                os.environ.pop("AGENT_PROFILER_STORE", None)
            else:
                os.environ["AGENT_PROFILER_STORE"] = saved


def test_a_gate_refusal_is_not_stored_as_a_measurement():
    """`evaluate` emits a row per candidate whatever happens, including ones its spend gate
    refused and servers excluded by request. Those rows exist to say why nothing was
    measured. Storing them filled the store with entries contributing nothing, so a report
    claiming "cost figures come from N measurements" named more evidence than it had."""
    refused = {"mode": "evaluate", "evaluations": [
        {"server": "s", "status": "not_measured",
         "gate": {"decision": "refused", "reason": "below threshold"}}]}
    with _store() as directory:
        assert st.remember(refused, stage="evaluate") is None
        assert list(directory.glob("*.json")) == []
        assert st.stored() is None


def test_a_refusal_alongside_a_real_measurement_keeps_only_the_measurement():
    """One artifact carries both, so the filter has to be per row rather than per file."""
    mixed = {"mode": "evaluate", "evaluations": [
        {"server": "measured-one", "status": "measured", "tokens": 382},
        {"server": "refused-one", "status": "not_measured"},
        {"server": "excluded-one", "status": "excluded_by_request"}]}
    with _store():
        assert st.remember(mixed, stage="evaluate") is not None
        out = st.stored()
    assert [e["server"] for e in out["evaluations"]] == ["measured-one"]


def test_a_failed_probe_is_not_stored_as_a_reading():
    """Failures are reported rather than omitted, but storing one would let it read
    downstream as a measurement that happened."""
    with _store():
        assert st.remember({"mode": "evaluate", "instruction_measurements": [
            {"ok": False, "file": "/w/CLAUDE.md", "error": "timeout"}]},
            stage="measure-instructions") is None


def test_a_completed_null_result_survives_storage_and_does_not_offer_the_same_probe():
    from agent_profiler import score as sc

    with _store() as directory:
        path = directory / "AGENTS.md"
        path.write_text("- 标题末尾带议题编号。\n", encoding="utf-8")
        reading = {"ok": False, "file": str(path), "useful": "unproven",
                   "blocker": "NO_SEPARATION", "reason": "no difference on the tested tasks",
                   "negative_control": {"held": True}, "probes_attempted": 12,
                   "probes_failed": 0}
        kept = st.remember({"useful_measurements": [reading]}, stage="measure-useful")
        assert kept is not None
        restored = st.stored(keys=st.USEFUL_KEYS)
        assert restored["useful_measurements"] == [reading]
        scan = {"files": [{"kind": "agents_md", "origin": "declared",
                           "scope": "user", "path": str(path)}]}
        component = sc._instruction_components(scan, {}, None, restored)[0]
        assert component["links"]["useful"]["blocker"] == "NO_SEPARATION"
        assert component["links"]["useful"]["status"] == sc.UNPROVEN
        assert sc._candidate(component, "useful")["achievable"] is False
        component["links"]["useful"]["blocker"] = sc.NOT_MEASURED
        assert sc._candidate(component, "useful")["achievable"] is True, \
            "before the experiment this fixture must offer a real probe"
        skill_link = sc._entity_useful("skills", str(path), {sc._resolved_dir(str(path)): reading})
        assert skill_link["blocker"] == "NO_SEPARATION"
        assert skill_link["status"] == sc.UNPROVEN
        for invalid in ({"negative_control": {"held": False}},
                        {"abandoned": "fixture outage"}, {"probes_failed": 12}):
            assert st.remember({"useful_measurements": [{**reading, **invalid}]},
                               stage="measure-useful") is None


def test_the_store_follows_a_redirected_environment():
    """A measurement describes one environment, so it must not be filed under another's.

    Keying the store on the real home regardless meant a run pointed at a redirected
    environment wrote into the real one's history — measurements of a sandbox filed as
    measurements of the user's own stack, which every later report would read as its own.
    This project's own suite demonstrated it by writing fixture rows into a real state
    directory, and a reader profiling a copied environment would hit the same thing.
    """
    saved = {k: os.environ.get(k) for k in
             ("AGENT_PROFILER_STORE", "AGENT_PROFILER_HOME")}
    os.environ.pop("AGENT_PROFILER_STORE", None)
    with tempfile.TemporaryDirectory() as raw:
        os.environ["AGENT_PROFILER_HOME"] = raw
        try:
            assert Path(raw) in st.store_dir().parents
            # And an explicit store still wins over the redirect, because a reader who names
            # a location means it.
            os.environ["AGENT_PROFILER_STORE"] = f"{raw}/elsewhere"
            assert st.store_dir() == Path(raw) / "elsewhere"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_the_store_is_plain_json_anyone_can_read():
    """A measurement that cannot be read without this tool is one someone will pay for
    twice."""
    with _store():
        kept = st.remember(_measured(), stage="measure-instructions")
        data = json.loads(kept.read_text(encoding="utf-8"))
    assert data["instruction_measurements"][0]["file"] == "/w/CLAUDE.md"


def test_the_merge_order_is_reproducible():
    """Two runs over the same store must produce the same artifact, or a figure can change
    between reports with nothing having been measured."""
    with _store() as directory:
        for i in range(3):
            (directory / f"2026112{i}T100000-x-{i}.json").write_text(
                json.dumps(_measured(path=f"/w/{i}.md")), encoding="utf-8")
        first, second = st.stored(), st.stored()
    assert first == second
    assert [m["file"] for m in first["instruction_measurements"]] == [
        "/w/0.md", "/w/1.md", "/w/2.md"]


def test_a_report_says_when_its_cost_figures_were_remembered():
    """A report that silently used measurements from weeks ago reads as though it measured
    something this run."""
    from agent_profiler import report as rp

    stored = {"from_store": True, "store": "/w/store", "sources": ["a.json", "b.json"]}
    text = "\n".join(rp._render({
        "root": "/w", "generated_at": "2026-01-01 00:00",
        "measurement_source": stored,
        **_RENDER_STUB}).splitlines())
    assert "/w/store" in text
    assert "2" in text


# The smallest `data` `_render` accepts, so the provenance line can be rendered without
# building a whole report.
_RENDER_STUB = {
    "observed": {"installed": {"files": 0, "bytes": 0, "by_kind": {}, "entities": {},
                               "entity_files": {}, "bundled_files": 0},
                 "sessions": 0,
                 "link_coverage": {k: {"proven": 0, "partial": 0, "unproven": 0}
                                   for k in ("installed", "loaded", "used", "useful",
                                             "cost")},
                 "components_total": 0,
                 "components_scorable": 0, "tokens_spent": 0},
    "components": [], "unmatched_measurements": [], "relocation_candidates": [],
    "stack_score": {"withheld": True, "value": None, "reason": "x"}, "sessions": 0,
    "always_on": [], "diverged_duplicates": [], "reliability": [], "verdicts": [],
    "profile": "balanced", "cost": [], "carried": [], "stack_cost": {},
    "unanswerable": [], "blind_spots": [], "blind_spots_not_applicable": [],
    "next_steps": [], "headline": "x", "findings": [], "standing": {},
    "measurement_conflicts": [],
    "profiler_cost": {"stage": "report", "llm_input_tokens": 0, "llm_output_tokens": 0,
                      "cost_usd": 0.0, "elapsed_ms": 0.0},
}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} measurement-store tests passed (0 tokens spent)")


if __name__ == "__main__":
    raise SystemExit(main())

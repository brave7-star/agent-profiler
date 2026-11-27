"""Where paid measurements are kept, so money spent accumulates.

Every other stage here is free and re-runnable. Measurement is not: `measure-instructions`
and `measure-skills` issue real requests against a real harness, and the result is the only
thing in this project that costs money to produce.

It was also the only thing with nowhere to live. A measurement went to stdout, and feeding
it back meant remembering `--output`, remembering where the file went, and passing
`--evaluation` on every later run. A report run without that flag showed **no cost column at
all** — the most expensive output in the project, invisible by default, and recoverable only
by whoever still had the path. Four separate artifacts had to be hand-merged into one file to
produce a report with a cost column, which is a step no reader of this tool should have to
reinvent.

So a measurement is written here as well as to stdout, and `report` reads this directory
unless told otherwise. The design follows from what the data is:

* **Append-only, one file per saved artifact.** A measurement is an observation with a timestamp, not
  a current value. Overwriting would discard the earlier reading, and two readings that
  disagree are a *finding* this project already reports rather than a conflict to resolve by
  recency.
* **Keyed by nothing.** Merging is by file path and harness, which the scoring stage already
  does for supplied artifacts; the store does not duplicate that logic, it only preserves
  what was measured.
* **Never read as authority over an explicit flag.** `--evaluation` still wins, because a
  reader comparing a fresh measurement against the stored one needs the fresh one to be what
  they get.
* **Plain JSON on disk, under a documented path.** A measurement that cannot be read without
  this tool is a measurement someone will pay for twice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# The merge keys scoring already understands. A stored run that carries none of these holds
# no measurement, so it contributes nothing and is not an error either.
COST_KEYS = ("instruction_measurements", "skill_measurements", "evaluations")
# `measure-useful` is paid too — the dearest of the probes per component — and it answers a
# different link, so it is read back through `--useful` rather than `--evaluation`. Storing
# only the cost artifacts left the most expensive measurement in the project with nowhere to
# live, and a report run without the flag showed no verdicts at all, because `useful` is one
# of the three links a verdict requires.
# One key, because that is the one the scoring stage reads. Listing a second plausible name
# here would be a key nothing ever consumes.
USEFUL_KEYS = ("useful_measurements",)
MEASUREMENT_KEYS = COST_KEYS + USEFUL_KEYS

STORE_SCHEMA = "1"


def _is_measurement(row: object) -> bool:
    """Whether a row records something that was actually measured.

    These lists are not all measurements. `evaluate` emits one row per candidate whatever
    happens, including rows for candidates its spend gate *refused* and for servers excluded
    by request — rows that exist to say why no measurement was taken. Storing those filled
    the store with entries carrying `status: not_measured`, so `sources` counted files that
    contributed nothing, and a report's "cost figures come from N measurements" named a
    number larger than the evidence behind it.
    """
    if not isinstance(row, dict):
        return False
    status = row.get("status")
    if isinstance(status, str) and status != "measured":
        return False
    # A completed compliance experiment can report ok=False because no effect separated
    # the arms. That is a measured outcome, unlike a probe that never answered.
    if (row.get("useful") == "unproven" and row.get("blocker") == "NO_SEPARATION"
            and (row.get("negative_control") or {}).get("held") is True
            and not row.get("abandoned")
            and (row.get("probes_attempted") or 0) > (row.get("probes_failed") or 0)):
        return True
    # A probe that failed is reported, never stored as a reading.
    return row.get("ok") is not False


def store_dir() -> Path:
    """Where measurements are kept.

    A measurement describes **one** environment, so the store has to belong to the same
    environment the run was pointed at. `AGENT_PROFILER_STORE` names it outright; otherwise,
    if the harness homes have been redirected, the store follows them.

    That second rule is not a test convenience, it is the correctness condition. Keying the
    store on the real home regardless meant a run against a redirected environment wrote its
    results into the real one's history — measurements of a sandbox filed as measurements of
    the user's actual stack, where every later `report` would read them as its own. It was
    the project's own suite that demonstrated it, writing fixture rows into a real state
    directory, but a reader profiling a copied environment would have hit the same thing.
    """
    override = os.environ.get("AGENT_PROFILER_STORE")
    if override:
        return Path(override).expanduser()
    # Whichever home was redirected; the Claude one first, since a redirected environment
    # usually sets both and they sit side by side.
    for name in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME",
                 "AGENT_PROFILER_CODEX_HOME"):
        redirected = os.environ.get(name)
        if redirected:
            return Path(redirected).expanduser() / "agent-profiler-measurements"
    return Path.home() / ".local" / "state" / "agent-profiler" / "measurements"


def remember(data: dict, *, stage: str) -> Path | None:
    """Keep a measurement artifact, returning where it went.

    Failure is reported as `None` rather than raised: the measurement has already been paid
    for and printed, and losing the copy is worse than losing the run. A read-only or absent
    state directory is the ordinary case on a locked-down machine.
    """
    if not any(any(_is_measurement(row) for row in (data.get(key) or []))
               for key in MEASUREMENT_KEYS):
        return None
    directory = store_dir()
    # The stage says what produced it and the stamp orders it. The suffix is found by
    # probing rather than taken from the clock or the pid: a second-resolution stamp plus a
    # pid is *not* unique — two measurements kept within one second of each other collided
    # and the second overwrote the first, losing a reading that had been paid for. Keeping
    # both matters because two readings of one file that disagree are a finding.
    stamp = time.strftime("%Y%m%dT%H%M%S")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for attempt in range(1000):
            path = directory / f"{stamp}-{stage}-{attempt:03d}.json"
            try:
                # Exclusive create, so a parallel run cannot win the same name between the
                # existence check and the write.
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=1)
                    handle.write("\n")
                return path
            except FileExistsError:
                continue
    except OSError:
        return None
    return None


def stored(directory: Path | None = None, *, keys: tuple = COST_KEYS) -> dict | None:
    """Every measurement kept so far, merged into one artifact, or None if there are none.

    `keys` selects which link's measurements to return, because the two artifacts are read
    back through different arguments: cost through `--evaluation`, usefulness through
    `--useful`. Merging both into one dict would hand each consumer the other's rows.

    The merge is a concatenation per key, deliberately: two measurements of one file are two
    observations, and the scoring stage reports a disagreement between them instead of
    picking one. Collapsing them here would hide exactly the thing worth seeing.

    Files are read oldest-first by name so the order is reproducible, and a file that cannot
    be parsed is skipped and named in `unreadable` rather than failing the run — a corrupt
    entry must not take a free report down with it.
    """
    directory = directory or store_dir()
    if not directory.is_dir():
        return None
    merged: dict = {key: [] for key in keys}
    sources: list[str] = []
    unreadable: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable.append(path.name)
            continue
        if not isinstance(data, dict):
            unreadable.append(path.name)
            continue
        found = False
        for key in keys:
            rows = [r for r in (data.get(key) or []) if _is_measurement(r)]
            if rows:
                merged[key].extend(rows)
                found = True
        if found:
            sources.append(path.name)
    if not sources:
        return {"unreadable": unreadable, "from_store": True, "sources": []} \
            if unreadable else None
    return {
        "schema_version": STORE_SCHEMA,
        "mode": "evaluate",
        # Says where this came from, so a report can tell the reader it used stored
        # measurements rather than appearing to have measured something this run.
        "from_store": True,
        "store": str(directory),
        "sources": sources,
        "unreadable": unreadable,
        **{key: rows for key, rows in merged.items() if rows},
    }

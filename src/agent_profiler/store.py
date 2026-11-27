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

* **Append-only, one file per run.** A measurement is an observation with a timestamp, not
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
MEASUREMENT_KEYS = ("instruction_measurements", "skill_measurements", "evaluations")

STORE_SCHEMA = "1"


def store_dir() -> Path:
    """Where measurements are kept.

    `AGENT_PROFILER_STORE` overrides it, which is what lets the tests run against a
    temporary directory instead of the developer's own history — the same discipline the
    harness homes already follow.
    """
    override = os.environ.get("AGENT_PROFILER_STORE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "state" / "agent-profiler" / "measurements"


def remember(data: dict, *, stage: str) -> Path | None:
    """Keep a measurement artifact, returning where it went.

    Failure is reported as `None` rather than raised: the measurement has already been paid
    for and printed, and losing the copy is worse than losing the run. A read-only or absent
    state directory is the ordinary case on a locked-down machine.
    """
    if not any(data.get(key) for key in MEASUREMENT_KEYS):
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


def stored(directory: Path | None = None) -> dict | None:
    """Every measurement kept so far, merged into one artifact, or None if there are none.

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
    merged: dict = {key: [] for key in MEASUREMENT_KEYS}
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
        for key in MEASUREMENT_KEYS:
            rows = data.get(key)
            if isinstance(rows, list) and rows:
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

"""Prefix differencing: what one component adds to every request.

Named for where it started — Codex instruction files — it now carries the same method for
Claude Code's instruction files and skills, because the method is one thing: build two arms
that differ by one component, alternate them inside one window, and read the difference in
the fixed prefix.

This module exists because a documented limitation turned out to be false. The audit
recorded "Codex instruction loading is unobservable — CONFIRMED,
`session_meta.base_instructions` is empty in every session examined". The field is indeed
empty. The instructions still enter the prefix, and differencing finds them:

    an instruction file  ->  the prefix grows by about the file's own token count,
                             reproduced to the token in interleaved pairs, in both orders

So this is a `loaded` proof that does not depend on any record existing. A difference
larger than noise means the file entered the context window; the size of the difference is
its recurring prefix cost.

## Why injection rather than removal

Both harnesses punish the removal design. Hiding configuration also hides authentication —
on this machine the auth token lives in a settings `env` block — so a "without X" arm built
by subtraction measures a harness that never started. Here the arm is built by *addition*:
two temp working directories, identical except that one carries a copy of the real
instruction file. The real file is read and never modified.

## Why pairs are interleaved

The equivalent Claude Code route was abandoned after an unpaired run suggested a large
positive cost per skill, while interleaved pairs of the *same* fixture came back slightly
negative. The baseline drifts between runs by hundreds of tokens as the project changes,
while within a tight window it is stable to ~10. Every measurement here therefore alternates
arms inside one window and reports each pair, so drift cannot be read as an effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .probe_cost import ProbeCost
from .core import _user_tree
from datetime import datetime, timezone

PROBE_PROMPT = "Say: X"
PROBE_PAIRS = 2
# Within one window the Codex prefix reproduced exactly, twice at each size, so
# this band is conservative rather than fitted.
MIN_DETECTABLE_TOKENS = 50
# A prefix probe carries ~14k tokens. Far above that is an agentic session, not a
# measurement — the same accident that a stdin leak caused on the Claude side.
CONTEXT_CEILING = 60_000
# Two arms of one pair differ by an instruction file — hundreds of tokens on a prefix of
# tens of thousands. Anything below this ratio means an arm failed to load, not that the
# file is enormous. This is a *relative* check on purpose: the absolute baseline varies by
# an order of magnitude between windows, so no fixed floor separates the two cases.
ARM_AGREEMENT = 0.5
# The auth/init failure class runs near 40% and clusters in time. Without a retry it does
# not merely waste a run: it kills the *pair*, and the surviving arm is still billed. Four
# of the first seven skill measurements lost a pair this way and each was left resting on a
# single observation. The delay matches the usefulness probe's, where it was calibrated.
ARM_RETRIES = 2
ARM_RETRY_DELAY = 5



def _now() -> str:
    """When this artifact was produced, in UTC.

    Added because two artifacts measuring the same file disagreed and neither carried a
    time, so the join could not show that one superseded the other — it silently kept
    whichever record came last. A figure without a date cannot be aged out.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _reproducible(diffs: list[int], spread: int) -> bool | None:
    """Whether the figure repeated, or `None` when only one pair survived.

    A single good pair has a spread of 0 by arithmetic, and reporting that as
    `reproducible: true` manufactures the strongest claim in the result out of one
    observation. Four of this project's first seven skill measurements lost a pair to the
    auth/init flake and every one of them asserted reproducibility.
    """
    if len(diffs) < 2:
        return None
    return spread <= MIN_DETECTABLE_TOKENS


def _measured_under(arms: list[dict]) -> dict:
    """The builds and models the arms of one measurement actually ran under.

    A list rather than a single value, because a build that changed mid-measurement would
    make the two arms incomparable — which is exactly the defect this field exists to catch,
    so it must be able to report it rather than silently keep the first value it saw.

    The serving route belongs here for the same reason, and it is the stronger of the two:
    a build and a model can be read off the config, while the route cannot. Arms answered by
    different routes carry token figures that were not produced by one accounting, so their
    difference measures the routes and not the file.
    """
    def distinct(key):
        return sorted({a[key] for a in arms if a.get(key)})

    versions, models = distinct("cli_version"), distinct("model")
    out = {"cli_version": versions[0] if len(versions) == 1 else None,
           "model": models[0] if len(models) == 1 else None}
    if len(versions) > 1 or len(models) > 1:
        out["mixed"] = {"cli_versions": versions, "models": models}
        out["warning"] = ("the arms did not all run under one build or model, so their "
                          "difference is not a clean measurement of the file")
    # Every arm must be on the route the harness asked for — not merely on the *same* route
    # as the other arm. Two replies that both carry a foreign id agree only in being
    # unidentified: nothing in the record says they came from the same place, so "they
    # match" is not a property that can be established about them. Requiring agreement
    # rather than provenance would have accepted exactly that pair.
    #
    # Collected as a list, not a set, because `False` is the observation being looked for
    # and any truthiness filter drops it.
    routes = [a["reply_carries_own_turn_id"] for a in arms
              if a.get("reply_carries_own_turn_id") is not None]
    # `None` means the build did not stamp turns in a way this can read, which is not the
    # same as having checked and found the route sound.
    out["arms_on_own_route"] = None if not routes else all(routes)
    return out


def _rollout_run(home: Path, since: float) -> dict:
    """The input-token total from the rollout this run just wrote, and what wrote it.

    The build and model are read here because a token count is only comparable to another
    one taken under the same tokenizer. Two runs of this probe over the *same* file, two
    weeks apart, returned different figures while each reported a pair spread of zero —
    reproducible within its own window and not across them. Neither artifact recorded a
    build or a model, so the two could not be reconciled at all, and the earlier figure had
    meanwhile been quoted elsewhere as settled.

    This does not explain that gap; it makes the next one answerable, which is the honest
    thing available. A missing value stays `None` rather than being guessed at: the rollout
    is the harness's own record, and inventing a build number would be worse than admitting
    the run did not say.

    The reply's own turn id is read for the same reason, one level further out. A CLI can be
    pointed at a relay rather than at the model provider, and a relay is free to answer two
    identical requests by two different routes. When it does, the usage figures stop being
    comparable — and the difference between two arms is *only* a measurement if both arms
    were served the same way. The signal is cheap and local: the harness stamps each turn
    with an id and the reply carries that id back, so a reply bearing an id the harness never
    generated was assembled by something in the middle. That is not a judgement about the
    relay; it is the reason two readings of one file need not agree.
    """
    newest, newest_mtime = None, since
    for path in home.rglob("rollout-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= newest_mtime:
            newest, newest_mtime = path, mtime
    if newest is None:
        return {"input_tokens": None, "cli_version": None, "model": None,
                "reply_carries_own_turn_id": None}
    total, version, model = None, None, None
    issued: set[str] = set()
    echoed: set[str] = set()
    for line in newest.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = record.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        # `session_meta` opens the file; `turn_context` carries the model per turn, which is
        # the one that actually priced the request.
        version = version or payload.get("cli_version")
        model = payload.get("model") or model
        kind = record.get("type") or payload.get("type")
        if kind == "turn_context" and isinstance(payload.get("turn_id"), str):
            issued.add(payload["turn_id"])
        if payload.get("role") == "assistant":
            meta = payload.get("internal_chat_message_metadata_passthrough") or {}
            if isinstance(meta, dict) and isinstance(meta.get("turn_id"), str):
                echoed.add(meta["turn_id"])
        info = payload.get("info") or {}
        if not isinstance(info, dict):
            continue
        usage = info.get("total_token_usage") or {}
        if isinstance(usage, dict) and usage.get("input_tokens"):
            total = int(usage["input_tokens"])
    # `None` when either half is absent: a build that does not stamp turns this way has not
    # told us the arms were served differently, and treating silence as a mismatch would
    # refuse every measurement on it.
    own = bool(echoed) and echoed <= issued if (issued and echoed) else None
    return {"input_tokens": total, "cli_version": version, "model": model,
            "reply_carries_own_turn_id": own}


def _codex_probe(home: Path, workdir: Path, timeout: int) -> dict:
    binary = shutil.which("codex")
    if binary is None:
        raise RuntimeError("the `codex` CLI is required for this measurement but was not found")
    started = time.time()
    env = dict(os.environ, CODEX_HOME=str(home))
    argv = [binary, "exec", "--skip-git-repo-check", "-s", "read-only", PROBE_PROMPT]
    try:
        # stdin closed for the same reason as the Claude probe: `codex exec` reads
        # instructions from stdin when stdin is piped, so an inherited stdin would make the
        # measurement depend on the caller's unconsumed input.
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=str(workdir), env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if done.returncode != 0:
        return {"ok": False, "error": f"exit {done.returncode}: {done.stderr.strip()[:160]}"}
    run = _rollout_run(home, started - 1)
    total = run["input_tokens"]
    if total is None:
        return {"ok": False, "error": "no rollout carrying token usage was written"}
    if total > CONTEXT_CEILING:
        return {"ok": False,
                "error": f"input tokens ({total}) exceed the {CONTEXT_CEILING} ceiling; "
                         "this probe did work instead of measuring a prefix"}
    return {"ok": True, "input_tokens": total,
            "cli_version": run["cli_version"], "model": run["model"],
            "reply_carries_own_turn_id": run["reply_carries_own_turn_id"]}


def _controlled_home(tmp: Path) -> Path:
    """A CODEX_HOME carrying only what a run needs: credentials and config.

    `AGENTS.md` is deliberately *not* copied, so the user-scope instruction file is absent
    from the baseline and can be injected as the treatment.
    """
    real = _user_tree("AGENT_PROFILER_CODEX_HOME", ".codex")
    home = tmp / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml"):
        source = real / name
        if source.is_file():
            shutil.copy2(source, home / name)
    return home


def measure_instruction_cost(instruction_file: Path, *, pairs: int = PROBE_PAIRS,
                             timeout: int = 420) -> dict:
    """Measure what one instruction file adds to every Codex request, by injection.

    Returns a `loaded` verdict as well as a cost: a difference beyond the noise band is
    direct evidence that the file's content entered the context window, which no Codex
    record reports.
    """
    if not instruction_file.is_file():
        return {"ok": False, "reason": f"not a file: {instruction_file}"}
    content = instruction_file.read_text(errors="replace")
    # Characters and bytes differ for non-ASCII instructions — this user's AGENTS.md is
    # a multi-byte file has more bytes than characters — and reporting one as the other
    # would misstate the
    # size the measured tokens correspond to.
    size_bytes = instruction_file.stat().st_size
    if not content.strip():
        # Guard against the invalid test that nearly became a finding: differencing an
        # empty file returned -3 tokens, which would have been reported as "not loaded".
        return {"ok": False, "reason": f"{instruction_file} is empty, so differencing it "
                                       "cannot show anything either way",
                "bytes": size_bytes, "characters": len(content)}

    measured_pairs: list[dict] = []
    measured_arms: list[dict] = []
    spend = ProbeCost()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        home = _controlled_home(tmp)
        without_dir = tmp / "without"
        with_dir = tmp / "with"
        for directory in (without_dir, with_dir):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "note.txt").write_text("x", encoding="utf-8")
        (with_dir / "AGENTS.md").write_text(content, encoding="utf-8")

        for index in range(max(1, pairs)):
            # Order alternates so that any monotonic drift cancels across pairs rather
            # than accumulating into the difference.
            order = (("with", with_dir), ("without", without_dir)) if index % 2 else \
                    (("without", without_dir), ("with", with_dir))
            arms: dict[str, dict] = {}
            for label, directory in order:
                # An arm answered by another route is retried rather than differenced. The
                # route is a *precondition* of the measurement, the way a working credential
                # is, so this has the shape of the flake retry and not of an outcome filter:
                # what is retried is a condition of the request, decided before any token
                # figure is read.
                #
                # Only the harness's own route is accepted, never merely "the two arms
                # agree". A reply carrying an id the harness never issued says it came from
                # somewhere else, and nothing in the record identifies two such replies as
                # having come from the *same* somewhere else. `None` is accepted because a
                # build that does not stamp replies this way has reported nothing either way.
                for attempt in range(ARM_RETRIES + 1):
                    arms[label] = self_probe = _codex_probe(home, directory, timeout)
                    spend.record(self_probe)
                    if not self_probe["ok"]:
                        break
                    if self_probe.get("reply_carries_own_turn_id") is not False:
                        break
                    if attempt < ARM_RETRIES:
                        time.sleep(ARM_RETRY_DELAY)
                if not self_probe["ok"]:
                    break
            measured_arms.extend(a for a in arms.values() if a.get("ok"))
            if all(arm["ok"] for arm in arms.values()) and len(arms) == 2:
                measured_pairs.append({
                    "pair": index + 1,
                    "order": [label for label, _ in order],
                    "with": arms["with"]["input_tokens"],
                    "without": arms["without"]["input_tokens"],
                    "difference": arms["with"]["input_tokens"] - arms["without"]["input_tokens"],
                })
            else:
                measured_pairs.append({
                    "pair": index + 1,
                    "order": [label for label, _ in order],
                    "error": next((a["error"] for a in arms.values() if not a["ok"]),
                                  "incomplete pair"),
                })

    good = [p for p in measured_pairs if "difference" in p]
    if not good:
        return {"ok": False, "reason": "no pair completed", "pairs": measured_pairs,
                **spend.summary()}

    diffs = [p["difference"] for p in good]
    mean = sum(diffs) / len(diffs)
    spread = max(diffs) - min(diffs)
    common = {
        "file": str(instruction_file),
        # Named on the measurement, not only in the artifact's header. One real environment
        # symlinks AGENTS.md to CLAUDE.md, so the same bytes are measured once per harness
        # and the two figures legitimately differ — and without this label a join cannot
        # tell that apart from one file measured twice with contradictory results.
        "harness": "codex",
        "bytes": size_bytes,
        "characters": len(content),
        "pairs": measured_pairs,
        "differences": diffs,
        "mean_tokens": round(mean),
        "pair_spread_tokens": spread,
        # What produced the figure, so two runs can be compared instead of merely differing.
        # A token count belongs to a tokenizer, and the build chooses the model.
        "measured_under": _measured_under(measured_arms),
        **spend.summary(),
        "method": "a copy of the file is injected as a project AGENTS.md in a controlled "
                  "CODEX_HOME; arms alternate within one window so drift cannot read as "
                  "an effect. The real file is read, never modified.",
        "comparability": "A token figure holds for the build and model named in "
                         "`measured_under`. `pair_spread_tokens` is agreement between arms "
                         "inside one window, which is not stability across runs: the same "
                         "file measured under a different build can differ while each run "
                         "still reports a spread of 0.",
    }
    # Checked before the noise band, because a mixed route invalidates the difference at
    # any magnitude: a large one is not evidence of loading and a small one is not evidence
    # of its absence. Repeating the measurement does not address it, so this is reported as
    # a condition to fix rather than as a figure with a caveat attached.
    if common["measured_under"].get("arms_on_own_route") is False:
        return {
            "ok": False,
            "blocker": "ARMS_NOT_SERVED_ALIKE",
            "loaded": "unproven",
            "reason": "the arms' replies did not all carry the turn id this harness issued, "
                      "so they were not all answered by the same route; a difference "
                      "between them measures the routes rather than the file. Each arm was "
                      f"retried up to {ARM_RETRIES} time(s) and the route did not settle, "
                      "so more pairs cannot fix this — point the CLI straight at the model "
                      "provider, or establish that whatever sits in front of it serves "
                      "every request alike.",
            **common,
        }
    if mean <= MIN_DETECTABLE_TOKENS:
        return {
            "ok": False,
            "inconclusive": True,
            "loaded": "unproven",
            "reason": f"the arms differ by {round(mean)} tokens on average, inside the "
                      f"{MIN_DETECTABLE_TOKENS}-token noise band, so neither a cost nor "
                      "loading was established",
            **common,
        }
    return {
        "ok": True,
        # This is the point of the module: a positive difference is direct evidence of
        # loading, from a harness that reports no such event.
        "loaded": "proven",
        "loaded_evidence": f"injecting this {size_bytes}-byte file moved the fixed prefix "
                           f"by {round(mean)} tokens across {len(good)} interleaved pair(s), "
                           f"spread {spread}; content that changes the prefix is content "
                           "that entered the context window",
        "reproducible": _reproducible(diffs, spread),
        **common,
    }


def measure_claude_instruction_cost(instruction_file: Path, *, pairs: int = PROBE_PAIRS,
                                    timeout: int = 420) -> dict:
    """Measure what one CLAUDE.md adds to every Claude Code request, by injection.

    This route was written off: "both routes to per-component Claude Code prefix cost
    failed". One of the two did fail — removing a file from a copied HOME hides the auth
    token with it. The other was abandoned for a different reason, recorded in this
    module's own docstring: an *unpaired* run suggested a large positive cost per skill and
    a repeat came back negative. That is baseline drift, and interleaved pairs fix it. They
    were adopted afterwards, for Codex, and never brought back here.

    The arms are the ones the usefulness probe already runs:

        without = --bare
        with    = --bare --append-system-prompt-file <verbatim copy>

    The channel caveat from that probe applies unchanged and travels with the result: the
    content is byte-identical to the real file but arrives as an appended system prompt, so
    this is the cost of the *content*, not of the harness's own CLAUDE.md plumbing, which
    may wrap it differently.
    """
    from .useful_probe import _probe

    if not instruction_file.is_file():
        return {"ok": False, "reason": f"not a file: {instruction_file}"}
    content = instruction_file.read_text(errors="replace")
    size_bytes = instruction_file.stat().st_size
    if not content.strip():
        return {"ok": False, "reason": f"{instruction_file} is empty, so differencing it "
                                       "cannot show anything either way",
                "bytes": size_bytes, "characters": len(content)}

    measured_pairs: list[dict] = []
    spend = ProbeCost()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        workdir = tmp / "work"
        workdir.mkdir()
        injected = tmp / instruction_file.name
        injected.write_text(content, encoding="utf-8")

        for index in range(max(1, pairs)):
            order = ("with", "without") if index % 2 else ("without", "with")
            arms: dict[str, dict] = {}
            for label in order:
                arms[label] = run = _probe(PROBE_PROMPT, workdir,
                                           injected if label == "with" else None, timeout)
                spend.record(run)
                if not run["ok"]:
                    break
            if len(arms) == 2 and all(a["ok"] for a in arms.values()):
                low, high = sorted((arms["with"]["ctx"], arms["without"]["ctx"]))
                if low < high * ARM_AGREEMENT:
                    # An arm that loaded a fraction of the prefix is not a measurement of
                    # this file. Observed: a `with` arm returned a near-empty context
                    # against a full baseline and passed the absolute floor, turning a
                    # reproducible positive difference into a large negative mean. A fixed
                    # floor cannot catch this, because
                    # the legitimate baseline itself differs by an order of magnitude
                    # between windows; what is stable is that the two arms of one pair
                    # differ by an instruction file, not by 95% of the prefix.
                    measured_pairs.append({
                        "pair": index + 1, "order": list(order),
                        "with": arms["with"]["ctx"], "without": arms["without"]["ctx"],
                        "error": f"arms disagree by more than {1 - ARM_AGREEMENT:.0%} "
                                 f"({low} vs {high}); one arm did not load the prefix",
                    })
                    continue
                measured_pairs.append({
                    "pair": index + 1, "order": list(order),
                    "with": arms["with"]["ctx"], "without": arms["without"]["ctx"],
                    "difference": arms["with"]["ctx"] - arms["without"]["ctx"],
                })
            else:
                measured_pairs.append({
                    "pair": index + 1, "order": list(order),
                    "error": next((a["error"] for a in arms.values() if not a["ok"]),
                                  "incomplete pair"),
                })

    good = [p for p in measured_pairs if "difference" in p]
    if not good:
        return {"ok": False, "reason": "no pair completed", "pairs": measured_pairs,
                **spend.summary()}

    diffs = [p["difference"] for p in good]
    mean = sum(diffs) / len(diffs)
    spread = max(diffs) - min(diffs)
    common = {
        "file": str(instruction_file),
        "harness": "claude_code",
        "bytes": size_bytes,
        "characters": len(content),
        "pairs": measured_pairs,
        "differences": diffs,
        "mean_tokens": round(mean),
        "pair_spread_tokens": spread,
        **spend.summary(),
        "method": "a verbatim copy is appended with --append-system-prompt-file against a "
                  "--bare baseline; arms alternate within one window so drift cannot read "
                  "as an effect. The real file is read, never modified.",
        "channel_caveat": "the content is byte-identical but arrives as an appended system "
                          "prompt rather than through the harness's own CLAUDE.md loading, "
                          "so this is the content's prefix cost and not that plumbing's",
    }
    if mean <= MIN_DETECTABLE_TOKENS:
        return {"ok": False, "inconclusive": True, "loaded": "unproven",
                "reason": f"the arms differ by {round(mean)} tokens on average, inside the "
                          f"{MIN_DETECTABLE_TOKENS}-token noise band",
                **common}
    return {
        "ok": True,
        # Unlike Codex, Claude Code already proves `loaded` for free from attachment
        # records, so this measurement is not needed for that link and does not claim it.
        "loaded": "proven",
        "loaded_evidence": f"injecting this {size_bytes}-byte file moved the fixed prefix "
                           f"by {round(mean)} tokens across {len(good)} interleaved pair(s), "
                           f"spread {spread}",
        "reproducible": _reproducible(diffs, spread),
        **common,
    }


SKILL_LIST_PROMPT = ("List the names of the skills available to you, one per line, with no "
                     "other text. If you have none, reply NONE.")
# The arm's plugin needs a name, and it ends up in the listing as `<plugin>:<skill>`, so a
# short one keeps the channel's own overhead small.
ARM_PLUGIN_NAME = "arm"
# Measured, not guessed: from two real two-pair runs, agreeing to three decimal places.
    # Rounded up,
# because a price tag in a recommendation should not undersell the spend it is proposing.
SKILL_PROBE_COST_USD = 0.18


def _plugin_with_one_skill(tmp: Path, skill_dir: Path) -> Path:
    """A throwaway plugin carrying a copy of exactly one skill.

    The real skill directory is copied, never moved or modified: it is the object under
    measurement.
    """
    plugin = tmp / "arm-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": ARM_PLUGIN_NAME, "version": "0.0.1",
                    "description": "one skill, for an isolated measurement arm"}),
        encoding="utf-8")
    shutil.copytree(skill_dir, plugin / "skills" / skill_dir.name)
    return plugin


def _listed_skills(text: str) -> set[str]:
    """Skill names from an arm's own listing, with any `plugin:` prefix dropped."""
    names = set()
    for line in text.splitlines():
        candidate = line.strip().strip("-*` ").split(":")[-1].strip()
        if candidate and " " not in candidate:
            names.add(candidate)
    return names


def measure_skill_cost(skill_dir: Path, *, pairs: int = PROBE_PAIRS,
                       timeout: int = 420) -> dict:
    """Measure what one skill adds to every Claude Code request.

    This project recorded "no CLI flag disables one skill, so there is no single-component
    arm" five separate times, and it was the sixth unverified blocker here to turn out
    false. It is true that nothing disables one skill — and beside the point, because every
    working arm in this project is built by *addition*, not subtraction:

        without = --setting-sources ""                      (11 built-in skills, no user ones)
        with    = --setting-sources "" --plugin-dir <copy>   (the same 11, plus this one)

    `--plugin-dir` loads a plugin "for this session only", and a plugin may carry skills, so
    a plugin carrying exactly one skill is exactly one skill added.

    The arm only works in **default mode**. Under `--bare` the flag is accepted and loads
    nothing, because bare mode skips plugin sync. That is not an inference: a nonce skill
    named `zqx-nonce-probe` was injected both ways and asked for by name — absent under
    `--bare`, present under default mode, with the context size differing by an order of
    magnitude between them. Any future change here
    should be checked the same way, because a silently empty treatment arm reports a real
    component as free.

    Each pair therefore *verifies its own arms* by asking each one to list its skills: the
    name must appear in the `with` listing and be missing from the `without` one. A pair
    that fails that check is rejected rather than averaged, which is what keeps this a
    measurement of this skill instead of of plugin plumbing.
    """
    from .useful_probe import _probe

    skill_dir = Path(skill_dir)
    entry = skill_dir / "SKILL.md"
    if not entry.is_file():
        return {"ok": False, "reason": f"no SKILL.md under {skill_dir}"}
    content = entry.read_text(errors="replace")

    measured_pairs: list[dict] = []
    spend = ProbeCost()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        workdir = tmp / "work"
        workdir.mkdir()
        plugin = _plugin_with_one_skill(tmp, skill_dir)
        base = ("--setting-sources", "")
        arm_args = {"without": base, "with": base + ("--plugin-dir", str(plugin))}

        for index in range(max(1, pairs)):
            order = ("with", "without") if index % 2 else ("without", "with")
            arms: dict[str, dict] = {}
            for label in order:
                for attempt in range(ARM_RETRIES + 1):
                    run = _probe(SKILL_LIST_PROMPT, workdir, None, timeout,
                                 bare=False, extra_args=arm_args[label])
                    spend.record(run)
                    if run["ok"]:
                        break
                    if attempt < ARM_RETRIES:
                        time.sleep(ARM_RETRY_DELAY)
                arms[label] = run
                if not run["ok"]:
                    break
            record = {"pair": index + 1, "order": list(order)}
            if len(arms) != 2 or not all(a["ok"] for a in arms.values()):
                record["error"] = next((a["error"] for a in arms.values() if not a["ok"]),
                                       "incomplete pair")
                measured_pairs.append(record)
                continue

            listed = {label: _listed_skills(arms[label]["text"]) for label in arms}
            record.update({"with": arms["with"]["ctx"], "without": arms["without"]["ctx"],
                           "listed_with": len(listed["with"]),
                           "listed_without": len(listed["without"])})
            if not listed["without"]:
                record["error"] = "the baseline listed no skills at all, so the absence " \
                                  "check is vacuous"
            elif skill_dir.name not in listed["with"]:
                record["error"] = f"{skill_dir.name} is missing from the treatment arm's " \
                                  "own listing, so the arm loaded nothing"
            elif skill_dir.name in listed["without"]:
                record["error"] = f"{skill_dir.name} is already in the baseline's listing, " \
                                  "so the arms do not differ by it"
            else:
                low, high = sorted((record["with"], record["without"]))
                if low < high * ARM_AGREEMENT:
                    record["error"] = (f"arms disagree by more than {1 - ARM_AGREEMENT:.0%} "
                                       f"({low} vs {high}); one arm did not load the prefix")
                else:
                    record["difference"] = record["with"] - record["without"]
            measured_pairs.append(record)

    good = [p for p in measured_pairs if "difference" in p]
    if not good:
        return {"ok": False, "reason": "no pair completed with both arms verified",
                "skill": skill_dir.name, "pairs": measured_pairs,
                **spend.summary()}

    diffs = [p["difference"] for p in good]
    mean = sum(diffs) / len(diffs)
    spread = max(diffs) - min(diffs)
    common = {
        "skill": skill_dir.name,
        "file": str(entry),
        "digest": _skill_digest(skill_dir),
        "harness": "claude_code",
        "bytes": entry.stat().st_size,
        "characters": len(content),
        "pairs": measured_pairs,
        "differences": diffs,
        "mean_tokens": round(mean),
        "pair_spread_tokens": spread,
        **spend.summary(),
        "method": "a copy of the skill is added through --plugin-dir against a "
                  '--setting-sources "" baseline; each arm lists its own skills so the '
                  "pair proves it differs by this one, and arms alternate within one "
                  "window so drift cannot read as an effect. The real skill is copied, "
                  "never modified.",
        "channel_caveat": "the skill arrives as a plugin, so its listing entry reads "
                          f"`{ARM_PLUGIN_NAME}:{skill_dir.name}` and carries a few tokens "
                          "the installed copy does not; and the baseline holds 11 skills "
                          "rather than the real listing's full set, so this is the marginal "
                          "cost of adding one, not a share of a measured total",
    }
    if mean <= MIN_DETECTABLE_TOKENS:
        return {"ok": False, "inconclusive": True,
                "reason": f"the arms differ by {round(mean)} tokens on average, inside the "
                          f"{MIN_DETECTABLE_TOKENS}-token noise band",
                **common}
    return {"ok": True, "reproducible": _reproducible(diffs, spread), **common}


def _skill_digest(directory: Path) -> str:
    """A content hash of a whole skill directory, for recognizing an identical second copy."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _installed_skills(scan: dict) -> tuple[list[Path], list[dict]]:
    """Skill directories Claude Code can actually be asked to load, and what was dropped.

    Two filters here were added only after a real run, because a stubbed test cannot see
    either of them. The inventory listed 12 candidates under 6 names; the harness's own
    listing names 4. The difference:

    * **Nested fixtures.** `nested-skill` ships eval fixtures at
      `nested-skill/evals/fixtures/skills/sample-with-evals/SKILL.md`. Those have the shape
      of a skill and are not one — the harness does not list them — so injecting them would
      have measured something that is never loaded, and reported it beside real figures.
    * **A second identical copy.** A name can be installed under both a harness-specific
      and a shared skills root, byte-identical. The harness lists each name once, so paying
      twice buys a provably equal number.
    """
    candidates, dropped = [], []
    for item in scan.get("files") or []:
        if item.get("kind") != "skills" or item.get("origin") != "declared":
            continue
        path = Path(item["path"])
        # The inventory lists every file inside a skill, scripts and references included.
        # One skill is one SKILL.md, and taking any other file's parent would inject a
        # subdirectory as if it were the skill.
        if path.name != "SKILL.md":
            continue
        directory = path.parent
        # Only Claude Code has the flag. A skill under `~/.agents/skills` is read by both
        # harnesses, so it is measurable here too — what it is not is a Codex result.
        if "/.codex/" in str(directory):
            continue
        if any((parent / "SKILL.md").is_file() for parent in directory.parents
               if parent != directory):
            dropped.append({"skill": directory.name, "path": str(directory),
                            "reason": "nested inside another skill, so the harness does not "
                                      "list it as an installed skill"})
            continue
        candidates.append(directory)

    targets, seen = [], {}
    for directory in sorted(set(candidates)):
        digest = _skill_digest(directory)
        if digest in seen:
            dropped.append({"skill": directory.name, "path": str(directory),
                            "identical_to": seen[digest], "digest": digest,
                            "reason": f"byte-identical to {seen[digest]}, which the harness "
                                      "lists once; measuring it again would buy the same "
                                      "number twice"})
            continue
        seen[digest] = str(directory)
        targets.append(directory)
    return targets, dropped


def _ambiguous_names(targets: list[Path]) -> dict[str, list[str]]:
    """Names that survive as two *different* files, where the harness loads only one.

    Identical copies were already folded together, so what is left here is a name installed
    twice with diverging content — one skill differed by about a hundred bytes between
    under the other. Both are measurable and only one is loaded, and there is
    already established that which one cannot be recovered: a `Skill` call carries only
    `{skill, args}`, and the listing's digests did not reproduce from the files on disk. So
    each figure is reported with the other named beside it, rather than one of them being
    silently chosen.
    """
    by_name: dict[str, list[str]] = {}
    for directory in targets:
        by_name.setdefault(directory.name, []).append(str(directory))
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def measure_skills(root: str, *, pairs: int = PROBE_PAIRS, timeout: int = 420,
                   whole_environment: bool = False, only: str | None = None,
                   resume: bool = False, checkpoint: bool = False) -> dict:
    """Measure the per-request cost of each Claude Code skill discovered for a root.

    Skills are the largest group of components in this stack and every one of them had an
    unproven `cost` link, on a blocker that does not hold. `only` measures a single skill
    by name, because at a fraction of a dollar a skill the whole set is a deliberate spend rather
    than something to trigger by accident.
    """
    from .core import inspect as inspect_stack

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")
    started = time.perf_counter()
    scan = inspect_stack(str(base), whole_environment=whole_environment)

    targets, skipped = _installed_skills(scan)
    if only:
        targets = [d for d in targets if d.name == only]

    ambiguous = _ambiguous_names(targets)
    from .measurement_batch import MeasurementBatch

    batch = MeasurementBatch(base, "measure-skills", resume=resume, checkpoint=checkpoint,
                             parameters={"pairs": pairs})
    measurements = []
    spend = ProbeCost()
    for directory in targets:
        path = directory / "SKILL.md"
        digest, skip = batch.prepare(path, "skill_measurements", "claude_code", skill=True)
        if skip:
            skipped.append({"skill": directory.name, "path": str(directory), "reason": skip})
            continue
        result = measure_skill_cost(directory, pairs=pairs, timeout=timeout)
        spend.include(result)
        siblings = [p for p in ambiguous.get(directory.name, []) if p != str(directory)]
        if siblings:
            result["ambiguous_with"] = siblings
            result["ambiguous"] = ("this name is installed twice with different content and "
                                   "the harness lists it once, so this figure belongs to "
                                   "whichever copy is loaded — which cannot be determined")
        batch.record(result, path, "skill_measurements", "claude_code", digest, skill=True)
        if result.get("ok") or result.get("inconclusive"):
            measurements.append(result)
        else:
            skipped.append({"skill": directory.name, "path": str(directory),
                            **result})

    return {
        "schema_version": "0.1",
        "mode": "measure_skills",
        **batch.summary(),
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        # So a later comparison can tell which figure is newer. Two artifacts
        # measuring one file disagreed and neither carried a time, so neither
        # could be shown to supersede the other.
        "measured_at": _now(),
        "harness": "claude_code",
        "skill_measurements": measurements,
        "profiler_cost": {
            "stage": "measure_skills",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "claude_runs": spend.calls,
            **spend.summary(),
            "spend_usd": spend.summary()["cost_usd"],
        },
        "skipped": skipped,
        "notes": [
            "Each pair asks both arms to list their own skills, so a pair is only counted "
            "when the treatment arm really carries the skill and the baseline really does "
            "not. This replaces the assumption that a flag did what its help text says.",
            "The figure is a marginal cost against an 11-skill baseline, measured through "
            "the plugin channel. It is the cost of adding this skill, not its share of a "
            "measured total.",
            "A measurement carrying `ambiguous_with` is one of two installed copies of a "
            "name the harness lists once. It is a real figure for that file and cannot be "
            "attributed to the listed skill.",
        ],
    }


def measure_instructions(root: str, *, pairs: int = PROBE_PAIRS, timeout: int = 420,
                         whole_environment: bool = False, resume: bool = False,
                         checkpoint: bool = False) -> dict:
    """Measure every instruction file discovered for a root, in the harness that loads it.

    Claude Code instruction *loading* is already provable from `attachment` records for
    free, so nothing is spent to establish that. Its *cost* is not: the record reports
    characters, and no tokenizer runs here. So a CLAUDE.md is probed for the token figure
    only, and the free record remains the evidence for loading.
    """
    from .core import inspect as inspect_stack

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")
    started = time.perf_counter()
    scan = inspect_stack(str(base), whole_environment=whole_environment)

    measurements, skipped = [], []
    spend = ProbeCost()
    runs = {"codex": 0, "claude_code": 0}
    from .measurement_batch import MeasurementBatch

    batch = MeasurementBatch(base, "measure-instructions", resume=resume, checkpoint=checkpoint,
                             parameters={"pairs": pairs})
    measurers = {"agents_md": measure_instruction_cost,
                 "claude_md": measure_claude_instruction_cost}
    for item in scan.get("files") or []:
        measure = measurers.get(item.get("kind"))
        if measure is None or item.get("origin") != "declared":
            continue
        path = Path(item["path"])
        harness = "codex" if item["kind"] == "agents_md" else "claude_code"
        digest, skip = batch.prepare(path, "instruction_measurements", harness)
        if skip:
            skipped.append({"file": str(path), "reason": skip})
            continue
        result = measure(path, pairs=pairs, timeout=timeout)
        spend.include(result)
        runs[harness] += result.get("llm_calls", 0)
        result["file"] = str(path)
        batch.record(result, path, "instruction_measurements", harness, digest)
        if result.get("ok") or result.get("inconclusive"):
            measurements.append(result)
        else:
            # `pairs` carries each attempt's own error, which is the difference between
            # "the harness never answered" and "a pair ran and was rejected as unsound".
            # Dropping it left `no pair completed` as the only account of a failure, so a
            # dead harness and a method that does not apply to this file read identically —
            # the same defect the usefulness probe was fixed for, still in the cost path.
            # `measure_skills` already carried the detail; this branch did not.
            skipped.append({**result,
                            "arm_errors": sorted({
                                str(p.get("error")) for p in (result.get("pairs") or [])
                                if p.get("error")})})

    return {
        "schema_version": "0.1",
        "mode": "measure_instructions",
        **batch.summary(),
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        # So a later comparison can tell which figure is newer. Two artifacts
        # measuring one file disagreed and neither carried a time, so neither
        # could be shown to supersede the other.
        "measured_at": _now(),
        "harnesses": sorted({m["harness"] for m in measurements + skipped if m.get("harness")}),
        "instruction_measurements": measurements,
        "skipped": skipped,
        "profiler_cost": {
            "stage": "measure_instructions",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "codex_runs": runs["codex"],
            "claude_runs": runs["claude_code"],
            **spend.summary(),
        },
        "notes": [
            "A measured difference is Observed: content that changes the fixed prefix is "
            "content that entered the context window.",
            "Arms alternate within one window. The equivalent Claude Code route was "
            "abandoned because an unpaired run read baseline drift as a treatment effect.",
            "Claude Code instruction loading is not probed here: `attachment` records "
            "already prove it for free.",
        ],
    }

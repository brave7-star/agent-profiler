"""Counterfactual evaluation, gated on expected information gain.

Stage 6 answers "what does carrying this cost?" — and refuses to spend when the answer is
not worth the price of asking.

## Why there is a gate at all

Stage 5 measured its own economics and failed them: the semantic judge spent an order of
magnitude more than the avoidable tokens it identified were worth. The lesson is not that evaluation is useless, but
that *the quantity under test must be large and recurring* before paying to measure it.

So every evaluation here declares, before running:

    expected_information_gain / evaluation_cost

and anything below `MIN_EIG_RATIO` is **refused**, with the arithmetic shown. A refusal is
a successful outcome, not an error.

## The cheap experiment beats the expensive one

The obvious way to ask "what would this configuration cost?" is to re-run real sessions
with and without it. That is enormously expensive and, because model output varies, a
single pair proves almost nothing.

A tool schema's cost can instead be measured *directly*: send one trivial prompt with the
server configured and one without, and difference the total context. On this machine that
cost cents and returned a figure reproducible to within a rounding error, which then
projects across an entire corpus. That is the shape every evaluation here should take:
isolate the quantity, measure it once, project deterministically.

## Cost is not waste

A schema that costs tokens may be earning them. This module reports what carrying something
costs and how often it was used; it does **not** conclude that anything should be removed.
Establishing that requires showing the work still succeeds without it, which is a quality
evaluation this module deliberately does not fake.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .core import is_excluded as _is_excluded_entity
from .traces import SELF_PROBE_PROMPT, collect_root


def _is_excluded(exclusions: dict | None, name: str) -> bool:
    return _is_excluded_entity(exclusions, "mcp", name)

EVALUATE_SCHEMA_VERSION = "0.6"

# An evaluation must be expected to reveal this many dollars of cost per dollar spent.
# Set from the Stage 5 result: the judge ran at roughly 0.5 and was judged a failed trade.
MIN_EIG_RATIO = 5.0

# Opus pricing per million tokens, used only to express measurements in dollars. These are
# published rates, not measurements, and are stated as an assumption wherever they are used.
RATE_CACHE_WRITE_1H = 18.75
RATE_CACHE_READ = 1.50
RATE_INPUT = 15.00
RATE_OUTPUT = 75.00

# A trivial prompt: the goal is to measure the fixed prefix, not to exercise the model.
# Defined in traces.py so the collector can recognise these runs in the corpus and keep
# the profiler's own sessions out of the user's totals.
PROBE_PROMPT = SELF_PROBE_PROMPT
PROBE_REPEATS = 2

# Probes fail outright, often. The failure shape is now identified: it is what the harness
# returns when it cannot authenticate or initialize — exit 0, `subtype: "success"`, a
# well-formed envelope reporting `input_tokens: 2`, billed ~$0.0001. Reproduced
# deterministically by running with an empty auth token, and again with
# `--setting-sources project,local`, which drops the user settings this token lives in.
#
# What is still unattributed is why it happens *intermittently* on an otherwise healthy
# invocation, at a pooled 40% (16 of 39). So the handling stays the same: detect reliably,
# retry within a budget derived from the rate, and report the money spent on failures
# separately.
PROBE_FAILURE_RATE = 0.44
# Enough retries that an arm almost never ends up with nothing, without chasing a clean run:
# at a 44% failure rate, 3 retries leave a 3.8% chance of an empty arm.
PROBE_RETRY_PER_ARM = 3
# Below this, a difference between the two arms is not distinguishable from probe noise, so
# it is not a measurement of a schema. Observed: a server whose token had expired
# could not connect, registered no tools, and the pairing returned -6 tokens as
# `ok: true, reproducible: true`. A reproducible zero reads as *more* trustworthy than a
# noisy real figure, which makes it the worst possible shape for a wrong answer.
MIN_DETECTABLE_SCHEMA_TOKENS = 50

# Measured cost of one *successful* probe, from a real run of four. It is
# used only as a stand-in for a failure that returned no envelope to read a figure from, so
# it is an upper bound there: the common failure mode bills two input tokens.
PROBE_COST_USD = 0.073

# A prefix probe carries ~19k tokens. Anything far above that is not measuring a prefix —
# it is an agentic session that ran away, which is how a stdin leak cost an order of
# magnitude more than a controlled probe before being
# caught. The ceiling turns that class of accident into a rejected sample.
PROBE_CONTEXT_CEILING = 60_000


def _probe(config_path: Path | None, timeout: int) -> dict:
    """One probe call. Returns the total context it carried, or an error."""
    binary = shutil.which("claude")
    if binary is None:
        raise RuntimeError("the `claude` CLI is required for schema measurement but was not found")
    argv = [binary, "-p", PROBE_PROMPT, "--output-format", "json", "--strict-mcp-config"]
    if config_path is not None:
        argv += ["--mcp-config", str(config_path)]
    try:
        # stdin MUST be closed. `capture_output` redirects only stdout/stderr, so the child
        # inherits this process's stdin — and `claude -p` appends piped stdin to its
        # prompt. A probe invoked from a shell heredoc therefore measured a prompt
        # containing whatever text the parent had not yet consumed. Observed, not
        # theorised: two such probes ran dozens of agentic turns each and cost an order of
        # magnitude more than a controlled probe. A measurement whose input depends
        # on the caller's stdin is not a measurement.
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if done.returncode != 0:
        return {"ok": False, "error": f"exit {done.returncode}: {done.stderr.strip()[:160]}"}
    try:
        envelope = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non-JSON envelope"}

    usage = envelope.get("usage") or {}
    total = (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    # A probe that carried almost no context did not load a prefix at all; treating it as a
    # valid sample would silently corrupt the difference. The envelope's own cost figure is
    # carried through, because a failure that was billed must not be accounted as free — and
    # because this failure mode is billed almost nothing, which is itself worth reporting.
    #
    # The cause of this shape is now known: it is what the harness returns when it cannot
    # authenticate or initialize. Reproduced deterministically with an empty auth token —
    # exit 0, subtype "success", two input tokens.
    if total < 1000:
        return {
            "ok": False,
            "error": f"implausible context total ({total}); the harness did not load a "
                     "prefix, which is what an authentication or initialization failure "
                     "looks like",
            "cost_usd": envelope.get("total_cost_usd") or 0.0,
        }
    # The opposite accident: a probe that became a working session. Caught after a stdin
    # leak let two probes run dozens of turns each, so the guard is a hard ceiling
    # rather than a hope that the fix holds.
    if total > PROBE_CONTEXT_CEILING:
        return {
            "ok": False,
            "error": f"context total ({total}) exceeds the {PROBE_CONTEXT_CEILING} ceiling; "
                     "this probe did work instead of measuring a prefix",
            "cost_usd": envelope.get("total_cost_usd") or 0.0,
        }
    return {
        "ok": True,
        "total_context_tokens": total,
        "cache_read": int(usage.get("cache_read_input_tokens") or 0),
        "cost_usd": envelope.get("total_cost_usd") or 0.0,
    }


def _probe_accounting(results: dict) -> dict:
    """Separate the spend that bought a measurement from the spend that bought nothing.

    Both are real money, but only the second is avoidable, so folding it into one total
    would hide the only figure worth acting on. The word reserved for judging a user's
    configuration is deliberately not used here: this is the profiler's own spend.
    """
    attempted, failed = results["attempted"], results["failed"]
    failed_usd, assumed = results["failed_usd"], results["failed_assumed"]
    note = "every probe returned a usable sample"
    if failed:
        note = (
            f"{failed} of {attempted} probes returned no usable sample, costing "
            f"${failed_usd:.3f} that bought no measurement. The dominant failure mode is "
            "billed almost nothing — it reports two input tokens — so what it mostly costs "
            "is latency and a lost sample, not dollars. The cause is not known; see "
            "."
        )
        if assumed:
            note += (
                f" {assumed} failure(s) returned no envelope at all, so ${PROBE_COST_USD} "
                "per probe is assumed for those rather than booking them as free."
            )
    return {
        "probes_attempted": attempted,
        "probes_failed": failed,
        "observed_failure_rate": round(failed / attempted, 3) if attempted else None,
        "expected_failure_rate": PROBE_FAILURE_RATE,
        "retries_used": results["retries"],
        "retry_budget_per_arm": PROBE_RETRY_PER_ARM,
        "failed_probe_usd": round(failed_usd, 6),
        "failed_probe_usd_assumed": assumed,
        "failed_probe_note": note,
    }


def measure_schema_cost(servers: dict, *, repeats: int = PROBE_REPEATS, timeout: int = 420) -> dict:
    """Measure what a set of MCP servers adds to every request's fixed prefix.

    Paired probes differing only in MCP configuration. The difference in total context is
    the schema's size; repeating it establishes whether the figure is stable enough to
    project from.
    """
    results = {"with": [], "without": [], "errors": [], "retries": 0, "cost_usd": 0.0,
               "failed_usd": 0.0, "failed_assumed": 0, "attempted": 0, "failed": 0}
    wanted = max(1, repeats)
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.json"
        empty.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        loaded = Path(tmp) / "with.json"
        loaded.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

        # Budget per arm, not per run: a bad streak in one arm must not consume the retries
        # the other arm needs, because an arm with no sample makes the whole pairing useless.
        for label, config in (("without", empty), ("with", loaded)):
            budget = PROBE_RETRY_PER_ARM
            while len(results[label]) < wanted:
                probe = _probe(config, timeout)
                results["attempted"] += 1
                spent = probe.get("cost_usd") or 0.0
                results["cost_usd"] += spent
                if probe["ok"]:
                    results[label].append(probe["total_context_tokens"])
                    continue
                results["failed"] += 1
                # Where no envelope came back at all there is no figure to use, so the
                # measured per-probe cost stands in rather than booking the failure as free.
                if probe.get("cost_usd") is None:
                    results["failed_usd"] += PROBE_COST_USD
                    results["failed_assumed"] += 1
                else:
                    results["failed_usd"] += spent
                results["errors"].append({"arm": label, "error": probe["error"]})
                if budget <= 0:
                    break
                budget -= 1
                results["retries"] += 1

    accounting = _probe_accounting(results)
    if not results["with"] or not results["without"]:
        return {
            "ok": False,
            "reason": "a probe arm produced no usable sample",
            "errors": results["errors"],
            "cost_usd": round(results["cost_usd"], 6),
            **accounting,
        }

    with_mean = sum(results["with"]) / len(results["with"])
    without_mean = sum(results["without"]) / len(results["without"])
    spread = max(results["with"]) - min(results["with"]) if len(results["with"]) > 1 else None
    schema_tokens = round(with_mean - without_mean)
    common = {
        "schema_tokens": schema_tokens,
        "samples": {"with": results["with"], "without": results["without"]},
        "with_mean": round(with_mean),
        "without_mean": round(without_mean),
        "repeat_spread_tokens": spread,
        "errors": results["errors"],
        "cost_usd": round(results["cost_usd"], 6),
        **accounting,
        # The probes run `claude -p`, so this figure is the prefix cost *under Claude Code*.
        # A server declared for several harnesses is reached differently by each, so the
        # consumer must not add this number to another harness's total.
        "harness": "claude_code",
        "method": "paired `claude -p` probes differing only in --mcp-config; the statistic is "
        "input + cache_read + cache_creation, which is stable across cache states",
    }

    # Noise within an arm bounds what a difference between arms can mean.
    noise = max(spread or 0, _arm_spread(results["without"]), MIN_DETECTABLE_SCHEMA_TOKENS)
    if schema_tokens <= noise:
        return {
            "ok": False,
            "inconclusive": True,
            "reason": (
                f"the two arms differ by {schema_tokens} tokens, which is inside the "
                f"{noise}-token noise band, so no schema was isolated. The likeliest cause "
                "is that the server registered no tools in the probe — a server that fails "
                "to connect contributes nothing to context. `claude mcp list` reports "
                "connection status and is free to run."
            ),
            **common,
        }

    return {
        "ok": True,
        "reproducible": spread is not None and spread <= max(50, 0.01 * with_mean),
        **common,
    }


def _arm_spread(samples: list) -> int:
    return max(samples) - min(samples) if len(samples) > 1 else 0


def project_corpus_cost(schema_tokens: int, sessions: int, responses: int) -> dict:
    """Project one fixed-prefix cost across an observed corpus.

    The prefix is written once per session and read on every subsequent response, so the
    recurring cost is dominated by cache reads rather than by the one-off write.
    """
    created = sessions * schema_tokens
    read = responses * schema_tokens
    usd = created / 1e6 * RATE_CACHE_WRITE_1H + read / 1e6 * RATE_CACHE_READ
    return {
        "schema_tokens": schema_tokens,
        "sessions": sessions,
        "responses": responses,
        "cache_creation_tokens": created,
        "cache_read_tokens": read,
        "total_tokens": created + read,
        "estimated_usd": round(usd, 2),
        "assumptions": [
            "The schema sits in the cached prefix, so it is written once per session and "
            "read once per model response. This follows from how prompt caching works and "
            "from the measurement, which showed the difference entirely in cache creation.",
            f"Dollar figures use published Opus rates (cache write 1h ${RATE_CACHE_WRITE_1H}/M, "
            f"cache read ${RATE_CACHE_READ}/M). Rates are an input, not a measurement.",
            "Sessions on a cheaper model, or with 5-minute cache TTL, cost less than this.",
        ],
    }


def eig_gate(*, name: str, dollars_at_stake: float, evaluation_cost_usd: float) -> dict:
    """Decide whether an evaluation is worth running, and show the arithmetic either way."""
    ratio = (dollars_at_stake / evaluation_cost_usd) if evaluation_cost_usd > 0 else float("inf")
    approved = ratio >= MIN_EIG_RATIO
    return {
        "candidate": name,
        "dollars_at_stake": round(dollars_at_stake, 4),
        "evaluation_cost_usd": round(evaluation_cost_usd, 4),
        "eig_ratio": round(ratio, 2) if ratio != float("inf") else None,
        "threshold": MIN_EIG_RATIO,
        "decision": "approved" if approved else "refused",
        "reason": (
            f"Expected to reveal ${dollars_at_stake:.2f} of recurring cost for "
            f"${evaluation_cost_usd:.2f} of measurement."
            if approved
            else f"Would spend ${evaluation_cost_usd:.2f} to resolve ${dollars_at_stake:.2f}; "
            f"below the {MIN_EIG_RATIO}x threshold, so it is not run."
        ),
    }


def _behavioural_ab_estimate(sessions: int, responses: int, tokens: int) -> dict:
    """What a real behavioural A/B would cost, so the gate can refuse it with numbers.

    Re-running sessions reproduces their whole token consumption, twice, and model output
    varies — so one pair per session is a weak signal for a large bill.

    The cost is bracketed rather than pinned: a re-run benefits from caching to an unknown
    degree. The gate is handed the *cheapest* bound, so a refusal is never an artefact of
    having assumed the expensive one.
    """
    upper = tokens * 2 / 1e6 * RATE_INPUT
    lower = tokens * 2 / 1e6 * RATE_CACHE_READ
    return {
        "method": "re-run each session under both configurations and compare outcomes",
        "sessions": sessions,
        "arms": 2,
        "estimated_tokens": tokens * 2,
        "estimated_usd": round(lower, 2),
        "estimated_usd_range": [round(lower, 2), round(upper, 2)],
        "range_basis": "lower bound assumes every re-run token is a cache read, upper bound "
        "assumes every one is fresh input; the gate uses the lower bound",
        "limits": [
            "Model output varies between runs, so a single pair per session does not "
            "establish a difference; several repeats per arm would be needed.",
            "Re-running a session requires reproducing its environment, which this tool "
            "cannot guarantee for sessions that mutated files or called external services.",
        ],
    }


def component_cost_status() -> dict:
    """Why per-component prefix cost is not measured, and what is measurable instead.

    Two routes were tried and both failed:

    - **Removal.** The *authentication* obstacle turned out to be solvable: supplying the
      auth block through `--settings` alongside `--setting-sources project,local`
      authenticates reliably (3/3, against a no-auth control failing 0/3). What still
      blocks this route is **granularity**, not auth — `--setting-sources` is
      all-or-nothing, so it can exclude the user settings source entirely and cannot
      disable one skill, one agent or one command. An all-or-nothing arm measures an
      aggregate, and dividing an aggregate among its members is the attribution this
      module refuses to invent.
    - **Additive injection.** `--plugin-dir` skills do not join the measured prefix. An
      early sequential test appeared to show a cost per skill that grew with the number of
      skills; under *interleaved* pairs the identical fixture comes back slightly negative.
      The apparent effect was baseline drift between runs read as a treatment effect.

    What is measurable, for free, from records the harness already writes: the skill
    listing's **aggregate** size. It is one shared block covering every skill a stack has
    installed, read from `attachment` records. Dividing
    it by the skill count would invent an attribution, which is why no per-skill figure is
    reported rather than a wrong one.
    """
    return {
        "measurable": False,
        "blocker": "NO_WORKING_METHOD",
        "routes_tried": [
            {"route": "removal by excluding the user settings source",
             "result": "failed, for a different reason than first recorded",
             "evidence": "the auth obstacle is solved — `--settings` carrying the auth env "
                         "block with `--setting-sources project,local` authenticates 3/3, "
                         "control without auth 0/3 — but the flag is all-or-nothing, so it "
                         "cannot isolate one component, which is what a per-component cost "
                         "requires"},
            {"route": "additive injection via --plugin-dir",
             "result": "failed",
             "evidence": "interleaved pairs of the identical fixture give -8 and -7 tokens; "
                         "an earlier sequential run suggesting +132 was baseline drift"},
        ],
        "measurable_instead": "the skill listing's aggregate size, read free from "
                              "attachment records; it is shared across all skills and is "
                              "deliberately not divided among them",
    }


def evaluate_root(
    root: str,
    *,
    measure: bool = False,
    harness: str | None = None,
    timeout: int = 420,
    exclusions: dict | None = None,
    whole_environment: bool = False,
) -> dict:
    """Evaluate the recurring cost of a root's declared MCP servers.

    With `measure=False` (the default) nothing is spent: the report shows what would be
    measured, what it would cost, and whether the gate approves it.

    Passing no `exclusions` means "find them", not "there are none". This is the only stage
    that opens a connection to someone else's endpoint, so the default has to be the
    cautious one: a caller who simply forgot the argument would otherwise probe every
    declared server, including one the user had written down as off limits.
    """
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    started = time.perf_counter()
    from .core import inspect as inspect_stack
    from .core import load_exclusions

    if exclusions is None:
        exclusions = load_exclusions(None, base)

    scan = inspect_stack(str(base), exclusions=exclusions,
                         whole_environment=whole_environment)
    traces = collect_root(str(base), whole_environment=whole_environment)
    observed = traces["observed"]
    sessions = observed["sessions_collected"]
    responses = observed["assistant_messages"]
    calls_by_tool = observed["calls_by_tool"]

    # Claude Code is the harness whose schema can be probed from here.
    servers: dict[str, dict] = {}
    for declaration in scan.get("mcp_servers") or []:
        if declaration.get("source") in {"project_root", "user_claude_json", "project_claude_dir"}:
            servers.setdefault(declaration["name"], declaration)

    # No transcript means nothing was observed, which is a different fact from a server
    # that cost nothing and was called nothing. Observed: this stage reported
    # `calls_observed: 0` and nothing at stake on a root whose history was simply filed
    # elsewhere, which reads as a clean result and is not one.
    relocation = observed.get("relocation_candidates") or []

    evaluations = []
    spend = 0.0
    for name in sorted(servers):
        used = sum(n for tool, n in calls_by_tool.items() if tool.startswith(f"mcp__{name}__"))
        entry: dict = {"server": name, "calls_observed": used, "scope": servers[name].get("scope")}

        # A probe launches `claude -p` with this server configured, which makes the
        # process connect to its endpoint. For an excluded server that must never happen,
        # so the check comes before anything else and before any spend.
        if _is_excluded(exclusions, name):
            entry["status"] = "excluded_by_request"
            entry["reason"] = (
                "Excluded by request, so no probe was launched and nothing was sent to it. "
                "Its tool schema still occupies context in every session that loads it; that "
                "cost is real and simply not measured here."
            )
            if (exclusions or {}).get("reason"):
                entry["exclusion_reason"] = exclusions["reason"]
            evaluations.append(entry)
            continue

        if not sessions:
            entry["calls_observed"] = None
            entry["status"] = "no_corpus"
            entry["blocker"] = "NO_TRACES_COLLECTED"
            entry["reason"] = (
                "No transcript was collected for this root, so neither the cost of carrying "
                "this server nor the number of times it was called is known. This is not a "
                "measurement of zero."
            )
            if relocation:
                entry["reason"] += (
                    " History recorded under another working directory was found; see "
                    "`relocation_candidates`."
                )
            evaluations.append(entry)
            continue

        if not measure:
            # Project from a nominal size purely to decide whether measuring is worthwhile.
            nominal = project_corpus_cost(3000, sessions, responses)
            entry["gate"] = eig_gate(
                name=f"mcp_schema:{name}",
                dollars_at_stake=nominal["estimated_usd"],
                evaluation_cost_usd=0.30,
            )
            entry["status"] = "not_measured"
            evaluations.append(entry)
            continue

        config = _server_config(base, name, whole_environment=whole_environment)
        if config is None:
            entry["status"] = "unmeasurable"
            entry["reason"] = "the server's launch configuration could not be reconstructed"
            evaluations.append(entry)
            continue

        measurement = measure_schema_cost({name: config}, timeout=timeout)
        spend += measurement.get("cost_usd", 0.0)
        entry["measurement"] = measurement
        if not measurement.get("ok"):
            # An inconclusive pairing is a different outcome from a probe that never
            # returned: the calls worked, and what they showed was nothing to measure.
            entry["status"] = ("measurement_inconclusive" if measurement.get("inconclusive")
                               else "measurement_failed")
            entry["reason"] = measurement.get("reason")
            evaluations.append(entry)
            continue

        projection = project_corpus_cost(measurement["schema_tokens"], sessions, responses)
        entry["status"] = "measured"
        entry["projection"] = projection
        entry["tokens_per_call"] = (
            round(projection["total_tokens"] / used) if used else None
        )
        entry["interpretation"] = (
            f"Carrying this server cost about {projection['total_tokens']:,} tokens across "
            f"{sessions} sessions. It was called {used} time(s). Cost is not waste: this says "
            "what it costs to have available, not whether it earns that."
            if used
            else f"Carrying this server cost about {projection['total_tokens']:,} tokens across "
            f"{sessions} sessions, and no call to it was observed. Whether it is removable "
            "depends on work these sessions did not cover."
        )
        entry["behavioural_ab"] = eig_gate(
            name=f"behavioural_ab:{name}",
            dollars_at_stake=projection["estimated_usd"],
            evaluation_cost_usd=_behavioural_ab_estimate(
                sessions, responses, observed["tokens"]["total_tokens"]
            )["estimated_usd"],
        )
        evaluations.append(entry)

    measurements = [e["measurement"] for e in evaluations if e.get("measurement")]
    return {
        "schema_version": EVALUATE_SCHEMA_VERSION,
        "mode": "evaluate",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "measured": measure,
        "corpus": {
            "sessions": sessions,
            "responses": responses,
            **({"blocker": "NO_TRACES_COLLECTED",
                "note": "Nothing was observed for this root, so no evaluation here is a "
                        "measurement of zero."} if not sessions else {}),
        },
        **({"relocation_candidates": relocation} if relocation else {}),
        # This is the one stage that contacts a third party, so what it was told to leave
        # alone belongs in its own record. `searched` matters most when nothing was found:
        # a standing exclusion that silently did not apply used to be indistinguishable
        # here from a user who never asked for one, and a run with `--all` looked the same
        # either way while the probe went ahead and connected.
        "exclusions": {
            "source": (exclusions or {}).get("source"),
            "searched": (exclusions or {}).get("searched") or [],
            "reason": (exclusions or {}).get("reason"),
            "mcp": sorted((exclusions or {}).get("names", {}).get("mcp") or []),
        },
        "evaluations": evaluations,
        "behavioural_ab_reference": _behavioural_ab_estimate(
            sessions, responses, observed["tokens"]["total_tokens"]
        ),
        "profiler_cost": {
            "stage": "evaluate",
            "cost_usd": round(spend, 6),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            # Every probe that ran, including the ones that failed: they cost money, so
            # reporting only the usable samples would understate the profiler's own spend.
            "llm_calls": sum(m.get("probes_attempted", 0) for m in measurements),
            # The avoidable part of that spend, kept visible so it can be acted on rather
            # than discovered in a bill.
            "failed_probes": sum(m.get("probes_failed", 0) for m in measurements),
            "failed_probe_usd": round(
                sum(m.get("failed_probe_usd", 0.0) for m in measurements), 6
            ),
        },
        "notes": [
            "Schema cost is measured directly with paired probes, not by re-running sessions: "
            "it is far cheaper and the result is reproducible.",
            "A measured schema size is Observed. The corpus projection is Estimated and lists "
            "its assumptions, including the published rates used to express it in dollars.",
            "Cost is not waste. This reports what carrying a server costs and how often it was "
            "called; deciding its fate needs evidence this stage does not produce.",
            "A behavioural A/B is costed and gated rather than run, because output varies "
            "between runs and a single pair per session would not establish a difference.",
            f"Any evaluation below {MIN_EIG_RATIO}x expected information gain over cost is refused.",
        ],
    }


def _server_config(base: Path, name: str, *, whole_environment: bool = False) -> dict | None:
    """Recover a server's launch configuration so it can be probed in isolation.

    A per-project declaration under another project was unreachable, because only the key
    equal to `base` was read. That made the one server this environment could measure
    `unmeasurable` — while the scanner was already reporting its scope as `other_project`,
    i.e. it knew the declaration existed elsewhere.

    Under `whole_environment` the subject is the environment, so any project's declaration
    is in scope; the matching project is still preferred, so a server declared in several
    places is probed with this root's configuration rather than an arbitrary one.
    """
    candidates = [base / ".mcp.json", Path.home() / ".claude.json"]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        servers = data.get("mcpServers")
        if isinstance(servers, dict) and name in servers:
            return servers[name]
        projects = data.get("projects")
        if not isinstance(projects, dict):
            continue
        fallback = None
        for key, cfg in projects.items():
            if not isinstance(cfg, dict):
                continue
            nested = cfg.get("mcpServers")
            if not isinstance(nested, dict) or name not in nested:
                continue
            if Path(str(key)).resolve() == base:
                return nested[name]
            if whole_environment and fallback is None:
                fallback = nested[name]
        if fallback is not None:
            return fallback
    return None

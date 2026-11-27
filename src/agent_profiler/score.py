"""Preference-aware scoring — and the conditions under which a score is refused.

Stage 7 assembles what the earlier stages proved about each component of a stack and asks
whether a score can be responsibly produced from it. Often the answer is no, and saying so
is the product.

## Why a score is dangerous here

Stage 6 measured the single largest recurring cost a real stack showed: an MCP server's tool schema,
a large recurring spend. That same server was also among the most used. A score that ranks by
what has been measured would therefore rank the most-used component *worst*, purely because
cost is measurable and usefulness is not.

So this module never scores a component on its measured links alone. It scores only when
the links a verdict actually depends on — used, useful, cost — are all present, and
withholds the number otherwise, naming what is missing.

## Whose preferences?

A "preference-aware" score needs someone's preferences. Nobody supplied any. The profiles
here are therefore **named hypotheses, not the user's settings**, and every report says so.

Their purpose is sensitivity analysis rather than ranking: run the same evidence under
several plausible preference sets and see whether the answer changes. If a component's rank
flips between profiles, the evidence does not determine it and the honest output is the
flip, not a number from whichever profile happened to be the default.

## The evidence chain is reported, never interpolated

Each component carries a per-link status drawn from real stage output: installed from the
scanner, used from the trace collector, cost from the Stage 6 measurement. An unproven link
stays unproven. Nothing is inferred from a neighbouring link — "installed and plausible"
is not evidence of "used", and "used" is not evidence of "useful".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .i18n import count, tr

SCORE_SCHEMA_VERSION = "0.7"

# The chain the project set out to prove, in order. `recommendation` is the output of a
# verdict rather than an input to one, so it is reported but never scored.
EVIDENCE_LINKS = ("installed", "loaded", "used", "useful", "cost", "recommendation")
SCORED_LINKS = ("installed", "loaded", "used", "useful", "cost")

# A verdict about whether a component earns its place depends on these three. Missing any of
# them, a total is withheld: ranking on the others would rank by what happens to be
# measurable, which is the specific failure this stage exists to avoid.
REQUIRED_FOR_SCORE = ("used", "useful", "cost")

# Named hypotheses about what someone might care about — NOT elicited from the user.
# Weights are over normalized per-link signals and sum to 1.0 within a profile.
PREFERENCE_PROFILES = {
    "cost_sensitive": {
        "description": "token spend dominates; a component must earn its context",
        "weights": {"cost": 0.6, "used": 0.3, "useful": 0.1},
    },
    "capability_first": {
        "description": "availability matters more than spend; a rarely-used tool still earns "
        "its place if it works when needed",
        "weights": {"cost": 0.1, "used": 0.3, "useful": 0.6},
    },
    "balanced": {
        "description": "no link dominates",
        "weights": {"cost": 1 / 3, "used": 1 / 3, "useful": 1 / 3},
    },
}
DEFAULT_PROFILE = "balanced"

# Where a real set of preferences can live, so "preference-aware" can stop being a claim
# about weights this tool invented. Supplying this file is the only thing that changes
# `preference_source` away from `assumed_by_tool_not_elicited`.
PREFERENCES_FILENAME = "agent-profiler-preferences.json"
ASSUMED, ELICITED = "assumed_by_tool_not_elicited", "supplied_by_user"

PROVEN, PARTIAL, UNPROVEN = "proven", "partial", "unproven"


def load_preferences(path: str | None, root: Path) -> tuple[dict, str, str | None]:
    """Load real preferences if they exist, and otherwise say plainly that none were given.

    Returns `(profiles, source, origin)`. The default is not a silent fallback: a report
    built on invented weights says so in a machine-readable field, every time.

    A supplied file looks like:

        {"profiles": {"mine": {"description": "...",
                               "weights": {"cost": 0.5, "used": 0.3, "useful": 0.2}}}}
    """
    candidate = Path(path).expanduser() if path else root / PREFERENCES_FILENAME
    if not candidate.is_file():
        if path:
            # An explicitly named file that is missing is an error, not a fallback: the
            # caller asked for their own weights and would not want the tool's instead.
            raise FileNotFoundError(f"preferences file does not exist: {candidate}")
        return dict(PREFERENCE_PROFILES), ASSUMED, None

    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"preferences file is not valid JSON: {candidate}: {exc}") from exc
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"preferences file has no `profiles` object: {candidate}")

    cleaned: dict[str, dict] = {}
    for name, profile in profiles.items():
        weights = (profile or {}).get("weights") if isinstance(profile, dict) else None
        if not isinstance(weights, dict):
            raise ValueError(f"profile `{name}` has no `weights` object")
        missing = [k for k in REQUIRED_FOR_SCORE if k not in weights]
        if missing:
            raise ValueError(f"profile `{name}` is missing weights for: {', '.join(missing)}")
        unknown = [k for k in weights if k not in REQUIRED_FOR_SCORE]
        if unknown:
            raise ValueError(f"profile `{name}` has weights for unscorable links: "
                             f"{', '.join(sorted(unknown))}")
        try:
            numeric = {k: float(weights[k]) for k in REQUIRED_FOR_SCORE}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profile `{name}` has a non-numeric weight") from exc
        if any(v < 0 for v in numeric.values()):
            raise ValueError(f"profile `{name}` has a negative weight")
        total = sum(numeric.values())
        if total <= 0:
            raise ValueError(f"profile `{name}` has weights summing to zero")
        # Normalized rather than rejected: someone writing 3/2/1 means a ratio, and the
        # scores stay comparable across profiles only if every profile sums to 1.
        cleaned[name] = {
            "description": str((profile.get("description") or "")) or "supplied by the user",
            "weights": {k: v / total for k, v in numeric.items()},
        }
    return cleaned, ELICITED, str(candidate)


def _link(status: str, *, evidence: str, strength: str | None = None, value=None,
          blocker: str | None = None, unit: str | None = None,
          harness: str | None = None) -> dict:
    """One link of the chain for one component.

    `unit` exists because a bare number is ambiguous and was rendered wrongly: an
    instruction file's cost is measured in *tokens*, and the report printed it as `$380`.
    A value that means something other than calls-or-dollars must say so.

    `harness` is where a figure was *measured*, which is not always where the component is
    installed. A search-server figure measured through Claude Code was being added to Codex's
    floor as well, and a skill in `~/.agents/skills` is read by both harnesses while only
    one of them has the flag that isolates it. A stack total needs to know the difference.
    """
    link = {"status": status, "evidence": evidence, "strength": strength, "value": value,
            "blocker": blocker, "unit": unit}
    if harness:
        link["harness"] = harness
    return link


# Why a link is unproven, as a machine-readable cause. Recommending a next step requires
# knowing which of these it is: collecting more traces fixes HARNESS_NOT_TRACED and does
# nothing at all for NO_EVENT_EMITTED, where the harness never records the fact.
HARNESS_NOT_TRACED = "harness_not_traced"
NO_EVENT_EMITTED = "no_event_emitted"
NOT_MEASURED = "not_measured"
REQUIRES_EARLIER_LINK = "requires_earlier_link"
# The user asked this tool not to touch the thing. Not a gap in the tool, and not a gap
# in the harness: a decision, which the output has to attribute to the user rather than
# report as its own shortfall.
EXCLUDED_BY_REQUEST = "excluded_by_request"
# No transcript was collected at all, so a zero is the absence of evidence rather than
# evidence of absence. Distinct from HARNESS_NOT_TRACED, where other harnesses were read.
NO_TRACES_COLLECTED = "no_traces_collected"

# Entity kinds the harness names in a context listing. Only these can have `loaded` decided
# from a transcript; for the rest the record simply does not exist.
LISTED_KINDS = ("skills", "agents")


def _unproven(reason: str, blocker: str) -> dict:
    return _link(UNPROVEN, evidence=reason, blocker=blocker)


def codex_skill_usage(scan: dict, root: Path | None = None,
                      limit: int | None = None) -> dict | None:
    """Codex skill usage, keyed to the component names `build_chain` uses.

    Attribution is by installed path rather than by name, so this reads the paths the
    inventory actually found instead of rebuilding them from a convention. Returns None
    when the environment declares no Codex skills, so a caller with nothing to look for
    does not walk the whole rollout corpus to learn that.
    """
    from .events import codex_component_usage

    paths = codex_skill_paths(scan)
    if paths is None:
        return None
    return codex_component_usage(paths, root, limit=limit)


def codex_skill_paths(scan: dict) -> dict[str, str] | None:
    """The installed Codex-visible skill paths to attribute reads to, or None if there are
    none to look for.

    `shared` is included, not just `codex`: `~/.agents/skills` is read by both harnesses,
    and Codex's own listing names it as a root. Filtering to `codex` meant a shared skill's
    reads were collected for nobody, so its `used` link could never be proven.

    None rather than `{}` is the signal for "nothing to attribute", because the walk that
    consumes this treats an empty mapping as a real question with an empty answer.
    """
    paths = {k: v for k, v in entity_paths(scan).items()
             if k.split("/")[1] in ("codex", "shared")}
    return paths or None


def entity_paths(scan: dict) -> dict[str, str]:
    """Map each skill component key to the `SKILL.md` the inventory found for it.

    Both `loaded` and `used` attribute Codex evidence by path rather than by name, because
    the same name exists under more than one root in a real environment.
    """
    out = {}
    for item in scan.get("files") or []:
        if item.get("kind") != "skills" or item.get("origin") != "declared":
            continue
        path = Path(item["path"])
        if path.name != "SKILL.md":
            continue
        out[f"{item['scope']}/{item['harness']}/{path.parent.name}"] = item["path"]
    return out


def build_chain(*, scan: dict, traces: dict, evaluation: dict | None = None,
                useful: dict | None = None, codex_usage: dict | None = None,
                codex_listing: dict | None = None) -> list[dict]:
    """Assemble the evidence chain per component from real stage output.

    Nothing here infers one link from another. A component that is installed and looks
    useful is still `unproven` on `used` until a call to it appears in a trace.
    """
    observed = traces.get("observed") or {}
    calls_by_tool = observed.get("calls_by_tool") or {}
    invoked = observed.get("components_invoked") or {}
    loaded = observed.get("components_loaded") or {}
    sessions = observed.get("sessions_collected") or 0
    traced_harnesses = set(observed.get("harnesses") or ["claude_code"])
    # `shared` is not a harness. It is the marker for `~/.agents/skills`, a directory both
    # harnesses read — confirmed by Codex's own listing, whose roots include it. Treating
    # it as an untraced harness made its components permanently unprovable on `used`,
    # waiting for traces from a harness that does not exist. A shared entity is covered
    # once either real harness has been traced.
    if traced_harnesses:
        traced_harnesses.add("shared")

    measured_cost: dict[str, dict] = {}
    for entry in (evaluation or {}).get("evaluations") or []:
        if entry.get("status") == "measured":
            measured_cost[entry["server"]] = entry
    # Skill measurements are keyed by installed path, not by name. A name installed twice
    # with diverging content is two files with two figures, and by-name keying would hand
    # one component the other copy's number.
    measured_skills = {_resolved_dir(m["file"]): m
                       for m in (evaluation or {}).get("skill_measurements") or []
                       if m.get("ok")}
    # A second byte-identical copy is deliberately not measured — it would buy the same
    # number twice — but it is the same bytes, so it carries the same figure with the
    # measured copy named. Resolved from the artifact's own `identical_to`, not inferred.
    for entry in (evaluation or {}).get("skipped") or []:
        # Both sides resolved, for the same reason the indexes are: these are directories
        # recorded by whichever command wrote the artifact, not by the scan.
        source = measured_skills.get(_resolve(entry.get("identical_to")) or "")
        key = _resolve(entry.get("path"))
        if source and key and key not in measured_skills:
            measured_skills[key] = dict(source, measured_on=source["file"])

    # Compliance measurements, keyed the same way so an entity and an instruction file join
    # through one index. Every install of a name is measured separately, so each copy keeps
    # its own verdict rather than one copy's result standing for the listed name.
    useful_skills = {_resolved_dir(m["file"]): m
                     for m in (useful or {}).get("useful_measurements") or []
                     if m.get("file")}

    components: list[dict] = []

    # -- MCP servers: the only component kind whose use is visible in a trace, because a
    # call is named `mcp__<server>__<tool>`.
    # A server declared in several places is one component available to several harnesses;
    # collapsing it to whichever declaration was seen first would misattribute it.
    declared: dict[str, dict] = {}
    for declaration in scan.get("mcp_servers") or []:
        name = declaration["name"]
        harness = "codex" if "codex" in (declaration.get("source") or "") else "claude_code"
        entry = declared.setdefault(name, {"harnesses": set(), "sources": [],
                                           "scope": declaration.get("scope"),
                                           "excluded": False})
        entry["harnesses"].add(harness)
        entry["sources"].append(declaration.get("declared_in", ""))
        # One declaration being excluded excludes the server: the flag is about whether
        # this tool may contact it, and it may not contact it by any route.
        entry["excluded"] = entry["excluded"] or bool(declaration.get("excluded_by_request"))

    # Codex addresses a server by its underscore form, so a declared `demo-server` is
    # called as `mcp__demo_server__*`.
    codex_mcp = (codex_usage or {}).get("mcp_calls") or {}
    codex_mcp_sessions = (codex_usage or {}).get("mcp_sessions") or {}

    for name, entry in sorted(declared.items()):
        harnesses = entry["harnesses"]
        calls = sum(n for tool, n in calls_by_tool.items() if tool.startswith(f"mcp__{name}__"))
        from_codex = codex_mcp.get(name.replace("-", "_"), 0)

        if calls:
            used = _link(PROVEN, evidence=f"{calls} calls across {sessions} sessions",
                         strength="observed", value=calls)
        elif from_codex:
            # Codex emits no MCP tool call of its own; the model writes
            # `tools.mcp__<server>__<tool>(...)` inside the program it hands to `exec`.
            # Every such program in this corpus completed, so the call site was executed.
            spread = codex_mcp_sessions.get(name.replace("-", "_"), 0)
            used = _link(PROVEN,
                         evidence=f"{from_codex} call(s) across {spread} Codex session(s), "
                         f"written as `tools.mcp__{name.replace('-', '_')}__*` inside the "
                         f"programs Codex ran; Codex emits no MCP tool call of its own",
                         strength="observed", value=from_codex)
        elif not sessions:
            # A zero over an empty corpus proves nothing about usage.
            used = _unproven("no transcript was collected for this root, so a zero call "
                             "count is the absence of evidence", NO_TRACES_COLLECTED)
        elif harnesses & traced_harnesses:
            used = _link(PROVEN, evidence=f"no call in {sessions} traced sessions",
                         strength="observed", value=0)
        elif codex_usage and harnesses <= {"codex"}:
            # A zero is only evidence once the detector is known to see positives, and it
            # does: other servers in this same corpus show thousands of calls.
            scanned = codex_usage.get("sessions_scanned") or 0
            used = _link(PROVEN,
                         evidence=f"no `tools.mcp__{name.replace('-', '_')}__*` call in "
                         f"{scanned} Codex session(s), in a corpus where other servers are "
                         f"detected this way",
                         strength="observed", value=0)
        else:
            used = _unproven(
                f"declared for {'/'.join(sorted(harnesses))}, whose traces were not analyzed",
                HARNESS_NOT_TRACED)

        measured = measured_cost.get(name)
        if measured:
            projection = measured["projection"]
            # The measured quantity is tokens per request; the dollar figure is that
            # projected over a corpus, which is a *cumulative* total and belongs in the
            # evidence, not in the value. Carrying the projection as the value put this
            # server on the cost curve as if a whole corpus projection were its per-request
            # price, and kept the one directly measured figure — the schema's own prefix
            # cost — out of the stack total it belongs in. The aggregation trap again: a
            # corpus total is not a unit cost.
            cost = _link(PARTIAL,
                         evidence=f"schema {projection['schema_tokens']} tokens measured in "
                         f"the request prefix; over this corpus that projects to "
                         f"{projection['total_tokens']:,} tokens, about "
                         f"${projection['estimated_usd']:,.2f} cumulative",
                         strength="observed", value=projection["schema_tokens"],
                         unit="tokens_per_request")
            # Which harness the probe ran under, read from the measurement rather than
            # assumed, so a stack total never adds it to a harness it was not measured in.
            cost["harness"] = (measured.get("measurement") or {}).get("harness")
        elif entry["excluded"]:
            # "Not measured" and "not measurable because you asked us not to touch it" are
            # different facts, and only the second one is the user's own decision.
            cost = _unproven("excluded by request, so no probe was run against it",
                             EXCLUDED_BY_REQUEST)
        else:
            cost = _unproven("tool-schema cost not measured for this server", NOT_MEASURED)

        components.append({
            "kind": "mcp", "name": name,
            "harness": "+".join(sorted(harnesses)),
            "harnesses": sorted(harnesses),
            "excluded_by_request": entry["excluded"],
            "scope": entry["scope"],
            "links": {
                "installed": _link(PROVEN, evidence="; ".join(entry["sources"]),
                                   strength="observed"),
                # A server's schema is loaded whenever the harness starts with it configured;
                # Stage 6 measured exactly that difference, so this is evidence, not inference.
                "loaded": _server_loaded(name, loaded, measured, sessions),
                "used": used,
                "useful": _unproven("no stage measured whether the work would have succeeded "
                                    "without this server", NOT_MEASURED),
                "cost": cost,
                "recommendation": _unproven("withheld: requires the useful link",
                                            REQUIRES_EARLIER_LINK),
            },
        })

    # -- Entity components (skills, commands, agents, hooks): installed is provable, and
    # essentially nothing else is, because entering context leaves no event.
    paths = entity_paths(scan)
    # Which harnesses each entity name is installed under. A name present under several is
    # one a by-name invocation record cannot distinguish between.
    installed_under: dict[tuple[str, str], set[str]] = {}
    for kind, rollup in (scan.get("observed", {}).get("components") or {}).items():
        for qualified in rollup.get("names") or []:
            _, one_harness, entity = qualified.split("/", 2)
            installed_under.setdefault((kind, entity), set()).add(one_harness)
    installed_under = {k: v for k, v in installed_under.items() if len(v) > 1}

    for kind, rollup in (scan.get("observed", {}).get("components") or {}).items():
        for qualified in rollup.get("names") or []:
            scope, harness, name = qualified.split("/", 2)
            components.append({
                "kind": kind, "name": name, "harness": harness, "harnesses": [harness],
                "scope": scope,
                # The installed file, which is what distinguishes two copies of one name.
                # Attribution of evidence is by path wherever a path exists; only a `Skill`
                # call forces by-name attribution, which is why duplicates stay `partial`.
                "path": paths.get(qualified),
                "links": {
                    "installed": _link(PROVEN, evidence=f"{kind} entity on disk",
                                       strength="observed"),
                    "loaded": _entity_loaded(name, kind, harness, loaded, traced_harnesses,
                                             sessions, codex_listing,
                                             paths.get(qualified)),
                    "used": _entity_used(name, kind, harness, invoked, traced_harnesses,
                                         sessions, codex_usage, qualified,
                                         installed_under.get((kind, name))),
                    "useful": _entity_useful(kind, paths.get(qualified), useful_skills),
                    "cost": _entity_cost(kind, paths.get(qualified), measured_skills),
                    "recommendation": _unproven("withheld: requires the useful link",
                                                REQUIRES_EARLIER_LINK),
                },
            })

    components += _instruction_components(scan, traces, evaluation, useful)
    _attach_carried_tokens(components, traces, codex_listing)
    return components


def _claude_requests_by_cwd(traces: dict) -> dict:
    """Requests per directory, from the per-project breakdown `collect` already keeps.

    Only present on a whole-environment collect; a single-root one has nothing to split, and
    for that case the environment total *is* the project total.
    """
    out: dict[str, int] = {}
    for entry in ((traces.get("observed") or {}).get("by_project") or {}).values():
        cwd = entry.get("recorded_cwd")
        if isinstance(cwd, str):
            out[cwd] = out.get(cwd, 0) + (entry.get("assistant_messages") or 0)
    return out


def _requests_elsewhere(by_directory: dict, directory: Path) -> list[dict]:
    """Requests recorded for a same-named project at a *different* path.

    Transcripts are filed by working directory, so moving a project leaves its history under
    the old path. Attributing a project-scoped file by its current path alone then reports a
    small figure, or zero, for a file that was carried by thousands of requests — the
    "clean-looking answer that is wrong" the relocation check already warns about, which is
    switched off for a whole-environment pass because nothing is lost for *discovery*.
    Attribution is a different question, and there it does matter.

    These are reported beside the figure and never added to it. A directory with the same
    final segment is a reason to ask, not proof that two paths are one project, and this
    function cannot tell a move from two unrelated projects that happen to share a name.
    """
    out = []
    for cwd, n in sorted(by_directory.items(), key=lambda kv: -(kv[1] or 0)):
        candidate = Path(cwd)
        if candidate == directory or not isinstance(n, int) or n <= 0:
            continue
        target = directory.parts
        if candidate.parts[:len(target)] == target:
            continue  # already counted: this is inside the project
        if candidate.name == directory.name:
            out.append({"cwd": cwd, "requests": n})
    return out


def _requests_under(by_directory: dict, directory: Path) -> int:
    """Requests made in a directory or anywhere below it.

    A project's sessions are not all recorded at its root — a session started in a
    subdirectory records that subdirectory — so an exact match would report zero for a
    project that was worked in every day. The containment test is on path parts rather than
    string prefix, so `/a/b` does not swallow `/a/bc`.
    """
    total = 0
    target = directory.parts
    for cwd, n in by_directory.items():
        parts = Path(cwd).parts
        if parts[:len(target)] == target:
            total += n if isinstance(n, int) else 0
    return total


def _attach_carried_tokens(components: list[dict], traces: dict,
                           codex_listing: dict | None) -> None:
    """Turn a per-request rate into what this environment actually carried.

    A cost of "N tokens per request" is a rate, not a figure about anyone's stack, and the
    stage that reports it said so: the withheld reason named a *request volume* as the
    missing input. That volume is in the corpus and costs nothing — one model response is
    one request — so leaving it unmultiplied was asking the user to pay for a measurement
    that had already been taken.

    Both counts exclude the sessions this tool wrote itself, because a probe issues
    requests too and billing them to the person being profiled is the same defect the
    collector was already fixed for. Two independent walkers agree on them: the collector
    and the normalizer arrive at the same count of profiler sessions, and the Codex rollup
    and the normalizer arrive at the same Codex request count independently.
    """
    requests = {
        # Already self-generated-aware: `collect` excludes those sessions entirely.
        "claude_code": (traces.get("observed") or {}).get("assistant_messages"),
        "codex": (codex_listing or {}).get("requests"),
    }
    by_directory = {
        "claude_code": _claude_requests_by_cwd(traces),
        "codex": (codex_listing or {}).get("requests_by_cwd") or {},
    }
    for component in components:
        cost = component["links"]["cost"]
        value = cost.get("value")
        if cost.get("unit") != "tokens_per_request" or not isinstance(value, (int, float)):
            continue
        # The harness the probe actually ran under, when the artifact recorded it. A rate
        # measured under one harness must not be multiplied by another's traffic, and a
        # component available to both does not say which one the figure belongs to.
        harness = cost.get("harness") or component["harness"]
        n = requests.get(harness)
        if not isinstance(n, int):
            cost["requests_observed"] = None
            cost["tokens_carried"] = None
            cost["carried_blocker"] = tr(
                en=f"the rate was measured under {harness}, which this corpus has no "
                   f"separate request count for, so multiplying it would mix traffic",
                zh=f"这个速率是在 {harness} 下测的，本语料没有它单独的请求数，"
                   f"乘起来会把两边的流量混在一起")
            continue
        # A project-scoped file is only in the prefix of requests made inside that project,
        # so the environment's total is the wrong multiplier for it. Using it charged one
        # project's CLAUDE.md to every request in the corpus, and the largest "carried"
        # figure in a real report belonged to a project with **no** requests at all — a file
        # that was never loaded once, reported as the heaviest thing in the stack.
        elsewhere: list[dict] = []
        if component.get("scope") == "project" and component.get("path"):
            directory = Path(component["path"]).parent
            n = _requests_under(by_directory.get(harness) or {}, directory)
            elsewhere = _requests_elsewhere(by_directory.get(harness) or {}, directory)
            cost["requests_scope"] = "project"
            cost["requests_basis"] = tr(
                en="requests recorded in this project's own directory; the environment "
                   "total would charge this file to sessions that never loaded it",
                zh="只数这个项目目录下记录到的请求；用全环境的总数，会把这个文件算到"
                   "从没载入过它的会话上")
        else:
            cost["requests_scope"] = component.get("scope") or "user"
        cost["requests_observed"] = n
        cost["tokens_carried"] = int(round(value * n))
        if elsewhere:
            # Beside the figure, never inside it: a name match is a reason to ask whether
            # the project moved, not evidence that it did.
            cost["requests_elsewhere"] = elsewhere
            moved = ", ".join(f"{e['cwd']} ({e['requests']:,})" for e in elsewhere[:3])
            cost["requests_elsewhere_note"] = tr(
                en=f"a same-named project has history at another path, which this figure "
                   f"excludes: {moved}. If the project moved, those requests carried this "
                   f"file too; if they are different projects, they did not — the name is "
                   f"all this can see",
                zh=f"同名项目在另一个路径下还有历史，这个数字没有包含它："
                   f"{moved}。如果是项目搬过家，那些请求也背了这个文件；如果是两个不同的项目，"
                   f"就没有 —— 这里能看到的只有名字")
        if n == 0:
            # Carried nothing is a finding, not a blank. The file is installed and has a
            # measured rate, and this corpus holds no request that ever paid it.
            cost["carried_is_zero_because"] = tr(
                en="no request in this corpus was made inside this project, so this file's "
                   "measured rate was never paid here",
                zh="本语料里没有任何请求发生在这个项目目录下，所以这个文件测出来的速率"
                   "在这里一次都没被付过")
        # A component installed under more harnesses than the one its rate was measured
        # under is carried by traffic this figure does not include. Without the flag the
        # product reads as a total, which is the aggregation mistake this project keeps
        # making: `search-server` is loaded by Codex too, and its Codex rate is unmeasured.
        # `shared` is not a harness with its own traffic — it is where a component sits
        # (`~/.agents`) when both harnesses list it. Reporting "excludes shared" would be
        # meaningless; what is actually excluded is the other harness's requests.
        carried_by = set()
        for h in (component.get("harnesses") or []):
            carried_by.update(("claude_code", "codex") if h == "shared" else (h,))
        unmeasured = carried_by - {harness}
        if unmeasured:
            cost["carried_is_lower_bound"] = True
            cost["carried_excludes"] = sorted(unmeasured)


def unmatched_measurements(scan: dict, evaluation: dict | None,
                           useful: dict | None = None) -> list[dict]:
    """Measurements in a supplied artifact that matched nothing this scan found.

    Every join above is by absolute path or server name, and a measurement the join
    cannot place reads downstream exactly like a measurement nobody ran: the link says
    `unproven` and gives the reason "not measured". That is the one confusion this tool
    exists to prevent, and it is not hypothetical — an artifact records absolute paths, so
    moving machines, renaming a home directory, or anonymizing the file before sharing it
    each break the join while leaving the artifact perfectly well-formed. Naming the
    orphans costs a set lookup and is the whole difference between "you have not measured
    this" and "you measured it, somewhere that is not here".
    """
    def _key(value: str) -> str:
        try:
            return str(Path(value).resolve())
        except OSError:
            return value

    on_disk = {_key(item["path"]) for item in (scan.get("files") or [])
               if item.get("path")}
    servers = {d["name"] for d in (scan.get("mcp_servers") or []) if d.get("name")}

    out: list[dict] = []

    def _add(source: str, kind: str, subject: str) -> None:
        # Only what can actually be checked. A first version blamed the artifact's own
        # recorded root whenever it differed from the scan's, which is wrong in the normal
        # case: an artifact is recorded from a project root and `--all` scans the whole
        # home, so the two roots almost always differ legitimately. Whether the subject
        # exists is a fact; why it does not is a guess, and the two read very differently.
        if kind == "mcp_server":
            reason = tr(
                en="no declaration of this server was found in the scanned configuration, "
                   "so the name in the artifact is not a name this scan saw",
                zh="扫描到的配置里找不到这个服务的声明，产物里的名字不是本次扫描见到的名字")
        elif Path(subject).exists():
            # Worth separating: the measurement is sound and the scan was simply too
            # narrow, which `--all` fixes without spending anything again.
            reason = tr(
                en="the file exists on this machine but falls outside what this scan "
                   "covered; a wider scan would join it",
                zh="这个文件在本机存在，但不在本次扫描覆盖的范围内；扩大扫描范围就能对上")
        else:
            reason = tr(en="there is no file at this path on this machine",
                        zh="本机这个路径上没有文件")
        out.append({"source": source, "kind": kind, "subject": subject, "reason": reason})

    for group, kind in (("instruction_measurements", "instruction_file"),
                        ("skill_measurements", "skill")):
        for measurement in (evaluation or {}).get(group) or []:
            path = measurement.get("file")
            if path and _key(path) not in on_disk:
                _add("evaluation", kind, path)
    for entry in (evaluation or {}).get("evaluations") or []:
        name = entry.get("server")
        if name and name not in servers:
            _add("evaluation", "mcp_server", name)
    for measurement in (useful or {}).get("useful_measurements") or []:
        path = measurement.get("file")
        if path and _key(path) not in on_disk:
            _add("useful", "instruction_file", path)
    return out


# Instruction files sit in the fixed prefix of every request, so they are the largest
# always-on cost in most stacks — and they were missing from the chain entirely. Claude
# Code records them (`attachment` records name the path and content), and for Codex a
# measurement artifact can supply both `loaded` and `cost` by differencing, which is the
# only route to either.
INSTRUCTION_KINDS = {"claude_md": "claude_code", "agents_md": "codex"}


def measurement_conflicts(evaluation: dict | None) -> list[dict]:
    """Files a supplied artifact measured more than once, with different results.

    Exposed beside `unmatched_measurements` for the same reason: a figure that quietly
    depended on which record came last is not something a reader can check.
    """
    def key(value: str) -> str:
        try:
            return str(Path(value).resolve())
        except OSError:
            return value

    _, conflicts = _index_measurements(
        (evaluation or {}).get("instruction_measurements") or [], key,
        (evaluation or {}).get("harness"))
    return conflicts


def _measurement_harness(measurement: dict, header: str | None) -> str | None:
    """Which harness a measurement was taken in.

    Three sources in order, because artifacts written before the label existed must keep
    joining: the measurement's own field, then the artifact header's, then the file's own
    name. The name is a sound last resort rather than a guess — the probe measures a
    `CLAUDE.md` through Claude Code and an `AGENTS.md` through Codex by construction, which
    is why there are two measure functions at all.
    """
    explicit = measurement.get("harness")
    if explicit:
        return explicit
    if header:
        return header
    name = Path(measurement.get("file") or "").name.lower()
    if name == "claude.md":
        return "claude_code"
    if name == "agents.md":
        return "codex"
    return None


def _index_measurements(measurements: list[dict], key,
                        header: str | None = None) -> tuple[dict, list[dict]]:
    """Index measurements by resolved path, reporting any file measured twice differently.

    This was a dict comprehension, so a second measurement of the same file silently
    replaced the first and the figure a reader saw depended on list order. That is not
    hypothetical: re-running the probe produced a figure 25% different from the stored one
    for an unchanged file, and merging the two artifacts would have shown one of them with
    nothing said about the other.

    Precedence stays "later wins", which is what the comprehension already did — the change
    is that it is now visible. Nothing here guesses which is right: artifacts carry no
    timestamp, so "newer" is not recoverable from their contents, and picking the larger or
    the smaller would be inventing a rule. Both figures are reported with what each was
    measured under, and the reader decides whether to re-measure.
    """
    out: dict[tuple, dict] = {}
    conflicts: list[dict] = []
    for measurement in measurements:
        path = measurement.get("file")
        if not path:
            continue
        # Keyed by path *and* harness. Resolving the path is what lets an artifact and a
        # scan agree across a `/var` symlink, but it also collapses a user's own symlink:
        # one real environment points AGENTS.md at CLAUDE.md, so two components and two
        # separately paid measurements landed on one key and the Codex figure was
        # overwritten by the Claude one. Two harnesses reading the same bytes is two facts.
        resolved = (key(path), _measurement_harness(measurement, header))
        previous = out.get(resolved)
        if previous is not None:
            before, after = previous.get("mean_tokens"), measurement.get("mean_tokens")
            if before != after:
                conflicts.append({
                    "file": path,
                    "harness": _measurement_harness(measurement, header),
                    "figures": [
                        {"mean_tokens": before,
                         "measured_under": previous.get("measured_under")},
                        {"mean_tokens": after,
                         "measured_under": measurement.get("measured_under")},
                    ],
                    "used": after,
                    "reason": "the same file was measured twice with different results; the "
                              "later artifact is used. A token figure holds for the build "
                              "and model that produced it, and these artifacts carry no "
                              "timestamp, so neither can be shown to supersede the other.",
                })
        out[resolved] = measurement
    return out, conflicts


def _instruction_label(path: str) -> str:
    """A basename alone is ambiguous across scopes; add the parent directory when needed."""
    p = Path(path)
    home = Path.home()
    parent = p.parent
    if parent == home:
        return f"{p.name} (~)"
    try:
        return f"{p.name} (~/{parent.relative_to(home)})"
    except ValueError:
        return f"{p.name} ({parent.name})"


def _withheld_because(useful_link: dict, cost_link: dict) -> str:
    """Name the links a recommendation is still waiting on.

    A recommendation needs both usefulness and cost. Once one of them is proven, a fixed
    "requires the useful link" string would send the reader to re-measure what they have
    already paid for, so the missing ones are named.
    """
    missing = []
    if useful_link["status"] != PROVEN:
        missing.append("the useful link")
    if cost_link["status"] != PROVEN:
        # Distinguish "never measured" from "measured in tokens": telling someone who has
        # already paid for a token measurement that they need the cost link would send them
        # to re-measure it, when what is actually missing is a request volume.
        # Three states, not two. The request volume used to be the missing input and is
        # now read from the corpus for free, so saying "spend is not measured" would again
        # send someone to buy something they already have. What is left is a price.
        if cost_link.get("unit") != "tokens_per_request":
            missing.append("the cost link")
        elif cost_link.get("tokens_carried") is not None:
            missing.append("a price per token (the tokens carried are counted; what they "
                           "cost depends on a rate this tool will not invent)")
        else:
            missing.append("a cost in money (per-request tokens are measured; spend is not)")
    if not missing:
        # Both links hold. A recommendation is still not emitted here: turning evidence
        # into advice about what to remove needs the user's own weighting, which Stage 7
        # refuses to invent.
        return "withheld: evidence is complete, but the trade-off is the user's to weigh"
    return "withheld: requires " + " and ".join(missing)


def _instruction_components(scan: dict, traces: dict, evaluation: dict | None,
                            useful: dict | None = None) -> list[dict]:
    observed = traces.get("observed") or {}
    sessions = observed.get("sessions_collected", 0)
    instructions = observed.get("instructions_loaded") or {}
    # Keyed by resolved path. A measurement artifact and a scan can spell the same file
    # differently — on macOS `/var` is a symlink to `/private/var` — and an unresolved
    # join silently discards a measurement the user already paid for.
    def _key(value: str) -> str:
        try:
            return str(Path(value).resolve())
        except OSError:
            return value

    measured, conflicts = _index_measurements(
        (evaluation or {}).get("instruction_measurements") or [], _key,
        (evaluation or {}).get("harness"))
    conflict_by_path = {(_key(c["file"]), c.get("harness")): c for c in conflicts}
    useful_measured = {_key(m["file"]): m
                       for m in ((useful or {}).get("useful_measurements") or [])
                       if m.get("file")}

    out = []
    for item in scan.get("files") or []:
        kind = item.get("kind")
        if kind not in INSTRUCTION_KINDS or item.get("origin") != "declared":
            continue
        path = item["path"]
        harness = INSTRUCTION_KINDS[kind]
        record = instructions.get(path)
        # The harness is part of the key: a symlinked AGENTS.md -> CLAUDE.md is two
        # components, each measured in its own harness, and the Codex figure used to
        # be overwritten by the Claude one.
        measurement = measured.get((_key(path), harness))

        if record:
            loaded = _link(PROVEN,
                           evidence=f"an attachment record shows this file entering context "
                                    f"in {record['loads']} of {sessions} sessions "
                                    f"({record['chars']} chars)",
                           strength="observed", value=record["loads"])
        elif measurement and measurement.get("loaded") == "proven":
            # No record exists for Codex instructions; a measured prefix difference is the
            # evidence instead, which is why this link is provable at all.
            loaded = _link(PROVEN, evidence=measurement["loaded_evidence"],
                           strength="observed", value=measurement["mean_tokens"])
        elif not sessions and not measurement:
            loaded = _unproven("no transcript was collected and no measurement was run",
                               NO_TRACES_COLLECTED)
        elif harness == "codex":
            loaded = _unproven("Codex reports no event for instruction loading; a prefix "
                               "measurement would settle it", NOT_MEASURED)
        else:
            loaded = _unproven(f"no attachment record names this file in {sessions} sessions",
                               NOT_MEASURED)

        conflict = conflict_by_path.get((_key(path), harness))
        if measurement and measurement.get("ok"):
            evidence = (f"{measurement['mean_tokens']} tokens added to every "
                        f"request, measured over {len(measurement['differences'])} "
                        f"interleaved pair(s) with spread "
                        f"{measurement['pair_spread_tokens']}")
            if conflict:
                # The spread describes agreement inside one window. Printing it next to a
                # figure another artifact contradicts would read as settled precision.
                others = [str(f["mean_tokens"]) for f in conflict["figures"]]
                evidence += (f"; another supplied measurement of this file reports a "
                             f"different figure ({' vs '.join(others)}), so this number is "
                             f"not settled")
            cost = _link(PARTIAL, evidence=evidence, strength="observed",
                         value=measurement["mean_tokens"], unit="tokens_per_request")
            if conflict:
                cost["conflict"] = conflict
        else:
            cost = _unproven("prefix cost not measured for this file", NOT_MEASURED)

        # An instruction file is never "called", so `used` has no event to observe. What a
        # behavioural separation shows is that the file's content shaped the output, which
        # is the closest thing to exercise that such a file can have. Both links therefore
        # rest on *one* observation, and the evidence string says so — two columns turning
        # green off a single experiment would otherwise read as independent corroboration.
        influence = useful_measured.get(_key(path))
        verdict = (influence or {}).get("useful")
        if verdict == "proven":
            shared = ("the same observation as the useful link, not a second one: "
                      + influence["useful_evidence"])
            used = _link(PROVEN, evidence=shared, strength="observed",
                         value=len(influence["rules_proven"]))
            useful_link = _link(PROVEN, evidence=influence["useful_evidence"],
                                strength="observed",
                                value=len(influence["rules_proven"]),
                                unit="rules_proven")
            # The rule *names*, not just how many. A reader asking "what does useful mean
            # here" needs to see which conventions separated, and the count alone sends them
            # back to the artifact to find out.
            useful_link["rules_proven"] = list(influence["rules_proven"])
        elif verdict == "partial":
            used = _link(PARTIAL,
                         evidence="the same observation as the useful link: "
                                  + influence["useful_evidence"],
                         strength="observed")
            useful_link = _link(PARTIAL, evidence=influence["useful_evidence"],
                                strength="observed")
        elif influence:
            # Measured and refused. A blocker here means the experiment ran its controls
            # and declined to report, which is a different state from never having tried.
            useful_link = _unproven(influence.get("reason") or "the measurement was refused",
                                    influence.get("blocker") or NOT_MEASURED)
            used = _unproven("an instruction file is not invoked, so there is no call to "
                             "observe; the behavioural measurement was refused",
                             NO_EVENT_EMITTED)
        else:
            useful_link = _unproven("no stage measured whether the work would have "
                                    "succeeded without this file", NOT_MEASURED)
            used = _unproven("an instruction file is not invoked, so there is no call "
                             "to observe; influence on behaviour is the useful link",
                             NO_EVENT_EMITTED)

        out.append({
            # Several instruction files share a basename — a user-scope `AGENTS.md` and a
            # `~/.codex/AGENTS.md` are different components — so the name carries enough of
            # the location to tell them apart, as the duplicate-skill rows had to.
            "kind": kind, "name": _instruction_label(path), "harness": harness,
            "harnesses": [harness], "scope": item.get("scope"), "path": path,
            "bytes": item.get("bytes"),
            "links": {
                "installed": _link(PROVEN, evidence=path, strength="observed"),
                "loaded": loaded,
                # An instruction file is not "called"; it applies to every request. Saying
                # `used: 0` would be false, and saying `used: proven` would smuggle in the
                # assumption that loading equals influence.
                "used": used,
                "useful": useful_link,
                "cost": cost,
                # Which link is actually missing now varies per file, so the reason is
                # derived rather than hardcoded: a file whose usefulness is proven but whose
                # cost is not is blocked on cost, and saying "requires the useful link"
                # would send the reader to re-measure something already measured.
                "recommendation": _unproven(_withheld_because(useful_link, cost),
                                            REQUIRES_EARLIER_LINK),
            },
        })
    return out


def _resolve(value: str | None) -> str | None:
    """One spelling for one path, so two indexes can agree on a key.

    Resolution is the whole point. A scan records resolved paths while an artifact records
    whatever path the command was given, and where a temp or home directory is reached
    through a symlink the two spellings differ. Keying on the raw string makes a measurement
    that was paid for join to nothing and read as never attempted, which is the one
    confusion this tool exists to prevent.
    """
    if not value:
        return None
    try:
        return str(Path(value).resolve())
    except OSError:
        return value


def _resolved_dir(path: str | None) -> str | None:
    """The key both artifact indexes use: the resolved directory holding an entry file."""
    return _resolve(str(Path(path).parent)) if path else None


def _entity_useful(kind: str, path: str | None, useful_skills: dict[str, dict]) -> dict:
    """Whether a compliance probe proved this entity changes verifiable output.

    PROVEN only when the probe's own verdict is `proven`. It runs a negative control and an
    anti-recitation guard, and a run that fails either is reported as unproven rather than
    as a weaker pass, so a failed control must not arrive here as partial credit.

    `used` is deliberately left alone, unlike the instruction-file route which proves both
    links from one observation. An instruction file is always-on: installed means loaded
    means in effect in every session, so a separation is also evidence it was used. A skill
    is not always-on — it enters context on a description match — so the probe proves it
    *can* change the output, not that the user's own sessions used it. Writing the probe's
    arm into `used` would turn this tool's own run into evidence about the user's work.
    """
    measurement = useful_skills.get(_resolved_dir(path)) if path else None
    if not measurement:
        return _unproven("not measured", REQUIRES_EARLIER_LINK)
    if measurement.get("useful") != "proven":
        return _unproven(measurement.get("reason")
                         or "a compliance probe ran and did not separate the arms",
                         NOT_MEASURED)
    rules = list(measurement.get("rules_proven") or [])
    link = _link(PROVEN, evidence=measurement.get("useful_evidence") or "", value=len(rules),
                 strength="observed", unit="rules_proven")
    # The rule names, not just how many: a reader asking what "useful" means here needs to
    # see which conventions separated without going back to the artifact.
    link["rules_proven"] = rules
    return link


def _entity_cost(kind: str, path: str | None, measured_skills: dict[str, dict]) -> dict:
    """The per-request cost of one entity, from a `measure-skills` artifact.

    PARTIAL rather than PROVEN, for two reasons that both have to travel with the figure.
    The skill arrives through the plugin channel, which adds a few tokens of its own to the
    listing entry; and it is a *marginal* cost against whatever baseline the measurement
    ran against, so it is what adding this skill costs rather than its share of a measured
    total.

    What the figure settles is bigger than the figure. A measured skill comes out an order
    of magnitude cheaper per request than its body is long, so the body does not enter the
    prefix; the description does. That had been recorded as a blind spot ("是否进了上下文")
    and is now answered for the listing stage, which is what a stack's always-on cost is
    made of.
    """
    measurement = measured_skills.get(_resolved_dir(path)) if path else None
    if measurement:
        caveats = "through the plugin channel, against the measurement's own baseline"
        if measurement.get("measured_on"):
            caveats += f"; measured on the byte-identical copy at {measurement['measured_on']}"
        if measurement.get("ambiguous_with"):
            # Two installed copies, one listed name. Immaterial here as it happens — the
            # copies measured within 3 tokens of each other — but the figure still belongs
            # to a file rather than to the listed skill, and must say so.
            caveats += "; this name is installed twice and which copy the harness loads " \
                       "is not recoverable"
        return _link(PARTIAL,
                     evidence=f"{measurement['mean_tokens']} tokens added to every request, "
                              f"measured over {len(measurement['differences'])} interleaved "
                              f"pair(s) with spread {measurement['pair_spread_tokens']}, "
                              f"{caveats}",
                     strength="observed", value=measurement["mean_tokens"],
                     unit="tokens_per_request", harness=measurement.get("harness"))
    if kind == "skills":
        return _unproven("prefix cost not measured for this skill; `measure-skills` can "
                         "settle it", NOT_MEASURED)
    return _unproven(f"{kind} context cost not measured; it is only paid when the entity is "
                     "actually loaded, which is itself unproven", REQUIRES_EARLIER_LINK)


def _server_loaded(name: str, loaded: dict, measured: dict | None, sessions: int) -> dict:
    """Whether a server's instructions and schema actually entered the context window."""
    count = loaded.get(name, 0)
    if count:
        return _link(PROVEN,
                     evidence=f"its instructions were loaded into context in {count} of "
                     f"{sessions} sessions (attachment record)",
                     strength="observed", value=count)
    if measured:
        # The probe measures the live prefix, so it holds even with no transcripts at all.
        return _link(PROVEN, evidence="schema measurably present in the request prefix "
                     "(stage 6 paired probe)", strength="observed")
    if not sessions:
        return _unproven("no transcript was collected for this root", NO_TRACES_COLLECTED)
    return _unproven("no record shows this server entering the context window", NOT_MEASURED)


def _entity_loaded(name: str, kind: str, harness: str, loaded: dict, traced: set,
                   sessions: int, listing: dict | None = None, path: str | None = None) -> dict:
    """Whether an entity was listed into context, which the harness records for some kinds.

    An earlier draft called this permanently unobservable for every kind. It is not: a
    `skill_listing` attachment names every skill offered in a session. But that record
    covers *skills only* — reading a command's absence from it as "never loaded" would
    manufacture a proof out of a record that was never about commands.

    Codex records the same thing in `state.host_skills.entries`, keyed by the file's
    full path. That is stronger than the Claude listing, which names a skill without saying
    which root it came from — `demo-skill` exists under two here. It is also narrower: only a
    few sessions carry the record at all, so the evidence states how many did rather than
    generalizing to the corpus.
    """
    if kind not in LISTED_KINDS:
        return _unproven(f"no record names a {kind[:-1]} entering context; the skill listing "
                         "does not cover this kind", NO_EVENT_EMITTED)
    listed = (listing or {}).get("sessions_listing_path", {}).get(path or "", 0)
    if listed:
        carrying = listing.get("sessions_carrying_record") or listed
        return _link(PROVEN,
                     evidence=f"Codex's own skill listing names this exact file in {listed} "
                     f"of the {carrying} session(s) that recorded one, as a one-line "
                     f"description rather than its full body",
                     strength="observed", value=listed)
    # Load evidence comes from the traced harness's own records, and listings name a skill
    # without qualifying it. The same name can exist for both harnesses, so an untraced
    # harness's entity must not inherit the traced one's loads.
    if harness not in traced:
        return _unproven(f"declared for {harness}, whose traces were not analyzed",
                         HARNESS_NOT_TRACED)
    count = loaded.get(name, 0)
    if count:
        return _link(PROVEN,
                     evidence=f"listed into context in {count} of {sessions} sessions, as a "
                     "one-line description rather than its full body",
                     strength="observed", value=count)
    if not sessions:
        return _unproven("no transcript was collected for this root", NO_TRACES_COLLECTED)
    return _link(PROVEN, evidence=f"never listed into context in {sessions} traced sessions",
                 strength="observed", value=0)


def _entity_used(name: str, kind: str, harness: str, invoked: dict, traced: set, sessions: int,
                 codex_usage: dict | None = None, key: str | None = None,
                 ambiguous: set | None = None) -> dict:
    """Whether a skill-class entity was used, which is only partly observable.

    An explicit invocation *is* a tool call (`Skill`) and names the entity, so a positive is
    solid. A negative is not: the same entity can enter context by description match, and
    that leaves no record. So "never invoked" is reported as `partial`, never as a proven
    zero — the asymmetry is the finding.

    Codex has no such tool call at all; it uses a skill by reading `SKILL.md` through its
    shell. That read is evidence, but weaker, because authoring looks identical — so it is
    `partial` however many reads there were, and the evidence says why.
    """
    calls = invoked.get(name, 0)
    if calls:
        # A `Skill` call records the skill's *name* and nothing about which directory it
        # was loaded from — verified against every such call in the corpus, whose input is
        # only `{skill, args}`. So when the same name is installed under more than one
        # root, these calls cannot be attributed to one of them, and crediting each with
        # the full count reports one invocation as several. The count is still evidence
        # that the skill was used; which copy ran is the part that is unknown.
        if ambiguous:
            return _link(PARTIAL,
                         evidence=f"a {kind[:-1]} of this name was invoked {calls} time(s), "
                         f"but it is installed under {len(ambiguous)} roots "
                         f"({', '.join(sorted(ambiguous))}) and nothing records which copy "
                         f"ran: the call carries only the name, and the harness's own "
                         f"listing shows each name once, having already deduplicated the "
                         f"roots. So the invocations are real and this copy's share of "
                         f"them is unknown",
                         strength="observed", value=calls, blocker=NO_EVENT_EMITTED)
        return _link(PROVEN, evidence=f"invoked {calls} time(s) via an explicit {kind[:-1]} call",
                     strength="observed", value=calls)
    reads = (codex_usage or {}).get("reads", {}).get(key or "", 0)
    if reads:
        spread = codex_usage["sessions"].get(key, 0)
        return _link(PARTIAL,
                     evidence=f"its installed SKILL.md was read {reads} time(s) across "
                     f"{spread} session(s) that never wrote to it, which is how Codex uses "
                     f"a skill; authoring the file looks the same, so this is not proof",
                     strength="observed", value=reads, blocker=NO_EVENT_EMITTED)
    if (codex_usage or {}).get("authoring_sessions", {}).get(key or ""):
        edits = codex_usage["authoring_sessions"][key]
        return _unproven(f"the only session(s) touching its SKILL.md also wrote to it "
                         f"({edits}), which is authoring the skill rather than using it",
                         NO_EVENT_EMITTED)
    if harness not in traced:
        return _unproven(f"declared for {harness}, whose traces were not analyzed",
                         HARNESS_NOT_TRACED)
    if not sessions:
        return _unproven("no transcript was collected for this root", NO_TRACES_COLLECTED)
    return _link(PARTIAL,
                 evidence=f"never explicitly invoked in {sessions} traced sessions, but "
                 f"auto-loading a {kind[:-1]} into context leaves no record, so this is not "
                 "evidence that it went unused",
                 strength="observed", value=0, blocker=NO_EVENT_EMITTED)


# Where the cost curve bends, per unit. `1/(1 + value/scale)` is monotone in value for any
# positive scale, so the constant moves scores but never their order: a ranking is the same
# whichever of these is chosen. What it must not do is read a number in one unit on a curve
# calibrated for another, which is how a token count once scored as though it were dollars.
#
# `usd` is per request, not a total. `tokens_per_request` was a round 1,000 guess and is now
# calibrated against the per-request blocks that have actually been measured: one skill
# added back through --plugin-dir at the low end, a large instruction file at the high end,
# a server's tool schema and a shared skill listing in between — a set spanning more than an
# order of magnitude.
#
# The scale is the point where the curve reads 0.5, so putting it near the centre of that
# distribution is what makes the signal discriminate instead of saturating: at 1,000 every
# measured component bunched at the top of the range. 650 is that set's geometric mean to
# the nearest 50, chosen geometric because the values span more than an order of magnitude,
# where an arithmetic centre would be dragged by the largest block.
#
# The individual measurements are deliberately not listed here. They are one private
# stack's component sizes, and what a reader needs is the constant and how it was chosen,
# not an inventory of someone else's environment.
#
# This is a calibration, not a law. It moves scores and cannot move their order —
# `1/(1 + v/s)` is monotone in `v` for any positive `s`, which a test asserts across three
# scales spanning six orders of magnitude.
COST_SCALES = {
    None: 1.0,
    "usd": 1.0,
    "tokens_per_request": 650.0,
}
# What the figure above was computed from, so a later reader can check it rather than
# trust it. Updating the scale means updating this list and the test that reads it.
COST_SCALE_CALIBRATION = {
    "tokens_per_request": {
        "derivation": "the geometric mean, to the nearest 50, of every per-request token "
                      "block this project has measured — geometric rather than arithmetic "
                      "because the blocks span more than an order of magnitude, where a "
                      "mean is dragged by the largest one",
        "blocks": "a single skill added through --plugin-dir at the low end, a large "
                  "instruction file and a shared skill listing at the high end",
        # The constant is published and its inputs are not, which is a trade this project
        # refuses everywhere else. It is sound here for one specific reason: the cost signal
        # is monotone in the measured value, so this constant moves scores and *cannot* move
        # their order — a test asserts that across six orders of magnitude. The inputs are
        # one private stack's component sizes, and nothing a reader concludes depends on
        # them. Re-derive the scale from your own artifacts with `scale_from`.
        "from": "measurement artifacts, which are a profile of one machine and are kept out "
                "of this repository — so the samples are not listed here either",
    },
}


def scale_from(samples) -> int:
    """Re-derive a cost scale from measured per-request blocks.

    Published so the constant above is reproducible from a reader's own measurements rather
    than taken on trust. The rounding is cosmetic; what matters is landing near the centre
    of the distribution, because the scale is the point where the curve reads 0.5 and a
    scale far above the blocks makes every component score the same.
    """
    import statistics

    values = [float(v) for v in samples if v and v > 0]
    if not values:
        raise ValueError("a scale cannot be derived from no measurements")
    return int(round(statistics.geometric_mean(values) / 50) * 50)


def comparable_cost_units(components: list[dict]) -> set:
    """The cost units in play across a set of components.

    More than one means their scores are not comparable: the cost signal is a position on a
    curve, and two curves calibrated for different units share an axis only by coincidence.
    A ranking spanning both would look ordinary and mean nothing.
    """
    return {c["links"]["cost"].get("unit") for c in components
            if c["links"]["cost"]["status"] != UNPROVEN}


def _coverage(component: dict) -> dict:
    links = component["links"]
    proven = [k for k in SCORED_LINKS if links[k]["status"] == PROVEN]
    partial = [k for k in SCORED_LINKS if links[k]["status"] == PARTIAL]
    missing = [k for k in REQUIRED_FOR_SCORE if links[k]["status"] == UNPROVEN]
    # A cost in an unrecognized unit cannot be placed on the curve at all, so it blocks.
    # Units this tool knows how to scale do not: requiring dollars blocked every component
    # in the environment, because no route to a per-component cost reports money. Tokens
    # per request are a real cost measure; what was wrong was a curve calibrated for
    # dollars reading a token count as a dollar figure.
    if (links["cost"]["status"] != UNPROVEN
            and links["cost"].get("unit") not in COST_SCALES
            and "cost" not in missing):
        missing = [*missing, "cost"]
    return {
        "proven": proven,
        "partial": partial,
        "unproven": [k for k in SCORED_LINKS if links[k]["status"] == UNPROVEN],
        "fraction": round((len(proven) + 0.5 * len(partial)) / len(SCORED_LINKS), 3),
        "scorable": not missing,
        "blocking": missing,
        "cost_not_in_dollars": links["cost"].get("unit") not in (None, "usd"),
        "cost_unit": links["cost"].get("unit"),
    }


def score_component(component: dict, profile: str, profiles: dict | None = None) -> dict:
    """Score one component under one preference profile, or decline to.

    Declining is the common case in a real environment and is returned as a result, not an
    error: the caller gets the blocking links by name.
    """
    available = profiles if profiles is not None else PREFERENCE_PROFILES
    if profile not in available:
        raise ValueError(f"unknown preference profile: {profile}")
    coverage = _coverage(component)
    if not coverage["scorable"]:
        unit = coverage.get("cost_unit")
        return {
            "scored": False,
            "reason": (f"cost is measured in {unit!r}, a unit this tool has no scale for, "
                       f"so the measurement cannot be placed on the cost curve"
                       if unit not in COST_SCALES and coverage["blocking"] == ["cost"]
                       else "required evidence is missing"),
            "blocking_links": coverage["blocking"],
            "coverage": coverage["fraction"],
        }

    weights = available[profile]["weights"]
    links = component["links"]
    # Signals are normalized 0-1 where 1 is "good"; cost is inverted because less is better.
    # `useful` is read from the *status*, not from the value: the value is a count of rules
    # proven, and feeding a count into a 0-1 signal pushed the weighted total above 1 for a
    # two-rule file. The measurement establishes that influence exists, not how much, so a
    # proven link is 1.0 and nothing here pretends to a degree it did not measure.
    cost_unit = links["cost"].get("unit")
    scale = COST_SCALES[cost_unit]
    signals = {
        "used": 1.0 if (links["used"]["value"] or 0) > 0 else 0.0,
        "useful": {PROVEN: 1.0, PARTIAL: 0.5}.get(links["useful"]["status"], 0.0),
        "cost": 1.0 / (1.0 + float(links["cost"]["value"] or 0.0) / scale),
    }
    total = sum(weights[k] * signals[k] for k in weights)
    return {
        "scored": True,
        "profile": profile,
        "score": round(total, 4),
        "signals": {k: round(v, 4) for k, v in signals.items()},
        "weights": weights,
        "coverage": coverage["fraction"],
        "strength": "estimated",
        # Carried so a reader can tell which curve produced the cost signal, and so two
        # scores built on different units are not silently compared.
        "cost_unit": cost_unit,
        "cost_scale": scale,
        "caveat": ("the weights are a named hypothesis, not preferences supplied by anyone"
                   if profiles is None else "the weights were supplied by the user"),
    }


def sensitivity(component: dict, profiles: dict | None = None) -> dict:
    """Run a component under every profile and report whether the answer depends on which.

    A score that survives only one profile is a property of the weights, not of the stack.
    """
    available = profiles if profiles is not None else PREFERENCE_PROFILES
    results = {p: score_component(component, p, profiles) for p in available}
    scored = {p: r["score"] for p, r in results.items() if r["scored"]}
    if not scored:
        return {"profiles": results, "stable": None,
                "note": "not scorable under any profile, so sensitivity is undefined"}
    spread = max(scored.values()) - min(scored.values())
    return {
        "profiles": results,
        "spread": round(spread, 4),
        # A spread this wide means the profile, not the evidence, is driving the number.
        "stable": spread < 0.25,
        "note": "scores agree across profiles" if spread < 0.25 else
        "the profile chosen changes the answer more than the evidence does",
    }


def stack_cost(components: list[dict], shared: dict | None = None) -> dict:
    """What carrying this environment costs per request, as a floor.

    Per-*component* cost is unattributable for most of the stack and always will be for a
    shared block: one skill listing covers twenty skills and dividing it would invent the
    attribution this project refuses to invent. But the stack-level question never needed
    that division. "What does my environment cost per request" is a sum of disjoint prefix
    blocks, and summing them attributes nothing.

    Three rules keep the sum honest:

    1. **Per harness.** A Codex request does not carry Claude Code's CLAUDE.md. Totalling
       across harnesses would describe a session nobody ever runs.
    2. **One unit.** Only `tokens_per_request` values are summed. The Claude skill listing
       is recorded in *characters* because no tokenizer is run, and characters cannot be
       added to tokens. It is reported beside the total, never inside it.
    3. **A floor, never a total.** Unmeasured components can only add, so the sum is a
       lower bound. Calling it "the cost" would be the strongest claim in the project
       resting on the least complete evidence.

    The additivity itself is an assumption and is returned as one: the blocks are distinct
    sections of the prefix, which makes summing reasonable, but no measurement here shows
    that adding one changes nothing about another.
    """
    by_harness: dict[str, dict] = {}

    def bucket(harness: str) -> dict:
        return by_harness.setdefault(harness, {
            "contributions": [], "not_summable": [], "components_total": 0,
            "components_measured": 0, "unmeasured": [],
        })

    for component in components:
        harnesses = component.get("harnesses") or [component.get("harness") or "unknown"]
        # `shared` marks `~/.agents/skills`, which both harnesses read. It is not a harness,
        # so a floor for it is a per-request figure for a session nobody runs. A measured
        # shared component belongs to the harness it was measured under: adding it to both
        # would double-count a prefix block Codex already pays for through its own skill
        # listing, and leaving it in a `shared` bucket drops it from the one floor where it
        # is genuinely paid.
        measuring = (component["links"]["cost"] or {}).get("harness")
        if harnesses == ["shared"] and measuring:
            harnesses = [measuring]
        for harness in harnesses:
            entry = bucket(harness)
            entry["components_total"] += 1
            link = component["links"]["cost"]
            if link["status"] == UNPROVEN or link["value"] is None:
                entry["unmeasured"].append(component["name"])
                continue
            # A component available to several harnesses was measured under *one* of them.
            # Adding that figure to every harness's floor asserts the same prefix cost in
            # places it was never observed — search-server, measured through Claude Code,
            # was being added to Codex's floor as well, where it is reached through a
            # different plugin entirely.
            measured_under = link.get("harness")
            if len(harnesses) > 1 and measured_under and measured_under != harness:
                entry["not_summable"].append({
                    "component": component["name"],
                    "why": f"measured under {measured_under}, not under {harness}; the same "
                           f"server reached through another harness is not the same prefix",
                })
                continue
            if link.get("unit") == "tokens_per_request":
                entry["components_measured"] += 1
                entry["contributions"].append({
                    "name": component["name"], "tokens": link["value"], "shared": False,
                    "kind": component.get("kind"), "path": component.get("path"),
                    # A project-scoped block is in the prefix only inside its own project.
                    # Summing the blocks of *different* projects asserts a request that
                    # loaded two projects' instructions at once, which cannot happen: you
                    # are never in two working directories.
                    "scope": component.get("scope"),
                    # Whether this block was ever actually paid here. A project with no
                    # recorded request is a real case, and naming its file as this
                    # environment's worst case would be a worst case that never happened.
                    "requests": link.get("requests_observed"),
                })
            else:
                entry["not_summable"].append({
                    "name": component["name"], "value": link["value"],
                    "unit": link.get("unit") or "usd",
                })

    if shared and shared.get("skill_listing_tokens"):
        tokens = shared["skill_listing_tokens"]
        entry = bucket(shared.get("harness") or "codex")
        entry["contributions"].append({
            "name": tr(en="skill listing (shared)", zh="skill 列表（共享）"),
            "tokens": min(tokens),
            "tokens_max": max(tokens),
            "shared": True,
            "covers": len(shared.get("skills_covered") or []),
        })
        # A shared block is not one component's cost, so it is not counted toward the
        # per-component coverage fraction; it is a block the whole harness pays for.
        for name in shared.get("skills_covered") or []:
            entry["unmeasured"] = [u for u in entry["unmeasured"] if u != name]

    # A skill installed under two roots is two components and *one* listed skill. Summing
    # both copies inflated this floor by about 30% — the aggregation trap for the
    # sixth time, and the first time in the unit "components vs. what the harness lists".
    # The lower figure is kept because this is a floor, and the pair is recorded so the
    # choice is visible rather than silent.
    for entry in by_harness.values():
        deduplicated: dict[tuple, dict] = {}
        for contribution in entry["contributions"]:
            key = (contribution.get("kind"), contribution["name"], contribution["shared"])
            kept = deduplicated.get(key)
            if kept is None:
                deduplicated[key] = contribution
                continue
            low, high = sorted((kept, contribution), key=lambda c: c["tokens"])
            low = dict(low)
            low["duplicate_of"] = {
                "tokens": high["tokens"], "path": high.get("path"),
                "why": "the same name is installed twice and the harness lists it once, so "
                       "only one copy is in any prefix; the lower figure keeps this a floor",
            }
            deduplicated[key] = low
            entry["components_measured"] -= 1
        entry["contributions"] = list(deduplicated.values())

    out = {}
    for harness, entry in sorted(by_harness.items()):
        if not entry["contributions"]:
            out[harness] = {
                **entry, "per_request_tokens_floor": None,
                "blocker": "no component of this harness has a cost in tokens per request",
            }
            continue
        # Split before summing. "Every request carries at least N" has to be true of every
        # request, and adding two projects' instruction files made it true of none: a
        # request made in one project does not carry the other's. The always-on blocks are
        # the floor; the project blocks are reported separately, with the largest named,
        # because the honest statement is "in the heaviest project it becomes M".
        always = [c for c in entry["contributions"] if c.get("scope") != "project"]
        per_project = sorted((c for c in entry["contributions"] if c.get("scope") == "project"),
                             key=lambda c: -c["tokens"])
        low = sum(c["tokens"] for c in always)
        high = sum(c.get("tokens_max", c["tokens"]) for c in always)
        # The heaviest block that was *actually carried*. The plain heaviest can be a file
        # in a project with no recorded request, and naming that as this environment's worst
        # case states a cost nobody ever paid.
        heaviest = next((c for c in per_project if (c.get("requests") or 0) > 0), None)
        out[harness] = {
            **entry,
            "per_request_tokens_floor": low,
            "per_request_tokens_floor_high": high if high != low else None,
            # What a request made inside a project carries on top, never added into the
            # floor above. Each entry is an alternative to the others, not an addend.
            "project_blocks": [{"name": c["name"], "tokens": c["tokens"],
                                "path": c.get("path"), "requests": c.get("requests")}
                               for c in per_project],
            "in_heaviest_project": (low + heaviest["tokens"]) if heaviest else None,
            "heaviest_project_block": heaviest["name"] if heaviest else None,
            "basis": "floor",
            "assumptions": [
                tr(en="these are non-overlapping blocks of the prefix, so they are added "
                      "together; but \"adding one block does not affect another\" is "
                      "itself an assumption and has not been tested",
                   zh="这些是前缀里互不重叠的几块，所以相加；但「加一块不会影响另一块」"
                      "本身是假设，没有测过"),
                *([tr(en=f"the floor is the always-on blocks only. "
                         f"{count(len(per_project), 'project-scoped block is', 'project-scoped blocks are')} "
                         f"excluded from it, because a request made in one project does not "
                         f"carry another project's instructions — they are alternatives, "
                         f"not addends",
                      zh=f"这个下限只含常驻的那几块。另有 {len(per_project)} 块是项目级的，"
                         f"没有算进来 —— 在一个项目里发的请求不会背上另一个项目的规范，"
                         f"它们之间是互斥的，不能相加")] if per_project else []),
                tr(en=f"only {count(len(entry['contributions']), 'item has', 'items have')} "
                      f"a measured tokens-per-request; the other "
                      f"{count(len(entry['unmeasured']), 'component is', 'components are')} "
                      f"unmeasured, and what is unmeasured can only push the real figure up",
                   zh=f"只有 {len(entry['contributions'])} 项有「每次请求多少 token」的实测值，"
                      f"其余 {len(entry['unmeasured'])} 个组件没测，没测的只会让真实值更高"),
            ],
        }
    return out


# Why `useful` cannot be proven, where it cannot. The distinction earns its place because
# `unproven` reads as a to-do list: a reader who cannot tell "nobody has measured this yet"
# from "no method here can measure it" will put money behind the second kind. Every reason
# below is decided from what is already on disk, with no harness run and nothing spent.
NO_TEXT = "no_component_text"
EMPTY_TEXT = "empty_component"
NO_RULE = "no_stated_output_rule"
HAS_RULE = "states_checkable_rule"


def _component_text(component: dict) -> tuple[str, str | None]:
    """The text a component states its own rules in, or why there is none to read."""
    raw = component.get("path")
    if not raw:
        # A server is declared in configuration and ships no prose, so there is no sentence
        # to hold it to. That is a property of the component, not a gap in the tool.
        return NO_TEXT, None
    path = Path(raw)
    if path.is_dir():
        for name in ("SKILL.md", "CLAUDE.md", "AGENTS.md"):
            if (path / name).is_file():
                path = path / name
                break
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return NO_TEXT, None
    # An empty file is installed, is loaded, and still cannot state anything. Filing it with
    # the files that have prose but no quotable rule would hide a different fact.
    return (EMPTY_TEXT, None) if not text.strip() else (HAS_RULE, text)


def stated_rules(component: dict) -> tuple[str, list[str]]:
    """Rules this component states that a machine can check, and how that was decided."""
    from .useful_probe import scorable_checks

    status, text = _component_text(component)
    if text is None:
        return status, []
    stated, _ = scorable_checks(text)
    names = [check["name"] for check in stated]
    return (HAS_RULE if names else NO_RULE), names


def useful_measurability(component: dict, siblings: list[dict] | None = None) -> dict:
    """Can `useful` be proven for this component by any method this tool has?

    Answered for free, from the component's own text, because the alternative is paying to
    discover it. The rule library is the probe's own `scorable_checks` rather than a second
    one written here: a hand-rolled set of patterns was tried first and matched a sentence
    about what a *tool returns* and a CLI flag inside an example command as though they were
    rules about the agent's output, which would have put a probe behind two components that
    state nothing checkable at all.
    """
    if component["links"]["useful"]["status"] == PROVEN:
        return {"status": "proven", "reason": None, "rules": []}
    reason, rules = stated_rules(component)
    if not rules:
        return {"status": "not_measurable", "reason": reason, "rules": []}
    out = {"status": "measurable", "reason": None, "rules": rules}
    # The arm injects one copy. Where the installs diverge the harness lists the name once,
    # which copy it loaded is not recorded, and a result would then be about the copy that
    # was injected rather than about this component as it is installed.
    mine = tuple(sorted(rules))
    others = [s for s in (siblings or [])
              if s is not component and s.get("name") == component.get("name")
              and s.get("kind") == component.get("kind")]
    if any(tuple(sorted(stated_rules(s)[1])) != mine for s in others):
        out["copies"] = len(others) + 1
        out["result_is_about_injected_copy"] = True
    return out


def _measurability_totals(components: list[dict]) -> dict:
    """Stack-wide counts for `useful` measurability, kept beside the per-component reasons."""
    totals = {"proven": 0, "measurable": 0, "not_measurable": 0, "reasons": {}}
    for component in components:
        entry = component.get("useful_measurability") or {}
        status = entry.get("status")
        if status in totals:
            totals[status] += 1
        reason = entry.get("reason")
        if reason:
            totals["reasons"][reason] = totals["reasons"].get(reason, 0) + 1
    return totals


def _candidate(component: dict, link: str) -> dict:
    """What it would take to close one missing link, and whether that is possible at all."""
    blocker = component["links"][link]["blocker"]
    kind, harnesses = component["kind"], component.get("harnesses") or []

    # Never propose reaching out to something the user asked this tool to leave alone. The
    # report was offering a `evaluate --measure` probe against an excluded MCP server, which
    # would have sent traffic to the endpoint the exclusion exists to protect.
    if component.get("excluded_by_request"):
        return {"link": link, "achievable": False, "method": "excluded by request",
                "estimated_usd": None,
                "note": "you asked this tool not to contact this component, and every "
                "method for this link requires contacting it"}

    # A skill that states its own checkable rule is in the same position as an instruction
    # file, and the arm that adds exactly one skill back is verified. Both arms drop the
    # user's own instruction files, so a rule those happen to state as well cannot account
    # for a difference between them — the route removes that confound rather than needing it
    # argued away. Decided before the no-record branch below, which would otherwise answer
    # "instrumentation that does not exist" for a component whose rule a machine can already
    # check: a compliance probe reads the agent's output, not the harness's records.
    if link == "useful" and kind == "skills":
        rules = stated_rules(component)[1]
        if rules:
            from .useful_probe import PROBE_COST_USD

            return {"link": link, "achievable": True,
                    "method": 'compliance probe through the skill arm '
                              '(`--setting-sources ""` plus `--plugin-dir`)',
                    "estimated_usd": PROBE_COST_USD,
                    "note": f"this skill states {len(rules)} rule(s) a machine can verify, "
                    "so an interleaved pair decides the question with nothing grading it"}

    # No amount of further collection fixes a fact the harness never records. Saying so is
    # the difference between an actionable gap and a wasted suggestion.
    if blocker == NO_EVENT_EMITTED:
        return {"link": link, "achievable": False, "method": "unsolved", "estimated_usd": None,
                "note": f"the {link} link cannot be established from traces, because a {kind} "
                "leaves no record when it enters context. This needs harness instrumentation "
                "that does not exist, not more analysis."}

    if blocker == NO_TRACES_COLLECTED:
        return {"link": link, "achievable": True,
                "method": "collect traces for this root",
                "estimated_usd": 0.0,
                "note": "no transcript was found under this path. If the project was moved, "
                "its history is filed under the old working directory — see "
                "relocation_candidates."}

    if link == "used":
        return {"link": link, "achievable": True,
                "method": "collect traces for the declaring harness",
                "estimated_usd": 0.0, "note": "deterministic; no tokens"}

    if link == "cost":
        if kind == "agents_md":
            # An empty file cannot be differenced: an earlier attempt did exactly that,
            # read -3 tokens, and nearly published "not loaded" as a finding. Offering the
            # measurement would sell a result the probe will refuse.
            if not (component.get("bytes") or 0):
                return {"link": link, "achievable": False, "method": "unsolved",
                        "estimated_usd": None,
                        "note": "this file is empty, so differencing it cannot show a cost "
                        "either way"}
            # Differencing the Codex rollout's input-token total works, which is how this
            # file's `loaded` link became provable at all.
            return {"link": link, "achievable": True,
                    "method": "codex rollout differencing (`measure-instructions`)",
                    "estimated_usd": 0.15,
                    "note": "injects a copy of the file into a controlled CODEX_HOME and "
                    "differences the prefix across interleaved pairs; the dollar figure is "
                    "a rough estimate, the token difference is measured directly"}
        if kind == "claude_md":
            # Recorded as unsolved for weeks on a half-true premise. Removal from a copied
            # config does fail — it hides the auth token. Additive injection does not: it
            # was abandoned after an *unpaired* run drifted, before interleaved pairs
            # existed. With them it returns the same figure in both arm orders, spread 0.
            return {"link": link, "achievable": True,
                    "method": "claude prefix differencing (`measure-instructions`)",
                    "estimated_usd": 0.15,
                    "note": "appends a verbatim copy with --append-system-prompt-file "
                    "against a --bare baseline across interleaved pairs; the content is "
                    "byte-identical but arrives through a different channel, so this is "
                    "the content's prefix cost rather than the harness's own plumbing"}
        if kind == "skills":
            # Recorded as unsolved on the premise that nothing isolates one skill. Nothing
            # does — and every arm in this project is built by addition anyway:
            # `--setting-sources ""` leaves the built-ins and `--plugin-dir` adds one skill
            # back, which measures as a clean positive difference with spread 0.
            if "claude_code" not in harnesses:
                return {"link": link, "achievable": False,
                        "method": "paired probe, not implemented for this harness",
                        "estimated_usd": None,
                        "note": "`--plugin-dir` is a Claude Code flag, so a "
                        f"{'/'.join(harnesses)} skill's prefix cost cannot be measured by "
                        "this tool today"}
            from .codex_probe import SKILL_PROBE_COST_USD

            return {"link": link, "achievable": True,
                    "method": "skill prefix differencing (`measure-skills`)",
                    "estimated_usd": SKILL_PROBE_COST_USD,
                    "note": "adds a copy of the skill through --plugin-dir against a "
                    '--setting-sources "" baseline across interleaved pairs, with each arm '
                    "listing its own skills so the pair proves it differs by this one; the "
                    "figure is the marginal cost of adding this skill, through the plugin "
                    "channel, not its share of a measured total"}
        if kind != "mcp":
            return {"link": link, "achievable": False, "method": "unsolved",
                    "estimated_usd": None,
                    "note": f"a {kind} is only paid for when it enters context, and nothing "
                    "records that it did, so its cost cannot be isolated"}
        if "claude_code" not in harnesses:
            return {"link": link, "achievable": False,
                    "method": "paired probe, not implemented for this harness",
                    "estimated_usd": None,
                    "note": f"`evaluate` probes Claude Code only, so a {'/'.join(harnesses)} "
                    "server's schema cost cannot be measured by this tool today"}
        return {"link": link, "achievable": True, "method": "stage 6 paired probe",
                "estimated_usd": 0.30, "note": "already implemented; run `evaluate --measure`"}

    if link == "useful":
        if kind in INSTRUCTION_KINDS:
            # The narrower design that Stage 6 left open. It works only where the file
            # states a rule a machine can check, so that is tested rather than promised:
            # a probe launched against a file with no quotable rule would refuse and the
            # money would be spent discovering what is readable for free.
            from .useful_probe import PROBE_COST_USD, scorable_checks

            path = Path(component.get("path") or "")
            try:
                stated, _ = scorable_checks(path.read_text(errors="replace"))
            except OSError:
                stated = []
            if stated:
                return {"link": link, "achievable": True,
                        "method": "compliance probe (`measure-useful`)",
                        "estimated_usd": PROBE_COST_USD,
                        "note": f"this file states {len(stated)} rule(s) a machine can "
                        "verify, so interleaved arms with and without it decide the "
                        "question without a model grading anything"}
            return {"link": link, "achievable": False, "method": "unsolved",
                    "estimated_usd": None,
                    "note": "the compliance probe needs a rule a machine can check, and "
                    "this file states none that it recognizes; its influence may still be "
                    "real but this method cannot see it"}
        # Why it is blocked differs by kind, and saying "same reason" for all of them would
        # hide that two of them are now half-solved. Both of those arms were tested rather
        # than assumed: `--strict-mcp-config --mcp-config <one server>` loads exactly that
        # server, and `--setting-sources ""` plus `--plugin-dir` loads exactly one skill on
        # top of the built-ins (a reproducible positive difference with spread 0, confirmed
        # by a nonce skill appearing in one arm and not the other). What they still lack is a machine-checkable task,
        # which is the harder half: an instruction file states its own rules, while a
        # skill's or a server's job does not come with a pass/fail a machine can read.
        arms = {
            "mcp": "`--strict-mcp-config` isolates one server",
            # No figure here: this string is printed on whoever's machine runs the report,
            # so a measurement from another stack reads as theirs. What the reader needs is
            # that the arm is isolating, which "spread 0" states.
            "skills": '`--setting-sources ""` drops the user\'s skills and `--plugin-dir` '
                      "adds exactly one back, as a positive difference with spread 0",
        }
        if kind in arms:
            blocked = (f"an arm exists — {arms[kind]}, verified — but this method also needs "
                       "a task whose success a machine can check, the way a stated "
                       "commit-message rule can be checked, and this one has no such task yet")
        else:
            blocked = ("no arm has been found that loads one of these in isolation, and no "
                       "machine-checkable task for it either")
        return {"link": link, "achievable": False,
                "method": "behavioural A/B: re-run comparable work with and without the component",
                "estimated_usd": None,
                "note": "Stage 6 costed this in the thousands of dollars and refused "
                "it at 0.014x. The narrower design now works for instruction files, which can "
                f"be toggled and checked; for a {kind}, {blocked}."}

    return {"link": link, "achievable": False, "method": "unknown", "estimated_usd": None,
            "note": ""}


def _next_measurement(component: dict) -> dict | None:
    """The best next step for this component: cheapest among those that are actually possible.

    Chain order is deliberately *not* used to pick. The links are ordered as a narrative,
    but they are not all prerequisites of one another — Stage 6 measured cost without
    knowing usefulness. Recommending an unsolved four-figure experiment ahead of a solved cheap
    one because it comes earlier in the chain would be bad advice dressed up as rigour.
    """
    blocking = _coverage(component)["blocking"]
    if not blocking:
        return None
    candidates = [_candidate(component, link) for link in blocking]
    achievable = [c for c in candidates if c["achievable"]]
    if achievable:
        best = min(achievable, key=lambda c: c["estimated_usd"])
    else:
        # Nothing is possible today; name the earliest gap so the reader sees where the
        # chain actually breaks.
        order = {link: i for i, link in enumerate(EVIDENCE_LINKS)}
        best = min(candidates, key=lambda c: order[c["link"]])
    return dict(best, alternatives=[c for c in candidates if c["link"] != best["link"]])


def score_root(root: str, *, profile: str | None = None, evaluation_path: str | None = None,
               limit: int | None = None, preferences_path: str | None = None,
               whole_environment: bool = False, exclusions: dict | None = None,
               useful_path: str | None = None, use_cache: bool = True) -> dict:
    """Score a stack, or report precisely why it cannot be scored."""
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    profiles, source, origin = load_preferences(preferences_path, base)
    supplied = source == ELICITED
    # With real preferences there is no sensible default, so the first one wins only when
    # the caller did not choose; with invented ones the default is the neutral profile.
    chosen = profile or (next(iter(profiles)) if supplied else DEFAULT_PROFILE)
    if chosen not in profiles:
        raise ValueError(f"unknown preference profile: {chosen} "
                         f"(available: {', '.join(sorted(profiles))})")
    profile = chosen

    started = time.perf_counter()
    from .core import inspect as inspect_stack
    from .events import codex_rollups
    from .traces import collect_root

    scan = inspect_stack(str(base), whole_environment=whole_environment, exclusions=exclusions)
    traces = collect_root(str(base), limit, whole_environment=whole_environment,
                          use_cache=use_cache)
    # Same rule as `report`: an explicit artifact wins, otherwise earlier measurements are
    # read from the store so paid work is not lost to a forgotten path.
    evaluation = None
    if evaluation_path:
        import json
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    else:
        from .store import stored

        evaluation = stored()
    useful = None
    if useful_path:
        import json
        useful = json.loads(Path(useful_path).read_text(encoding="utf-8"))

    # One pass, for the reason `report._collect` has one: these two ask different questions
    # of the same rollouts, and calling them in turn read the whole Codex corpus twice.
    codex_listing, codex_usage = codex_rollups(
        codex_skill_paths(scan), None if whole_environment else base,
        limit=limit, use_cache=use_cache)
    components = build_chain(scan=scan, traces=traces, evaluation=evaluation,
                             useful=useful, codex_usage=codex_usage,
                             codex_listing=codex_listing)
    orphans = unmatched_measurements(scan, evaluation, useful)
    passthrough = profiles if supplied else None
    for component in components:
        component["coverage"] = _coverage(component)
        component["sensitivity"] = sensitivity(component, passthrough)
        component["useful_measurability"] = useful_measurability(component, components)
        component["next_measurement"] = _next_measurement(component)

    scorable = [c for c in components if c["coverage"]["scorable"]]
    unstable = [c["name"] for c in scorable if c["sensitivity"].get("stable") is False]

    # Link coverage across the whole stack, which is the real headline.
    by_link = {}
    for link in SCORED_LINKS:
        statuses = [c["links"][link]["status"] for c in components]
        by_link[link] = {
            "proven": statuses.count(PROVEN),
            "partial": statuses.count(PARTIAL),
            "unproven": statuses.count(UNPROVEN),
        }

    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "mode": "score",
        "root": str(base),
        "read_only": True,
        "profile_used": profile,
        "profiles_available": sorted(profiles),
        "preference_source": source,
        "preferences_file": origin,
        "preferences_hint": (None if supplied else
                             f"no preferences were supplied, so the weights below are this "
                             f"tool's own hypotheses. Put real ones in "
                             f"{PREFERENCES_FILENAME} at the project root, or pass "
                             f"--preferences."),
        "components": components,
        # Splits the components without a `useful` verdict into the ones still waiting for a
        # measurement and the ones no method here can measure, with the reason for each.
        # Collapsing the two is what makes an unproven link look like unfinished work.
        "useful_measurability": _measurability_totals(components),
        # A supplied measurement that joined to nothing. Downstream it is indistinguishable
        # from one nobody ran, so it is reported beside the components rather than dropped.
        "unmatched_measurements": orphans,
        # Two supplied measurements of one file that disagree. The index keeps the later,
        # which is what it always did — what is new is saying so, because the figure a
        # reader saw used to depend on the order of a list.
        "measurement_conflicts": measurement_conflicts(evaluation),
        # History is filed by working directory, so a moved project reads as having none.
        # Passing the collector's finding through keeps a zero from looking like a fact.
        "relocation_candidates": (traces["observed"].get("relocation_candidates") or []),
        "observed": {
            "components_total": len(components),
            "components_scorable": len(scorable),
            "components_withheld": len(components) - len(scorable),
            "link_coverage": by_link,
            "rank_unstable": unstable,
        },
        "stack_score": _stack_score(scorable, components, profile),
        "profiler_cost": {
            "stage": "score",
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "cost_usd": 0.0,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        },
        "notes": [
            "A total is withheld for any component missing used, useful or cost. Ranking on "
            "whichever links happen to be measured would rank the best-evidenced component "
            "worst, which is the failure mode this stage exists to prevent.",
            # This note contradicted `preference_source` when weights were supplied: the
            # field said the user's while the prose said nobody's.
            (f"The weights come from {origin}, so they are the user's own and not this "
             f"tool's hypothesis." if source == ELICITED else
             "The preference profiles are named hypotheses authored by this tool. Nobody "
             "supplied preferences, so no profile here represents the user's."),
            "Links are never interpolated: installed is not evidence of loaded, and used is "
            "not evidence of useful.",
            "Every component carries the cheapest measurement that would unblock it, so a "
            "withheld score is actionable rather than a dead end.",
        ],
    }


def _stack_score(scorable: list[dict], components: list[dict], profile: str,
                 profiles: dict | None = None) -> dict:
    """A single number for the whole stack, which is almost always withheld."""
    if not scorable:
        return {
            "value": None,
            "withheld": True,
            "reason": f"0 of {len(components)} components have the evidence a score requires "
            f"({', '.join(REQUIRED_FOR_SCORE)}); an average over the rest would be a number "
            "about this tool's coverage, not about the stack",
        }
    covered = len(scorable) / len(components)
    if covered < 0.5:
        return {
            "value": None,
            "withheld": True,
            "reason": f"only {len(scorable)} of {len(components)} components are scorable "
            f"({covered:.0%}); a stack total would generalize from a minority",
        }
    units = comparable_cost_units(scorable)
    if len(units) > 1:
        named = ", ".join(sorted(str(u) for u in units))
        return {
            "value": None,
            "withheld": True,
            "reason": f"the scorable components price cost in different units ({named}); "
            "their cost signals sit on curves calibrated separately, so averaging them "
            "would produce a number with no single meaning",
        }
    values = [score_component(c, profile, profiles)["score"] for c in scorable]
    return {
        "value": round(sum(values) / len(values), 4),
        "withheld": False,
        "basis": f"mean over {len(scorable)} scorable components",
        "strength": "estimated",
        "coverage": round(covered, 3),
    }

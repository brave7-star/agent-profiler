from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_root, analyze_session
from .core import EXCLUSIONS_FILENAME, compare, inspect, load_exclusions, trace
from .codex_probe import measure_instructions, measure_skills
from .i18n import LANGUAGES, set_lang
from .verify import verify, verify_text
from .evaluate import evaluate_root
from .judge import BUDGET_TIERS, judge_root
from .report import report_root
from .score import PREFERENCE_PROFILES, DEFAULT_PROFILE, score_root
from .events import normalize_root, normalize_session, write_events
from .traces import collect_root, collect_session, discover_sessions
from .useful_probe import measure_useful


def dump(data, output):
    s = json.dumps(data, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(s + "\n", encoding="utf-8")
    else:
        print(s)


def _whole_env(parser):
    """Shared flags for asking about the environment rather than one project."""
    parser.add_argument("--all", dest="whole_environment", action="store_true",
                        help="profile the whole environment: every project's transcripts and "
                             "every per-project MCP declaration, not just this root's")
    return parser


def _exclusion_flags(parser):
    parser.add_argument("--exclude", action="append", metavar="NAME",
                        help="an MCP server this tool must not contact; repeatable. The server "
                             "is still inventoried and marked, never hidden")
    parser.add_argument("--exclude-file",
                        help=f"JSON file of exclusions; {EXCLUSIONS_FILENAME} at the root is "
                             f"read automatically when present")
    return parser


def _exclusions_for(a, root=None):
    """CLI exclusions, merged over any file, with the source named in the output."""
    base = Path(root).expanduser().resolve() if root else None
    loaded = load_exclusions(getattr(a, "exclude_file", None), base)
    named = [n for n in (getattr(a, "exclude", None) or []) if n.strip()]
    if named:
        merged = {k: set(v) for k, v in loaded["names"].items()}
        merged.setdefault("mcp", set()).update(named)
        sources = [s for s in (loaded["source"], "--exclude") if s]
        loaded = {"names": merged, "source": " + ".join(sources),
                  "reason": loaded["reason"] or "excluded on the command line"}
    return loaded


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agent-profiler", description="Read-only profiler for personal agent stacks")
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = _exclusion_flags(_whole_env(sp.add_parser("inspect")))
    x.add_argument("--root", required=True)
    x.add_argument("--output")
    x = _whole_env(sp.add_parser("sessions", help="list recorded Claude Code sessions"))
    x.add_argument("--root", required=True)
    x.add_argument("--output")
    x = _whole_env(sp.add_parser("collect", help="collect observations from recorded Claude Code traces"))
    group = x.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="collect every session recorded for this project root")
    group.add_argument("--session", help="collect one transcript file")
    x.add_argument("--limit", type=int, help="with --root, collect only the N most recent sessions")
    x.add_argument("--output")
    x = _whole_env(sp.add_parser("normalize", help="normalize traces into the harness-independent event model"))
    group = x.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="normalize sessions recorded for this project root")
    group.add_argument("--session", help="normalize one transcript file")
    x.add_argument("--harness", choices=["claude_code", "codex"], help="skip harness detection")
    x.add_argument("--limit", type=int, help="with --root, normalize at most N sessions per harness")
    x.add_argument("--events-out", help="write the event stream as JSONL to this path")
    x.add_argument("--output")
    x = _whole_env(sp.add_parser("analyze", help="deterministic findings from normalized events (0 LLM tokens)"))
    group = x.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="analyze every traced session for this project root")
    group.add_argument("--session", help="analyze one transcript file")
    x.add_argument("--harness", choices=["claude_code", "codex"], help="skip harness detection")
    x.add_argument("--limit", type=int, help="with --root, analyze at most N sessions per harness")
    x.add_argument("--full", action="store_true", help="include per-session detail in the output")
    x.add_argument("--output")
    x = _whole_env(sp.add_parser(
        "judge", help="semantic judge for what code cannot decide (spends LLM tokens)"))
    x.add_argument("--root", required=True)
    x.add_argument("--tier", choices=sorted(BUDGET_TIERS), default="standard")
    x.add_argument("--harness", choices=["claude_code", "codex"])
    x.add_argument("--limit", type=int)
    x.add_argument("--dry-run", action="store_true", help="build the funnel and spend nothing")
    x.add_argument("--force", action="store_true",
                   help="run even if the EIG gate refuses the trade")
    x.add_argument("--timeout", type=int, default=900, help="seconds to wait for the batched judge call")
    x.add_argument("--output")
    x = _exclusion_flags(_whole_env(sp.add_parser(
        "evaluate", help="what carrying a configuration costs (--measure spends tokens)")))
    x.add_argument("--root", required=True)
    x.add_argument("--measure", action="store_true",
                   help="actually run the paired probes; without it nothing is spent")
    x.add_argument("--harness")
    x.add_argument("--timeout", type=int, default=420, help="seconds to wait for each probe")
    x.add_argument("--output")
    x = _exclusion_flags(_whole_env(sp.add_parser(
        "report", help="the readable answer: what is installed, used, and what it cost")))
    x.add_argument("--root", required=True)
    x.add_argument("--evaluation", help="a stage-6 evaluate artifact, to supply measured cost")
    x.add_argument("--useful", help="a measure-useful artifact, to supply the useful link")
    x.add_argument("--limit", type=int)
    x.add_argument("--format", choices=("text", "json"), default="text")
    x.add_argument("--output")
    x = _exclusion_flags(_whole_env(sp.add_parser(
        "score", help="score the stack, or report why it cannot be scored")))
    x.add_argument("--root", required=True)
    x.add_argument("--profile",
                   help="which preference profile to score with; with no --preferences file "
                        "these are this tool's own hypotheses, not anyone's actual settings")
    x.add_argument("--preferences",
                   help="JSON file of real preference weights; without it the output says "
                        "the weights were assumed")
    x.add_argument("--evaluation", help="a stage-6 evaluate artifact, to supply the cost link")
    x.add_argument("--useful", help="a measure-useful artifact, to supply the useful link")
    x.add_argument("--limit", type=int)
    x.add_argument("--output")
    x = _whole_env(sp.add_parser(
        "measure-instructions",
        help="measure what each instruction file adds to every request (spends tokens)"))
    x.add_argument("--root", required=True)
    x.add_argument("--pairs", type=int, default=2,
                   help="interleaved A/B pairs per file; arms alternate so drift cannot "
                        "read as an effect")
    x.add_argument("--timeout", type=int, default=420)
    x.add_argument("--output")
    x = _whole_env(sp.add_parser(
        "measure-skills",
        help="measure what each skill adds to every request (spends tokens)"))
    x.add_argument("--root", required=True)
    x.add_argument("--only", help="one skill by directory name; at a fifth of a dollar a skill the whole "
                                  "set is a deliberate spend")
    x.add_argument("--pairs", type=int, default=2,
                   help="interleaved A/B pairs per skill; each arm lists its own skills so "
                        "a pair is only counted when it really differs by that one")
    x.add_argument("--timeout", type=int, default=420)
    x.add_argument("--output")
    x = _exclusion_flags(_whole_env(sp.add_parser(
        "measure-useful",
        help="whether an instruction file changes what the agent produces (spends tokens)")))
    x.add_argument("--root", required=True)
    x.add_argument("--repeats", type=int, default=3,
                   help="runs per arm per task; arms alternate within one window")
    x.add_argument("--timeout", type=int, default=240)
    x.add_argument("--output")
    x = sp.add_parser(
        "verify",
        help="check this machine's record formats against what the collector assumes "
             "(free, read-only, sends nothing)")
    x.add_argument("--limit", type=int,
                   help="transcripts per harness to read; the default is enough to decide")
    x.add_argument("--format", choices=("text", "json"), default="text")
    x.add_argument("--output")
    x = sp.add_parser("trace", help="schema-agnostic fallback parser for an arbitrary trace file")
    x.add_argument("--input", required=True)
    x.add_argument("--output")
    x = sp.add_parser("compare")
    x.add_argument("baseline")
    x.add_argument("candidate")
    x.add_argument("--output")

    # On every subcommand rather than before the subcommand, so it reads
    # `agent-profiler verify --lang en` — where a reader expects a flag to go.
    for sub in sp.choices.values():
        sub.add_argument("--lang", choices=list(LANGUAGES),
                         help="report language; default follows LANG (falls back to en)")
        # On every subcommand for the same reason, and because a reader who distrusts a
        # reused figure should not have to find out which stage does the reusing.
        sub.add_argument("--no-cache", dest="use_cache", action="store_false",
                         help="re-read every transcript instead of reusing a summary kept "
                              "for one whose bytes have not changed")

    a = ap.parse_args(argv)
    # Pinned before any renderer runs. An unset flag re-reads the environment, which is what
    # makes the the owner's machine Chinese and a fresh clone English without either asking.
    set_lang(getattr(a, "lang", None))

    try:
        if a.cmd == "inspect":
            data = inspect(a.root, whole_environment=a.whole_environment,
                           exclusions=_exclusions_for(a, a.root))
        elif a.cmd == "sessions":
            data = discover_sessions(a.root, whole_environment=a.whole_environment)
        elif a.cmd == "collect":
            data = (collect_root(a.root, a.limit, whole_environment=a.whole_environment,
                                 use_cache=a.use_cache)
                    if a.root is not None else collect_session(a.session))
        elif a.cmd == "normalize":
            if a.root is not None:
                data = normalize_root(a.root, a.limit, a.harness,
                                      whole_environment=a.whole_environment,
                                      use_cache=a.use_cache)
            else:
                data = normalize_session(a.session, a.harness)
                if a.events_out:
                    data["events_written"] = write_events(data, a.events_out)
            # The event stream can be large; the summary is what goes to stdout.
            data.pop("events", None)
        elif a.cmd == "analyze":
            if a.root is not None:
                data = analyze_root(a.root, a.limit, a.harness,
                                    whole_environment=a.whole_environment)
                if not a.full:
                    # Per-session detail is large; findings already carry their source.
                    data.pop("sessions", None)
            else:
                data = analyze_session(a.session, a.harness)
        elif a.cmd == "judge":
            data = judge_root(a.root, tier=a.tier, limit=a.limit, harness=a.harness,
                              dry_run=a.dry_run, force=a.force, timeout=a.timeout,
                              whole_environment=a.whole_environment)
        elif a.cmd == "report":
            data = report_root(a.root, evaluation_path=a.evaluation, limit=a.limit,
                               whole_environment=a.whole_environment,
                               exclusions=_exclusions_for(a, a.root),
                               useful_path=a.useful, use_cache=a.use_cache)
            if a.format == "text":
                # The whole point of this command is prose, so text goes out as text.
                text = data["text"]
                if a.output:
                    Path(a.output).write_text(text + "\n", encoding="utf-8")
                else:
                    print(text)
                return 0
        elif a.cmd == "score":
            data = score_root(a.root, profile=a.profile, evaluation_path=a.evaluation,
                              limit=a.limit, preferences_path=a.preferences,
                              whole_environment=a.whole_environment,
                              exclusions=_exclusions_for(a, a.root),
                              useful_path=a.useful, use_cache=a.use_cache)
        elif a.cmd == "evaluate":
            data = evaluate_root(a.root, measure=a.measure, harness=a.harness,
                                 timeout=a.timeout, exclusions=_exclusions_for(a, a.root),
                                 whole_environment=a.whole_environment)
        elif a.cmd == "measure-useful":
            data = measure_useful(a.root, repeats=a.repeats, timeout=a.timeout,
                                  whole_environment=a.whole_environment,
                                  exclusions=_exclusions_for(a, a.root))
        elif a.cmd == "measure-skills":
            data = measure_skills(a.root, pairs=a.pairs, timeout=a.timeout,
                                  whole_environment=a.whole_environment, only=a.only)
        elif a.cmd == "measure-instructions":
            data = measure_instructions(a.root, pairs=a.pairs, timeout=a.timeout,
                                        whole_environment=a.whole_environment)
        elif a.cmd == "verify":
            data = verify(limit=a.limit)
            if a.format == "text":
                text = verify_text(data)
                if a.output:
                    Path(a.output).write_text(text, encoding="utf-8")
                else:
                    print(text)
                return 0
        elif a.cmd == "trace":
            data = trace(a.input)
        else:
            data = compare(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.candidate).read_text()))
        dump(data, a.output)
        return 0
    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        PermissionError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"agent-profiler: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

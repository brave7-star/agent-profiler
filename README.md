**English** | [简体中文](README.zh-CN.md)

# Agent Profiler v0.1.0-dev

A read-only profiler for personal coding-agent stacks. Targets Claude Code and Codex on macOS.

It answers one question — **"my whole personal agent environment, how is it?"** — along a
chain where each link has to be earned separately:

```
Installed → Loaded → Used → Useful → Cost → Recommendation
```

The product is not a score. It is the evidence pipeline, and the rule that **a conclusion may
never be stronger than its evidence**. On a real environment most links come back unproven,
and the report says so on its first screen. That is the correct output, not a failure.

The seven rules it is built on are in `docs/principles.md`, in both languages.

## What it refuses to say

Worth knowing before reading any output, because it is the difference between this and a
dashboard:

- **No removal advice without a real A/B.** "Removing X would save Y%" is forbidden. Tests
  grep the whole output for that phrasing.
- **Cost is not waste.** A component can be expensive on every request and heavily used.
  Both facts are reported; neither implies the other.
- **Verdicts can only say "keep".** Separation proves influence; *no* separation is
  unattributable between three indistinguishable causes — the component does nothing, the
  task did not need it, or the baseline was already good enough. "Keep" is an abstention and
  needs no evidence. "Remove" is a claim, and the evidence for it usually does not exist.
- **A failed measurement is shown as failed**, never omitted, so it cannot be mistaken for
  one that was never attempted.
- **Links are never interpolated.** Installed is not evidence of loaded; used is not
  evidence of useful.

## Read-only, and that is a guarantee

The subject under test is the user's real configuration, so the tool never writes to it.
Nothing under `~/.claude`, `~/.codex`, `~/.agents`, no `CLAUDE.md`, `AGENTS.md`,
`settings.json`, `config.toml` or `.mcp.json` is modified, moved or created. The profiler
adapts to whatever is there.

`--exclude NAME` keeps a named MCP server from being contacted at all. It is still
inventoried and still reported — marked as excluded by request, so a blank cost cell is
attributed to the user's decision rather than to a gap in the tool.

## Scope: a project, or the whole environment

```bash
./bin/agent-profiler report --root ~/Projects/my-project          # this directory
./bin/agent-profiler report --root ~/Projects/my-project --all    # the whole environment
```

Transcripts are filed per working directory, so a project-scoped run misses work done
elsewhere — including MCP servers declared under the home directory's project key, and a
harness whose sessions are stored by date rather than by path. `--all` is the scope the
stated question actually needs.

The profiler also recognises **its own probe sessions**. Measuring a configuration means
running the harness, and the harness records those runs like any other session. Without that
recognition the tool counts its own exhaust as the user's work and bills its own spend to the
person being profiled. Those sessions are fingerprinted and excluded from every user-facing
total.

## Running it again

Every stage that walks the corpus keys a per-file cache on the file's **content hash**, never
on size and mtime. A same-size edit inside one second is invisible to a size-and-mtime check,
and a profiler that serves a stale answer for a file that changed is worse than a slow one.
The cache key also includes a fingerprint of the profiler's own source, so changing the
parser invalidates everything it produced.

`--no-cache` re-reads everything, for a reader who distrusts a reused figure.

What is deliberately **not** cached is anything attributed to an installed component. A
cached answer that already knew which skills exist would mean installing a skill did not
change the result — staleness that looks like a measurement.

## The biggest known risk: n = 1

Every conclusion here about *how a harness records things* was induced from one machine's
corpus. That is a larger unknown than any coverage gap, and more analysis cannot fix it — it
needs other machines.

So the assumptions are **executable**:

```bash
git clone <this repo>
./bin/agent-profiler verify      # read-only, sends nothing, a few seconds
```

Nine assumptions, three outcomes: `hold`, `violated` — with the record that disproves it
named — and `no_evidence`, which is **not a pass** but "this corpus has no instance, so this
machine says nothing about it". A fresh install has no rollouts and no MCP calls, and
reporting that as "all good" is exactly the overclaim this tool exists to avoid. An
assumption confirmed by only one or two instances also reads `no_evidence`: a format
invariant needs a handful of confirmations before a machine can speak to it.

Writing the verifier found four bugs in itself, every one in the direction that would have
handed a second machine a false all clear.

### What a second machine was worth

One has run it. Nothing was refuted, and the rows it could not test were the ones its corpus
had no instance of — the outcome working rather than a gap.

The valuable part was a sentence its operator added by hand, not the table: their rollouts
were stored in a different form from this machine's. That read as a cross-machine difference.
It was not — the storage form tracks file *age*, through a maintenance pass — but checking it
uncovered a much larger defect that had nothing to do with storage.

The harness writes token usage in two different records, and which one appears depends on its
**version**. The collector read only the record from one narrow version window, and correctly
refused to sum the other because it is session-cumulative — so it reported *zero tokens* for
the overwhelming majority of sessions, across a corpus several times larger than the one it
was counting. A guard against double counting had become a blind spot, and nothing in the
output said so. Both records are now read, kept in separate fields because they are not the
same quantity, with a coverage figure naming how many sessions have which kind of evidence.

Writing the ninth assumption for that record then reported `violated` *on the machine that
wrote it* — correctly, because the assumption was stated too broadly: the harness reuses the
same event type for rate-limit notices. Narrowing it also fixed a fresh bug in the day-old
parser, where such a notice arriving last would have overwritten a real session total with
zero.

None of that was in the second machine's command output. It came from one line of prose,
which is the argument for asking a person rather than only running the command.

## From a rate to a figure about a stack

A measured cost of "N tokens per request" is a rate, not a figure about anyone's stack. The
request volume that turns it into one is in the corpus and costs nothing to read: one model
response is one request. So the report multiplies, and says what it multiplied.

Three rules keep that honest. The request count **excludes the profiler's own probe
sessions**. A rate is multiplied **only by its own harness's traffic**, so a component
installed under more harnesses than it was measured under is marked `≥` with the excluded
harness named. And the rows are **never summed**: one skill installed under two scopes
appears twice carrying one prefix, and adding them would count it twice.

Money is withheld, and the blocker is named precisely: a price per token, which depends on
the model and on each request's cache state. An always-on prefix sits at the front of the
prompt, which is the most cacheable position, so charging it at full input price overstates
it badly. The tool does not pick a rate on the reader's behalf.

## Unproven, and whether that can change

An unproven link reads as unfinished work, which is the wrong reading for most components:
nothing this tool can run would change them. So each one is classified for free, from what
is already on disk, into one that still only needs a measurement and one that no method
here can measure — with the reason named. A component states no rule a machine can check;
a server is declared in configuration and ships no prose to quote; a file is empty. None of
those is answered by spending money, and saying so is the point.

The rule library used to decide this is the compliance probe's own. A separate set of
patterns written for the classifier was tried first and matched a sentence about what a
*tool* returns, and a flag inside a worked example, as though they were rules about the
agent's output — which would have put a paid probe behind two components that state nothing.

Where a component is measurable but installed in copies whose contents differ, every copy
is measured rather than one of them. The harness lists the name once and does not record
which copy it loaded, so a single measurement would be a verdict about an unidentified copy.
Measuring all of them makes the verdict copy-independent where they agree, and names the
disagreement where they do not.

### What the compliance probe covers

An instruction file and a skill are measured the same way, because a rule a machine can
verify does not care which kind of component stated it. Only the arm differs:

| component | the two arms differ by |
| --- | --- |
| instruction file | one verbatim copy appended, or present in the working directory |
| skill | one skill added through `--plugin-dir`, from a copy of the real directory |

The skill arm clears the user's own instruction files from *both* arms, so a rule those
state as well cannot account for a difference between them — the route removes that
confound rather than needing it argued away. It also never runs bare: bare mode skips
plugin sync, so the flag would load nothing while the arms appeared to differ.

A skill's own fixture skills are not components. One real skill ships them under
`evals/fixtures/skills/<name>/`, which satisfies every structural test for an install, so
nesting is detected by looking for an enclosing `SKILL.md`. Probing them would invent
installs that nothing loads.

`--plugin-dir` is a Claude Code flag, so a Codex-side skill has no arm here and is left
unmeasured rather than measured through a substitute route.

Two gates sit in front of the spend, both learned from a run that produced nothing useful:

- **The arm has to actually carry the component.** A skill loads on a description match, so
  a task unrelated to its description never reaches it and the arms differ by nothing. That
  reads as "this skill changes nothing" when it means "this task did not reach it". The
  harness is asked to list its skills first, and a skill absent from the arm meant to carry
  it is refused before anything runs. Word overlap between the description and the tasks was
  tried first and abandoned: a description in another language yields almost no tokens, and
  three incidental English fragments in a real one shared `loop` with a task about a page
  loop, so it passed by coincidence.
- **A batch stops when the harness stops answering.** The retry budget is sized for an
  independent failure rate, so every attempt against a persistent condition fails and the
  batch spends into a wall — 54 of 90 probes, in the run that prompted this. Six consecutive
  failures abandons the batch, and the artifact says it was abandoned so a short record set
  cannot be mistaken for a complete one.

A failure keeps the harness's own stderr, and `harness_health` reports per-harness probe
success once at the top. Without those, a dead arm is indistinguishable from a pile of
ordinary unprovens: in that run every Claude-side probe died while every Codex-side one
succeeded, and nothing in the output said so.

## Measurements that cost money, and the gate in front of them

Three `measure-*` commands and `evaluate --measure` run the harness and spend real money.
They are the deliberate-spend path and they say so before running.

`judge` and `evaluate` sit behind an expected-information-gain gate: before spending, each
declares what it expects to learn and what the answer is worth, and refuses the trade when
the spend exceeds the value. The semantic judge's own first run fails that gate — it spent
more than the avoidable waste it identified — and the gate now refuses it rather than the
repository merely documenting the lesson.

Probe reliability is **bounded, not explained**: the failure rate is measured, a per-arm
retry budget is derived from it, and failed spend is costed. The cause is unknown, and that
verdict is recorded as needing a different approach rather than as a solved problem.

### Measurements are kept for you

Measurement is the only thing here that costs money, and it is the only thing that cannot be
re-derived for free. So a `measure-*` or `evaluate --measure` run writes its result to
`~/.local/state/agent-profiler/measurements/` as well as to stdout, and `report` and `score`
read that directory automatically:

```bash
./bin/agent-profiler measure-instructions --root PATH --all   # costs money, once
./bin/agent-profiler report --root PATH --all                 # free, and has the cost column
```

Before this, a measurement went to stdout only. Feeding it back meant remembering
`--output`, remembering where the file went, and passing `--evaluation` on every later run —
and a report without that flag showed **no cost column at all**: the most expensive output in
the project, invisible by default. Producing a report with costs meant hand-merging several
artifacts into one file.

The store is append-only, one plain-JSON file per run, and `AGENT_PROFILER_STORE` overrides
the location. Two measurements of one file are both kept: readings that disagree are a
finding this tool reports, not a conflict to settle by recency. A report built from stored
figures says so on its first screen, so a remembered number is never mistaken for one
measured just now.

### Overriding what it uses

```bash
./bin/agent-profiler report --root PATH --all --evaluation EVAL.json --useful USEFUL.json
```

`--evaluation` wins over the store, which is what a reader comparing a fresh measurement
against the kept one needs. Pass both flags when both artifacts exist: supplying one alone
once dropped the other's measurements silently.

A measurement joins to a component by absolute path or server name, so an artifact produced
on another machine parses but joins to nothing — and the links it should have proven then
read as *never measured*, which is the one confusion this tool exists to prevent. `report`
and `score` carry `unmatched_measurements`, and the report says on its first screen how many
supplied measurements matched nothing and why. A path that exists but fell outside the scan
is told apart from a path that is not on this machine at all, because the two need different
actions.

Measurement artifacts are **not committed**. They are a profile of one private machine —
absolute home paths, installed components by name, and how much work was done in which
directories. Publishing a tool does not require publishing anyone's environment. Run the
commands on your own machine and you get your own.

## Commands

```
./bin/agent-profiler inspect   --root PATH [--all] [--exclude NAME] [--exclude-file FILE]
./bin/agent-profiler sessions  --root PATH [--all]
./bin/agent-profiler collect   (--root PATH | --session FILE) [--all] [--limit N]
./bin/agent-profiler normalize (--root PATH | --session FILE) [--all] [--harness H] [--events-out FILE]
./bin/agent-profiler analyze   (--root PATH | --session FILE) [--all] [--full]
./bin/agent-profiler report    --root PATH [--all] [--evaluation FILE] [--useful FILE] [--format text|json]
./bin/agent-profiler score     --root PATH [--all] [--profile P] [--preferences FILE] [--evaluation FILE] [--useful FILE]
./bin/agent-profiler verify    [--limit N] [--format text|json]
./bin/agent-profiler judge     --root PATH [--tier T] [--dry-run] [--force]
./bin/agent-profiler evaluate  --root PATH [--measure]
./bin/agent-profiler measure-instructions --root PATH [--pairs N]
./bin/agent-profiler measure-skills       --root PATH [--only NAME] [--pairs N]
./bin/agent-profiler measure-useful       --root PATH [--repeats N]
./bin/agent-profiler trace     --input FILE
./bin/agent-profiler compare   BASELINE.json CANDIDATE.json
```

Every subcommand takes `--lang en|zh`, `--no-cache` and `--output FILE`. With no `--lang`, the
language is read from `AGENT_PROFILER_LANG`, then `LC_ALL`/`LC_MESSAGES`/`LANG`, then falls
back to English.

`trace` is a schema-agnostic fallback for an arbitrary JSONL file; `collect` is the
format-aware path.

## Preferences, and why a score is usually withheld

A "preference-aware" score needs someone's preferences. With none supplied, the profiles are
**named hypotheses authored by this tool**, and every report says so in a machine-readable
field.

Their purpose is sensitivity analysis rather than ranking: run the same evidence under
several plausible preference sets and see whether the answer changes. A component whose rank
flips between profiles is one the evidence does not determine, and the honest output is the
flip.

Real weights go in `agent-profiler-preferences.json` at the project root, or `--preferences`.
See `docs/preferences.example.json`.

A stack total is withheld for any component missing `used`, `useful` or `cost`. Ranking on
whichever links happen to be measured would rank the best-evidenced component worst, which is
the specific failure this stage exists to avoid.

## Install

Nothing to install. Pure standard library, Python 3.9 or newer, and `bin/agent-profiler` is a
three-line shim.

```bash
PYTHONPATH=src python3 -B -m agent_profiler.cli verify     # equivalent to the shim
```

Run the tests with `python3 -B`. Without `-B`, a bytecode cache validated on a source file's
size and whole-second mtime can serve a stale module after a same-size edit inside one
second — and on some installs that cache lives outside the tree, so there is no `__pycache__`
to notice. That trap cost real debugging time here: `inspect.getsource` showed one thing
while the bytecode ran another.

```bash
for f in tests/test_*.py; do PYTHONPATH=src python3 -B "$f"; done
```

488 regression tests, each pinned to a defect or a format property observed in a real run
rather than to a hypothetical. No test spends a token.

## Format notes

`docs/dogfood/trace-schema.md` and `docs/dogfood/codex-schema.md` record what the two
harnesses actually write — record types, where usage lives, which fields are cumulative.
Those are properties of the harnesses, and they are the part another machine can refute, so
both are published in English and Chinese.

What they deliberately leave out is how large the corpus behind them was, and which sessions
were quoted. A volume describes the machine rather than the format, and a session id proves
nothing to a reader who cannot open that file. The claim a reader needs is whether a property
has a counterexample, and `verify` re-checks that against their own corpus. The same rule
covers the source: a comment may say what was measured and why it mattered, but not quote one
private stack's component sizes. `tests/test_privacy.py` enforces this by pattern class, after
two sweeps written as one-off scripts each found only what their author remembered to look
for.

## Licence

MIT. See `LICENSE`.

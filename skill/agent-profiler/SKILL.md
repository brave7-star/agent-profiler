---
name: agent-profiler
description: Profile a personal coding-agent stack with evidence-first, read-only analysis — what is installed, what actually loads into context, what gets called, what it costs per request, and what cannot be concluded. Use when the user asks to profile, audit, or understand their Claude Code or Codex setup, Skills, MCP servers, CLAUDE.md/AGENTS.md, context cost, token usage or possible agent-stack waste. Targets Claude Code and Codex on macOS. Never modify the user's agent configuration.
---

# Agent Profiler

A read-only profiler over the user's own agent environment. It answers one question —
**"my whole agent environment, how is it?"** — along a chain where each link is earned
separately:

```
Installed → Loaded → Used → Useful → Cost → Recommendation
```

The product is not a score. It is the evidence, plus the rule that **a conclusion may never
be stronger than the evidence behind it**. On a real environment most links are unproven, and
saying so is the correct output, not a failure.

## Two things to read before interpreting anything

- `references/reporting.md` — the reporting contract: what may be stated as Observed, what is
  Estimated, and what may never be claimed.
- `references/evidence.md` — what each link means, which commands prove it, and why "unproven"
  is not "absent".

## Workflow

1. **Confirm the root, and whether the question is about a project or the environment.**
   Transcripts are filed per working directory, so a project-scoped run misses work done
   elsewhere. "How is my setup?" almost always means `--all`.
2. **Spend nothing first.** `report --all` answers most of the question for $0 and names the
   cheapest measurement that would close each remaining gap. Read that list before proposing
   any paid command.
3. **Quote the report's own gaps.** It generates a section on what it cannot answer, from live
   link coverage. Do not summarize around it.
4. **Propose a paid measurement with its price, and wait for agreement.** The three `measure-*`
   commands have no gate: they spend as soon as they run.
5. **Feed artifacts back.** `report`/`score` take `--evaluation` and `--useful`. If both
   artifacts exist, pass **both** — passing one alone drops the other's measurements.
6. **Separate Observed from Review candidates** in anything written for the user.
7. **If the evidence is insufficient, say which command would settle it** rather than
   softening the claim into a guess.

## Commands

Free (0 LLM tokens):

```bash
bin/agent-profiler report   --root PATH --all [--evaluation F] [--useful F]   # start here
bin/agent-profiler score    --root PATH --all [--profile P] [--preferences F]
bin/agent-profiler inspect  --root PATH --all      # inventory
bin/agent-profiler analyze  --root PATH --all      # deterministic findings
bin/agent-profiler sessions --root PATH --all      # what transcripts exist
bin/agent-profiler verify                         # are the parser assumptions true here?
```

`verify` matters on a machine this tool has not seen before. Every measurement re-runs
anywhere; the **parsing** was induced from one machine's corpus, so run `verify` first on a
new machine and read its three outcomes carefully — `no_evidence` means that machine could
not test the assumption, which is **not** a pass. When it reports files it could not read,
that is a different answer from no records: install what it names and run it again.

Reports render in English or Chinese. Every command takes `--lang en|zh`; without it the
language comes from `AGENT_PROFILER_LANG`, then `LC_ALL`/`LC_MESSAGES`/`LANG`, then English.
Quote the report in whatever language it came out in rather than translating it — the
figures and their caveats are worded deliberately.

Lower-level, rarely needed directly — `report` runs these stages itself:

```bash
bin/agent-profiler collect   --root PATH --all     # per-session observations
bin/agent-profiler normalize --root PATH --all     # harness-independent events
bin/agent-profiler judge     --root PATH --all --dry-run   # the funnel only, $0
bin/agent-profiler trace     --input FILE.jsonl    # fallback parser for an arbitrary file
bin/agent-profiler compare   BASELINE.json CANDIDATE.json
```

Spends real money, only with the user's agreement:

```bash
bin/agent-profiler evaluate             --root PATH --measure   # ~$0.30/server
bin/agent-profiler measure-instructions --root PATH --all       # ~$0.15/file
bin/agent-profiler measure-skills       --root PATH --all       # ~$0.18/skill
bin/agent-profiler measure-useful       --root PATH --all       # ~$1.15/file
```

`judge` and `evaluate` sit behind an expected-information-gain gate and will refuse
themselves; the `measure-*` commands will not, so the judgment is the caller's.

## Guardrails

These are not style preferences. Each one exists because this project violated it and the
output was wrong in a way that looked fine.

- **Never modify** Claude Code, Codex, MCP, Skills, hooks, subagents or project files. The
  real configuration is the object under measurement. Every arm works by copying into a temp
  directory, never by editing in place.
- **Never say removing something would save anything** without a real A/B. "Cost is not waste":
  a component can be expensive and heavily used, and both facts are reported without either
  implying the other.
- **A verdict can only say "keep".** Separation proves influence; *no* separation is
  unattributable between three indistinguishable causes — the component does nothing, the task
  did not need it, or the baseline was already good enough. "Keep" is an abstention. "Remove"
  is a claim this tool cannot evidence.
- **Zero calls is not zero use.** A skill matched by description auto-loads with no record, so
  a zero proves only "never asked for by name".
- **Never divide a shared block.** One skill listing covers every skill; splitting it per skill
  invents an attribution.
- **Never add figures across harnesses or units.** A Codex request does not carry `CLAUDE.md`,
  and characters are not tokens.
- **Report a failed measurement as failed**, never omitted — otherwise it is indistinguishable
  from one never attempted.
- **No 0–100 stack score** unless most components are scorable; the tool withholds it and says
  why.
- **A blocker that has not been tested is not a finding.** Six documented "impossible" things
  in this project turned out false when someone finally tried them. If the user is told
  something cannot be measured, check whether that was ever verified.

## Known limits of this tool

State these rather than discovering them mid-analysis:

- **n = 1.** Every conclusion about *how a harness records things* came from one machine's
  corpus. The measurements re-run anywhere; the parsing is the exposed part. `verify` makes
  those assumptions executable, so on a new machine this is checkable rather than assumed.
- **`useful` works only for instruction files**, because they state their own rules and a
  machine can check compliance. The experimental arms for MCP servers and skills both exist
  and are verified; what is missing is a task with a machine-checkable outcome.
- **A duplicated skill name cannot be attributed** to the copy the harness loaded. Those
  components stay `partial`.
- **Probes fail ~40% of the time** with a signature that resembles success (exit 0,
  `input_tokens: 2`). Bounded by retries, cause unknown.
- **Codex skill cost is not measurable here** — `--plugin-dir` is a Claude Code flag.

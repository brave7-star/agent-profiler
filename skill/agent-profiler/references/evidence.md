# The evidence chain

Six links. Each is proved by its own observation, and **no link is ever inferred from
another** — installed is not evidence of loaded, used is not evidence of useful. Reading a
component's row means reading five independent answers, not one gradient.

## Statuses

| | meaning |
| --- | --- |
| `proven` | a direct observation establishes it |
| `partial` | an observation establishes something weaker than the link claims |
| `unproven` | not established — and the `blocker` says *why*, which decides what would fix it |

`unproven` is never "absent" or "zero". A skill with no recorded invocation is `partial`,
because description-match loading leaves no record at all.

## Blockers, and what each one implies

The blocker is the actionable part. Collecting more traces fixes one of these and does
nothing for another:

| blocker | what would change it |
| --- | --- |
| `no_traces_collected` | collect traces for this root — free. A moved project has its history under the old working directory (`relocation_candidates`) |
| `harness_not_traced` | collect the other harness's transcripts — free |
| `no_event_emitted` | nothing in the records will ever say it. Needs a measurement, or it is unanswerable |
| `not_measured` | a paid command exists and has not been run |
| `requires_earlier_link` | an earlier link must land first |
| `excluded_by_request` | the user asked this component to be left alone. **Not** the same as unmeasured |

## What proves each link

**Installed** — the file is on disk, with a kind, scope, harness and `origin`. `origin`
separates what the user declared from what a harness ships, so a vendored catalog never
inflates the stack.

**Loaded** — Claude Code writes `attachment` records naming the path and character count, so
this is free. Codex writes nothing, and it is proved instead by **prefix differencing**:
injecting a copy of the file moves the fixed prefix by a reproducible amount, which means the
content entered the context window. Codex's skill *listing* also records its own token count
per installed path.

**Used** — a tool call. MCP servers are attributed by tool namespace (`mcp__<server>__<tool>`)
and instruction files by path, both unambiguous. Skills are attributed **by name**, which is
weaker: a `Skill` call carries only `{skill, args}`, so a name installed under two roots gets
`partial`. Codex uses a skill by reading its `SKILL.md`, so reads count — with sessions that
also wrote to the file set aside, because authoring a skill looks identical to using it.

**Useful** — a behavioural separation under machine-verifiable checks. Works only for
instruction files, which state their own rules. Four controls make the result meaningful: a
negative control, an anti-recitation guard, a content-specificity test, and the rule that only
rules the file actually states may be scored.

**Cost** — tokens per request, from interleaved paired arms. Always in tokens, never silently
converted to money: a per-request token figure needs a request volume to become spend, and
printing a token count with a `$` in front of it was a real defect here.

**Recommendation** — requires `used`, `useful` and `cost` together, and can only ever say
*keep*.

## Reading a cost figure

Three things always travel with it:

1. **The harness it was measured under.** A figure from Claude Code must not be added to a
   Codex total; the same server reached through another harness is not the same prefix.
2. **The channel.** A `CLAUDE.md` measured through `--append-system-prompt-file` is the cost of
   the *content*, not of the harness's own loading. A skill measured through `--plugin-dir`
   carries a few tokens the installed copy does not.
3. **What it is relative to.** A skill figure is *marginal* — what adding it to an 11-skill
   baseline costs — not its share of a measured total.

A stack floor is a **floor**: unmeasured components can only push it up. Additivity across
prefix blocks is an assumption, stated as one.

## The aggregation trap

This project hit it six times, in six different units: tokens vs dollars, characters vs
tokens, sessions vs occurrences, a corpus projection vs a per-request value, a figure from one
harness added to two, and components vs what the harness lists. Every time, the output looked
fine.

So before combining two numbers, ask what unit each one is in, what it is per, and whether the
harness would ever pay both at once.

## When a figure is rejected

A rejected sample is not a data point. Specifically:

- An arm returning 2–4 tokens is an authentication/initialization failure, not a task failure.
  Counting it would manufacture evidence that the component was needed.
- Two arms of one pair differing by more than 50% mean one arm did not load the prefix.
- A probe far above the expected prefix did agentic work instead of measuring.
- A difference inside the noise band is `inconclusive`, with the samples kept visible.
- `reproducible` needs two surviving pairs. One pair has a spread of zero by arithmetic.

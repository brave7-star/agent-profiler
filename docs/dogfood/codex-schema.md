**English** | [简体中文](codex-schema.zh-CN.md)

# Codex transcript format: observed properties

Reverse-engineered from live rollout transcripts under `~/.codex/sessions/`, read-only.
Written because the normalized event model needs a *second* implementation to be worth
anything — one adapter proves only that a format can be renamed.

Induced from one machine's corpus, versions 0.144.2 through 0.162.0-alpha. As in the Claude
Code notes, the corpus size is deliberately left out: it describes the machine rather than
the format. Version numbers stay, because they are a property of the harness and a real
boundary on what these claims cover.

## Storage and access

`~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl.zst`, **zstd-compressed**.
There is no Python `zstandard` module here, so the adapter pipes through the `zstd` binary
and raises a clear error if it is missing rather than silently reporting nothing.

Sessions are stored **by date, not by project**. `~/.codex/session_index.jsonl` holds only
`id`, `thread_name` and `updated_at` — no cwd — so attributing a session to a project
requires reading its `session_meta` record. Discovery therefore reads just the opening
records of each file and stops as soon as `session_meta` is found, which keeps the cost
proportional to the session count rather than to the corpus size.

## Record shape

Every line is `{timestamp, ordinal, type, payload}` — a wrapper Claude Code does not have.
Observed `type` / `payload.type` combinations:

| Record | Meaning |
| --- | --- |
| `session_meta` | session id, cwd, `cli_version`, `model_provider` |
| `response_item/message` | a turn; `role` distinguishes user from developer |
| `response_item/custom_tool_call` | tool call with free-text `input` |
| `response_item/function_call` | tool call with JSON `arguments` |
| `response_item/custom_tool_call_output` | result |
| `response_item/function_call_output` | result |
| `response_item/reasoning` | reasoning content |
| `token_usage_record` | **per-response** usage, keyed by `response_id` |
| `event_msg/token_count` | **cumulative** session usage |
| `event_msg/item_completed` | duplicate view of `response_item` entries |
| `turn_context` | per-turn cwd, model, sandbox and approval policy |
| `compacted` | context compaction, with `replacement_history` |
| `event_msg/turn_aborted` | aborted turn |
| `world_state` | workspace state snapshot |

## Property 1 — the token fields mean something different

This is the finding that made a second adapter worth writing. Two identities hold on every
`token_usage_record` sampled, with no counterexample:

- `input_tokens + output_tokens == total_tokens`
- `cached_input_tokens <= input_tokens`

So Codex's `input_tokens` **already contains** `cached_input_tokens`. Anthropic's
`input_tokens` **excludes** `cache_read_input_tokens`, which is billed separately.

Mapping `cached_input_tokens` onto `cache_read_input_tokens` and adding it to the input
figure — the obvious thing to do — would have inflated every Codex input total by the cached
amount, which is the overwhelming majority of it: over 97% of input tokens on one root. The
normalized model states the invariant instead:

```
input_fresh + input_cached + input_cache_write == input_total
```

Claude Code: `input_total` is the **sum** of the three separately-billed quantities.
Codex: `input_total` is `input_tokens`, from which the cached part is **subtracted** to
get `input_fresh`.

`cache_write_input_tokens` is **0 in every record in the corpus**, so whether it sits
inside or outside `input_tokens` is *unverified*. The adapter assumes inside (consistent
with `cached_input_tokens`) and emits a `negative_fresh_input` integrity warning if a
future non-zero value ever contradicts that, rather than printing a negative number.

## Property 2 — cumulative figures sit next to per-response ones

Three different token views coexist, and two of them are cumulative:

- `token_usage_record.usage` — per response. **This is the only one summed.**
- `token_usage_record.turn_token_usage` — cumulative per turn; exceeded the per-response
  total in the large majority of records sampled, around 93% of them.
- `event_msg/token_count.info.total_token_usage` — cumulative per session.

Summing `turn_token_usage` would inflate totals several-fold. This is the same family of
trap as Claude Code's repeated `message.usage`, in a completely different shape.

## Property 3 — there is no trustworthy independent total

Claude Code's `cost-state` provides a harness-authored total that the collector is checked
against. Codex's nearest equivalent, the final cumulative `token_count`, does **not**
reconcile: compared against the sum of per-response usage over the largest sessions in the
corpus it gave deltas in **both** directions, far too large for any tolerance to absorb, and
growing with the number of compactions.

Rather than invent an explanation or tune a tolerance until it passed, this is recorded as
a limitation. **The Codex adapter is validated by the per-record identity
`input + output == total`, not by an independent session total**, and that is a genuinely
weaker guarantee than the Claude Code adapter has. Each record's stated `total_tokens` is
checked, and a mismatch raises a `harness_total_mismatch` warning.

## Property 4 — errors are implicit

Codex has no `is_error` flag. A failed tool call is visible only as a non-zero `exit_code`
inside the JSON embedded in the output text. The adapter parses that; a result whose output
cannot be parsed is reported as `ok`, so **Codex error counts are a lower bound**.

## Property 5 — no latency source

Codex records carry no request-start timestamp equivalent to Claude Code's `api-request`,
so `latency_seconds` is `null` for every Codex `model_response`. Reporting a derived
number here would exceed the evidence.

## Unverified

- Subagent / nested-turn attribution. `turn_attribution` carries `parent_turn_id` and
  `root_turn_id`, but no record in the corpus has a non-null `parent_turn_id`, so nested
  agent work is not proven to be captured.
- `event_msg/item_completed` is deliberately unused: a duplicate view of records already
  consumed.
- **`world_state` was wrongly dismissed as "a workspace snapshot".** It carries
  `state.host_skills`, which holds three things nothing else in either harness does:

      usage   -> {"tokens": <int>, "characters": <int>}   the listing's cost, measured
                                                          and recorded by the harness
      entries -> every skill's path, with a `shown` flag
      roots   -> the directories skills are loaded from

  `usage.tokens` is a **free, exact** cost for the skill listing, where the project
  otherwise pays for a paired probe to estimate one. Only the most recent sessions carry
  the record, so a figure read from it describes those sessions and not the whole history —
  which is why the report prints how much of the corpus carried it beside the figure.

  `roots` exposed `~/.agents/skills`, a skills directory the scanner was not looking
  at. `shown` is read but **not** treated as usage: it was `true` for every entry in
  every record sampled, so it does not vary and carries no information about which
  skill mattered. Reporting a constant as a measurement is the error this project
  exists to avoid.

**English** | [简体中文](trace-schema.zh-CN.md)

# Claude Code transcript format: observed properties

Reverse-engineered from live transcripts under `~/.claude/projects/`, not from
documentation. Every claim here was checked against real files before the collector relied
on it. Transcripts are read-only.

Induced from **one machine's** corpus, which is the largest known unknown in this project
(see the README, "the biggest known risk: n = 1"). That corpus's size is deliberately not
stated: a volume describes the machine, not the format. What a reader needs is whether a
claim has a counterexample, and `./bin/agent-profiler verify` re-checks each one against
their own corpus instead of asking them to trust a figure from someone else's.

## Record types observed

`assistant`, `user`, `api-request`, `api-request-blob`, `api-request-shape`, `attachment`,
`atis-latch`, `last-prompt`, `ai-title`, `queue-operation`, `file-history-snapshot`,
`file-history-delta`, `cost-state`, `system`, `mode`, `tools-used`, `permission-mode`.

Anything outside this set is reported in `integrity.unknown_record_types` rather than
dropped, so a harness upgrade surfaces as visible drift instead of silent undercounting.
`permission-mode` was found exactly this way: a run reported it as an unknown type instead
of discarding it.

## Property 1 — one response is written as several records

An API response with multiple content blocks is written as **one record per block**. All
such records share `message.id` and `requestId`, differ by `apiBlockIndex`, and **each
repeats the identical `message.usage`**.

Every session examined holds fewer distinct `message.id` values than usage-bearing records.
Duplicate usage carriers are the norm, not an edge case.

```
<message id>  blockIdx=0  blocks=[text]      in=2  cache_read=56285  out=186
<message id>  blockIdx=1  blocks=[tool_use]  in=2  cache_read=56285  out=186
```

**Consequence:** usage must be summed once per `message.id`. Summing per record inflated
one session's tokens by ~44%.

**`message.id` is the right key, not `requestId`.** Checked explicitly: in one session 19
distinct messages shared `requestId: null`, so de-duplicating by `requestId` would have
discarded every token they carried. No message id was ever observed carrying two different
usage values, so first-occurrence-wins is lossless (`first-wins == max-per-id`).

## Property 2 — `api-request-blob` replays the request

`api-request-blob` records repeat the outgoing request payload, including the assistant's
`tool_use` blocks. A naive sweep for `tool_use` across all records finds each call twice.

Verified by comparing the two sets within a session: every `tool_use` id in the `assistant`
records appeared again in the `api-request-blob` records — full overlap, and no id that only
the replay carried.

**Consequence:** tool calls are counted from `assistant` records only, de-duplicated by
`tool_use` id. Replays remain available as raw evidence of what was sent.

## Property 3 — latency links through `requestRef`, not `requestId`

`assistant.requestRef` equals `api-request.id`. `assistant.requestId` is a different
identifier and matches nothing in `api-request`.

```
api-request.id       = <uuid A>
assistant.requestRef = <uuid A>   <- links
assistant.requestId  = <uuid B>   <- does not
```

Matching on `requestId` linked nothing at all; `requestRef` linked every request.

## Property 4 — `cost-state` is the harness's own accounting

The final record of a modern session. Carries `totalCostUSD`, `totalAPIDuration`,
`totalToolDuration`, `totalLinesAdded/Removed`, `startTime`, and per-model
`modelUsage`. This is an **independent source** to check the collector against, which is
why `cross_check` exists.

Two real reasons the two accountings differ legitimately:

- **Resumed sessions.** `cost-state.startTime` can be *later* than the first record, so the
  transcript holds history the harness never billed. Found on a session whose transcript
  begins about 47 minutes before billing started, giving ratios of 2.1x–15x. Classified as
  `resumed_session_superset`, not an alarm.
- **Interrupted responses.** A response with `stop_reason: null` is recorded but excluded
  from the harness's cost accounting. Found on a session where one interrupted response
  carried **exactly** the cache-read tokens of the discrepancy, with `output_tokens: 8`.
  These are counted in `observed` (the transcript records them) and excluded from
  `cross_check` (the harness does not bill them), so the comparison stays like-for-like.

Compaction and pruning work the other way: the harness total survives, the records do
not, so collecting *less* is normal. Only collecting **more** indicates a de-duplication
defect, which is what `double_counting_suspected` guards. The alarm sits at +2%, above the
~1% accounting variance seen on real sessions and well below anything a repeated record
would produce.

Across every session in the corpus: **0 `double_counting_suspected`, 0 unexplained positive
variance.** Every divergence has an attributed cause.

## Coverage boundary: harness version

`cost-state` and `api-request` do not exist in older transcripts. Every session lacking them
is on a 2.1.2xx build; 2.1.3xx sessions carry both — apart from sessions still open at scan
time, which have not yet written their final record.

So coverage is a property of the session, not of the tool:

| Capability | Available in |
| --- | --- |
| Tokens, tools, turns, errors | every session |
| Independent cross-check | sessions that wrote `cost-state` |
| Per-request latency | sessions that wrote `api-request` |

`report` states the proportion for the corpus in front of it. A proportion quoted here would
describe one machine's upgrade history and nothing else.

## Not observed

No `isSidechain: true` record exists anywhere in the corpus, so subagent traces are
**unverified**. The collector counts `sidechain_records` but no real data exercises it;
`Task`-spawned work may be written elsewhere or not at all. This is a known gap, not a
supported feature.

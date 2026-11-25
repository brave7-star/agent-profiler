# Reporting contract

What may be written down, and in which of three registers. Mixing them is how an honest
measurement becomes a false claim.

## Observed

Directly measured or parsed facts: file counts and sizes, token fields present in traces,
tool-call counts with their success and error splits, repeated operations, latency, and a
measured per-request token difference from interleaved paired arms.

An Observed figure carries its unit and the harness it was measured under. Observed and
Estimated figures are **never merged into one total**.

## Estimated

Anything that applied a stated assumption: a dollar projection from a token figure, a corpus
total from a per-request measurement, a model's judgment. Each one lists its assumptions
inline, including the price list used, and any model verdict stays `estimated` — `unknown` is
an acceptable answer and is not converted into a finding.

## Review candidates

Patterns that may deserve a look and are **not** proven waste: a large instruction file, a
skill with no explicit invocation, overlapping MCP capabilities, a repeated read that may have
a semantic justification. These carry `judgment: review_only`. A byte count is a byte count.

## Never claimed

- **Savings.** "Removing X would save Y%" requires a real A/B. Several tests grep the entire
  output for this phrasing and have caught this project five times; each time the wording was
  changed rather than the test.
- **Waste**, for anything whose usefulness was not measured. Cost is not waste.
- **Removal**, ever, from this evidence. A verdict can only say *keep*.
- **Non-use**, from a zero call count. It proves "never asked for by name".
- **A per-component share of a shared block.** One skill listing covers every skill.
- **A stack score**, when most components are missing a required link. Withhold it and say
  which link and how many.

## Two things that must appear

**What the report cannot answer.** Generated from live link coverage and the confirmed blind
spots, not hand-written, so it cannot drift out of date while the data changes underneath. Do
not summarize around it.

**A failed measurement, shown as failed.** With its samples and the likeliest cause. An
omitted failure is indistinguishable from one that was never attempted, and the cheapest
command that would confirm the cause belongs beside it.

## Next steps are measurements, not advice

The closing section lists *measurements that would close an evidence gap*, each with a price
and whether it is actually achievable. It is not advice about changing the stack — that needs
`useful` and `cost` together. Never offer a measurement for a gap it could not close: a step
proposed against `no_event_emitted` spends money to rediscover that the record does not exist.

## Self-cost

Every run reports its own cost, including LLM tokens, and separates spend that bought a
measurement from spend that bought nothing. A profiler that burns more than it saves is a
failed product, and the field whose job is honesty about this tool's own waste is the last
place to round in its own favour.

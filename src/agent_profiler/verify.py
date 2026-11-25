"""Does this machine's harness write records the way this tool assumes?

Every measurement in this project re-runs anywhere. The **parsing** does not. Each
conclusion about *how a harness records things* was induced from one machine's corpus, and
that `n = 1` is the largest unknown in the project — larger than the coverage gap, because
a coverage gap is a known unknown while this one would make the collector quietly wrong.

Guessing at variants nobody has observed would be invented work. So instead this module
makes the assumptions **executable**: it runs each one against whatever corpus exists on
the machine it is pointed at, and reports hold / violated / no-evidence. Spending nothing
and sending nothing.

Three outcomes, and the middle one is the point:

    hold          the assumption was testable here and held
    violated      the assumption was testable here and FAILED — the collector is wrong
                  on this machine, and the specific record is named
    no_evidence   this corpus contains no instance, so this machine says nothing about it

`no_evidence` is not a pass. A fresh install has no Codex rollouts and no MCP calls, and
reporting that as "all good" is exactly the overclaim this project exists to avoid.

What is deliberately not checked: anything requiring a model call. This runs free, which is
what makes it reasonable to ask someone to run it on their own machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .core import _user_tree
from .i18n import tr

HOLD = "hold"
VIOLATED = "violated"
NO_EVIDENCE = "no_evidence"

# Enough instances to say an assumption holds here rather than coincidentally. Low on
# purpose: these are format invariants, not statistics — one counterexample refutes, and a
# handful of confirmations is all a format can offer.
MIN_INSTANCES = 3


class Check:
    """One assumption, with the evidence it was decided on."""

    def __init__(self, key: str, claim: str, why: str):
        self.key = key
        self.claim = claim
        self.why = why
        self.instances = 0
        self.violations: list[str] = []

    def saw(self, ok: bool, detail: str = "") -> None:
        self.instances += 1
        if not ok and len(self.violations) < 5:
            self.violations.append(detail)

    def result(self) -> dict:
        if self.violations:
            status = VIOLATED
        elif self.instances >= MIN_INSTANCES:
            status = HOLD
        elif self.instances:
            # Testable but barely. Saying "holds" off one instance is the single-pair
            # mistake of calling a single observation a confirmed result.
            status = NO_EVIDENCE
        else:
            status = NO_EVIDENCE
        return {
            "key": self.key, "claim": self.claim, "why": self.why, "status": status,
            "instances": self.instances, "violations": self.violations,
            "note": ("" if status != NO_EVIDENCE else
                     tr(en=f"only {self.instances} instance(s) found; {MIN_INSTANCES} "
                           "needed before this machine can confirm it",
                        zh=f"只找到 {self.instances} 个实例；这台机器要确认它需要 "
                           f"{MIN_INSTANCES} 个")),
        }


# Transcripts are sampled **largest first**, not newest first. The first version of this
# sorted by mtime and found no evidence for five of seven assumptions *on the machine the
# whole project was built on* — because the newest transcripts were this profiler's own
# `claude -p` probes, which are one turn long and contain almost no record types. A verifier
# that reports "no evidence" where evidence exists would hand a second machine a false all
# clear, which is worse than having no verifier.
SAMPLE_FILES = 8
# Some records are rare: Codex's skill listing appears in a handful of sessions, so a
# fixed sample of the largest few misses it entirely and the row reads `no_evidence` on the
# machine that record was discovered on. So sampling continues until every check has enough
# instances, bounded so an enormous corpus cannot turn a free command into a long one.
SAMPLE_CEILING = 80


def _largest(paths, count: int) -> list[Path]:
    return sorted(paths, key=lambda p: -p.stat().st_size)[:count]


def _settled(checks) -> bool:
    """True once every check has either enough instances or a counterexample."""
    return all(c.instances >= MIN_INSTANCES or c.violations for c in checks)


def _claude_checks(home: Path, limit: int | None) -> list[dict]:
    """The four Claude Code assumptions that would break counting if they differed."""
    usage_once = Check(
        "usage_per_message_id",
        tr(en="one API response is written as several records repeating the same "
              "`message.usage`",
           zh="一次 API 回复会被写成多条记录，每条重复同一个 `message.usage`"),
        tr(en="usage is counted once per `message.id`; summing per record overcounted one "
              "real session by ~44%",
           zh="用量按 `message.id` 只算一次；按记录逐条相加，曾把一个真实会话多算约 44%"),
    )
    replay = Check(
        "replay_records_carry_tool_use",
        tr(en="`api-request-blob` records replay the request, including its `tool_use` "
              "blocks",
           zh="`api-request-blob` 记录会重放请求，连同里面的 `tool_use` 块"),
        tr(en="tool calls are counted from `assistant` records only; counting replays too "
              "would double every call",
           zh="工具调用只从 `assistant` 记录里统计；把重放也算上会让每次调用翻倍"),
    )
    latency = Check(
        "latency_links_through_requestRef",
        tr(en="`assistant.requestRef` matches an `api-request.id`",
           zh="`assistant.requestRef` 能对上某条 `api-request.id`"),
        tr(en="latency joins on `requestRef`; `requestId` matched nothing at all where "
              "`requestRef` matched every record",
           zh="延迟是按 `requestRef` 关联的；`requestId` 一条都对不上，而 `requestRef` 条条都对上"),
    )
    attachments = Check(
        "attachment_is_a_record_type",
        tr(en="context loading is written as a top-level `attachment` record",
           zh="上下文的载入会被写成一条顶层 `attachment` 记录"),
        tr(en="this is what makes Claude Code's `loaded` link free rather than measured",
           zh="正因为有这条，Claude Code 的「载入过」才是免费读出来的，不用花钱测"),
    )
    checks = (usage_once, replay, latency, attachments)

    projects = home / "projects"
    if not projects.is_dir():
        return [c.result() for c in checks]

    for path in _largest(projects.rglob("*.jsonl"), limit or SAMPLE_CEILING):
        if _settled(checks):
            break
        seen_usage: dict[str, list[int]] = {}
        request_ids: set[str] = set()
        refs: list[str] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            message = record.get("message") if isinstance(record.get("message"), dict) else {}

            if kind == "assistant":
                msg_id, usage = message.get("id"), message.get("usage") or {}
                if isinstance(msg_id, str) and usage:
                    seen_usage.setdefault(msg_id, []).append(
                        sum(int(usage.get(k) or 0) for k in
                            ("input_tokens", "cache_read_input_tokens",
                             "cache_creation_input_tokens", "output_tokens")))
                ref = record.get("requestRef")
                if isinstance(ref, str):
                    refs.append(ref)
            elif kind == "api-request":
                if isinstance(record.get("id"), str):
                    request_ids.add(record["id"])
            elif kind in ("api-request-blob", "api-request-shape"):
                content = message.get("content")
                if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                    replay.saw(True)
            elif kind == "attachment" and isinstance(record.get("attachment"), dict):
                attachments.saw(True)

        for msg_id, totals in seen_usage.items():
            if len(totals) < 2:
                continue
            # The assumption is that the repeats are *identical*. Differing repeats would
            # make counting-once wrong in the other direction, which is worth knowing.
            usage_once.saw(len(set(totals)) == 1,
                           f"{path.name}: message {msg_id} repeats usage with "
                           f"{len(set(totals))} different totals")
        for ref in refs:
            if request_ids:
                latency.saw(ref in request_ids,
                            f"{path.name}: requestRef {ref} matches no api-request id")

    return [c.result() for c in checks]


def _codex_checks(home: Path, limit: int | None) -> list[dict]:
    """The Codex assumptions, including the one that differs from Claude Code's."""
    token_identity = Check(
        "codex_input_includes_cached",
        tr(en="in `token_usage_record.usage`, `input_tokens` already includes the cached "
              "portion (`input + output == total`)",
           zh="`token_usage_record.usage` 里，`input_tokens` 已经含了缓存那部分"
              "（`input + output == total`）"),
        tr(en="Claude Code bills the two separately. Treating them alike inflates every "
              "Codex input total by its cached portion",
           zh="Claude Code 是把这两样分开计费的。当成一样处理，会让每个 Codex 输入总数虚高"
              "一个缓存量"),
    )
    skills_by_path = Check(
        "host_skills_keyed_by_path",
        tr(en="`state.host_skills.entries` is keyed by the skill's full file path",
           zh="`state.host_skills.entries` 是用 skill 的完整文件路径做键的"),
        tr(en="this is what lets the `loaded` link be attributed per installed copy rather "
              "than per name",
           zh="正因为有这条，「载入过」才能归到具体装的那一份上，而不是只能归到名字上"),
    )
    mcp_in_program = Check(
        "mcp_calls_inside_programs",
        tr(en="Codex writes MCP calls inside the programs it runs, as "
              "`tools.mcp__server__tool(`",
           zh="Codex 把 MCP 调用写在它运行的程序里面，长成 `tools.mcp__server__tool(` 这样"),
        tr(en="Codex emits no MCP tool call of its own, so without this every one of them "
              "is invisible",
           zh="Codex 自己不发任何 MCP 工具调用，所以没有这一条的话，每一次调用都是看不见的"),
    )
    turn_context_at_top = Check(
        "turn_context_named_at_top_level",
        tr(en="a `turn_context` record names itself at the top level, and its payload "
              "carries no `type` of its own",
           zh="`turn_context` 这种记录把类型名写在顶层，它的 payload 里没有自己的 `type`"),
        tr(en="the per-turn model and reasoning effort are read from these records, and "
              "they are read by matching this name. Matching it against the payload "
              "instead — which is what this project did — dropped every one of them, silently",
           zh="每轮用的模型和推理档位就是从这些记录里读的，靠的就是匹配这个名字。"
              "拿 payload 里的名字去匹配（本项目之前就是这么写的），"
              "会把它们全部静默丢掉"),
    )
    token_count_universal = Check(
        "token_count_present_every_version",
        tr(en="an `event_msg/token_count` record with an `info` block carries a "
              "session-cumulative `total_token_usage`; one with `info: null` is a "
              "rate-limit notice, not a token report",
           zh="带 `info` 的 `event_msg/token_count` 记录会给出会话累计的 "
              "`total_token_usage`；`info` 为 null 的那种是限流通知，不是 token 报告"),
        tr(en="this is the only token record most versions write at all. "
              "`token_usage_record` exists in one version window (0.162), so reading that "
              "alone reported zero tokens for the overwhelming majority of sessions and hid "
              "input tokens. A machine where this record is absent would break the only "
              "route to Codex spend",
           zh="对大多数版本来说这是唯一的 token 记录。`token_usage_record` 只存在于 0.162 "
              "一个版本窗口，只读它的话，绝大多数会话都会报零 token，藏掉数倍于所统计的"
              "输入 token 量。要是某台机器上没有这条记录，通向 Codex 花费的唯一路就断了"),
    )
    checks = (token_identity, skills_by_path, mcp_in_program, turn_context_at_top,
              token_count_universal)
    if not home.is_dir():
        return [c.result() for c in checks]

    from .events import _codex_lines

    # Both forms, because this machine's corpus is split between them. The reason first
    # recorded here was wrong: `token_usage_record` was said to appear **only** in the
    # compressed rollouts, which was a correlation, not a cause. Compression tracks file
    # *age* — a maintenance pass, plain 2026-11-08 onward and packed through 2026-10-23,
    # with no overlap — while that record tracks Codex *version*, and only 0.162 writes it.
    # A second machine settled it: its rollouts were all plain and all carried the record,
    # because its Codex is older. Reading both forms is still right; the stated reason is
    # now the real one.
    candidates = [p for p in home.rglob("rollout-*") if p.is_file()]
    cap = limit or SAMPLE_CEILING
    plain = _largest((p for p in candidates if p.suffix != ".zst"), cap)
    packed = _largest((p for p in candidates if p.suffix == ".zst"), cap)

    # A file this machine *has* but cannot read is a different answer from a record this
    # machine does not have, and collapsing both into `no_evidence` hides a one-command fix.
    # Decompression shells out to the `zstd` binary, which macOS does not ship — so this is
    # likeliest on exactly the fresh second machine the command exists for, and cannot
    # most likely on exactly the fresh machine this command exists for.
    unread = 0
    needs_zstd = False

    for path in plain + packed:
        if _settled(checks):
            break
        try:
            lines = list(_codex_lines(path))
        except RuntimeError:
            # Missing zstd must not look like a refuted assumption — but it must not look
            # like an empty corpus either.
            unread += 1
            needs_zstd = True
            continue
        except OSError:
            unread += 1
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}

            if record.get("type") == "token_usage_record":
                usage = payload.get("usage")
                if isinstance(usage, dict) and usage.get("total_tokens") is not None:
                    inp = int(usage.get("input_tokens") or 0)
                    out = int(usage.get("output_tokens") or 0)
                    total = int(usage["total_tokens"])
                    # Zero-input rows are session bookkeeping rather than a response, and
                    # one of them carries a non-zero cumulative total — checking those
                    # would report this assumption as violated on the machine it came from.
                    if inp:
                        token_identity.saw(inp + out == total,
                                           f"{path.name}: input {inp} + output {out} "
                                           f"!= total {total}")

            # `payload.state`, not `world_state` — several of this project's own comments
            # said `world_state` and the verifier inherited the error, which is how it
            # reported "no evidence" for a record that is right there.
            state = payload.get("state")
            entries = ((state or {}).get("host_skills") or {}).get("entries") \
                if isinstance(state, dict) else None
            for key in entries or {}:
                skills_by_path.saw("/" in str(key),
                                   f"{path.name}: host_skills key {key!r} is not a path")

            if "tools.mcp__" in line:
                mcp_in_program.saw(True)

            if record.get("type") == "turn_context":
                turn_context_at_top.saw(
                    payload.get("type") is None,
                    f"{path.name}: turn_context payload carries its own type "
                    f"{payload.get('type')!r}")

            if record.get("type") == "event_msg" and payload.get("type") == "token_count":
                # The record has to be there *and* carry the cumulative block; a version
                # that renamed or dropped `total_token_usage` would leave the only route to
                # Codex spend reading zero, which is the failure this check exists to catch.
                info = payload.get("info")
                if not isinstance(info, dict):
                    # A rate-limit notice. Counting it as an instance of the token
                    # assumption was the first version of this check, and it reported
                    # `violated` where it was first run — correctly, for an assumption
                    # that was simply stated too broadly.
                    continue
                total = info.get("total_token_usage")
                token_count_universal.saw(
                    isinstance(total, dict) and total.get("total_tokens") is not None,
                    f"{path.name}: token_count has an info block but no "
                    f"total_token_usage.total_tokens")

    results = [c.result() for c in checks]
    if unread:
        reason = (tr(en="the `zstd` binary was not found, so compressed rollouts could not "
                        "be read",
                     zh="系统上找不到 `zstd` 命令，压缩会话读不了")
                  if needs_zstd else
                  tr(en="they could not be opened", zh="打不开"))
        for result in results:
            result["unread_files"] = unread
            result["needs_zstd"] = needs_zstd
            if result["status"] == NO_EVIDENCE:
                detail = tr(
                    en=f"{unread} session file(s) on this machine went unread: {reason}",
                    zh=f"这台机器上有 {unread} 个会话文件没读成：{reason}")
                result["note"] = f"{result['note']}; {detail}" if result["note"] else detail
    return results


def verify(*, limit: int | None = None) -> dict:
    """Check this machine's record formats against what the collector assumes.

    Free, read-only, and sends nothing. The point is to be run on a *second* machine: a
    `violated` row there means the collector is wrong on that machine and names the record
    that proves it, and a row of `no_evidence` means that machine cannot speak to it rather
    than that all is well.
    """
    started = time.perf_counter()
    claude_home = _user_tree("AGENT_PROFILER_CLAUDE_HOME", ".claude")
    codex_home = _user_tree("AGENT_PROFILER_CODEX_HOME", ".codex")

    checks = []
    for harness, home, runner in (("claude_code", claude_home, _claude_checks),
                                  ("codex", codex_home, _codex_checks)):
        present = home.is_dir()
        for result in runner(home, limit):
            checks.append({**result, "harness": harness,
                           "harness_present": present})

    counts = {HOLD: 0, VIOLATED: 0, NO_EVIDENCE: 0}
    for check in checks:
        counts[check["status"]] += 1

    # Surfaced at the top because it changes what the reader should do: an assumption this
    # machine could not test for want of a decompressor is one `brew install zstd` away from
    # testable, which a row reading `no_evidence` on its own does not say.
    unread = max((c.get("unread_files") or 0) for c in checks) if checks else 0
    blocked = {"unread_files": unread,
               "install": "zstd" if any(c.get("needs_zstd") for c in checks) else None}

    return {
        "schema_version": "0.1",
        "mode": "verify",
        "read_only": True,
        "machine": {"claude_home": str(claude_home), "codex_home": str(codex_home),
                    "claude_present": claude_home.is_dir(),
                    "codex_present": codex_home.is_dir()},
        "checks": checks,
        "summary": counts,
        "unreadable": blocked,
        # A verdict, because the caller is running this to answer one question.
        "verdict": (
            tr(en="parser assumptions VIOLATED on this machine",
               zh="解析假设在这台机器上不成立（VIOLATED）") if counts[VIOLATED] else
            tr(en="no assumption was refuted, but this machine could only confirm "
                  f"{counts[HOLD]} of {len(checks)}",
               zh=f"没有假设被推翻，但这台机器只能确认 {len(checks)} 条里的 {counts[HOLD]} 条")
            if counts[NO_EVIDENCE] else
            tr(en="every assumption was testable here and held",
               zh="每一条假设在这里都测得到，而且都成立")),
        "profiler_cost": {
            "stage": "verify",
            "llm_input_tokens": 0, "llm_output_tokens": 0, "cost_usd": 0.0,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        },
        "notes": [
            tr(en="`no_evidence` is not a pass: it means this corpus has no instance, so "
                  "this machine says nothing about that assumption.",
               zh="`no_evidence` 不等于通过：它是说这台机器的语料里没有相应实例，所以它对那条"
                  "假设无话可说。"),
            tr(en="Every measurement in this project re-runs anywhere; the parsing is what "
                  "is induced from one machine. That is what this command is for.",
               zh="这个项目里的测量换台机器都能重跑；只有「怎么解析记录」是从一台机器上归纳"
                  "出来的。这个命令就是为它存在的。"),
            tr(en="Nothing is sent anywhere and no file is modified. Record *contents* are "
                  "never printed — a violation is reported as a file name and a shape, "
                  "because a transcript holds the user's own work.",
               zh="不往外发任何东西，也不改任何文件。记录的**内容**从不打印 —— 不成立只报文件名"
                  "和形状，因为会话记录里装的是使用者自己的工作。"),
        ],
    }


def verify_text(data: dict) -> str:
    """The same result as something a person can read in ten seconds."""
    marks = {HOLD: tr(en="✓ holds", zh="✓ 成立"),
             VIOLATED: tr(en="✗ violated", zh="✗ 不成立"),
             NO_EVIDENCE: tr(en="— untestable here", zh="— 这台机器判断不了")}
    lines = [tr(en="# Parser assumptions, checked", zh="# 解析假设核对"), "",
             f"{data['verdict']}", "",
             # Spelled out because a reader took the bare term for a dependency they had
             # failed to install. It names two programs they already have.
             tr(en="\"Harness\" means the command-line program that runs the agent — here "
                   "just Claude Code and Codex. There is nothing else to install.",
                zh="「宿主程序」指跑 agent 的那个命令行程序，这里就是 Claude Code 和 Codex "
                   "两个，没有别的东西要装。"), "",
             tr(en="| harness | assumption | result | instances |",
                zh="| 宿主程序 | 假设 | 结果 | 实例数 |"),
             "| --- | --- | --- | --- |"]
    for check in data["checks"]:
        lines.append(f"| {check['harness']} | {check['claim']} | "
                     f"{marks[check['status']]} | {check['instances']} |")
    violated = [c for c in data["checks"] if c["status"] == VIOLATED]
    if violated:
        lines += ["", tr(en="## Violated (the collector is wrong on this machine)",
                         zh="## 不成立的那些（这台机器上采集器是错的）"), ""]
        for check in violated:
            lines += [f"**{check['claim']}**",
                      tr(en=f"- Why this one matters: {check['why']}",
                         zh=f"- 为什么这条重要：{check['why']}")]
            lines += [tr(en=f"- Counterexample: {v}", zh=f"- 反例：{v}")
                      for v in check["violations"]]
            lines.append("")
    unknown = [c for c in data["checks"] if c["status"] == NO_EVIDENCE]
    if unknown:
        # The blanket version of this line said the corpus holds no instance, which is false
        # when files went unread — the instances may be sitting in them. Two causes, so two
        # sentences; asserting the wrong one is the overclaim this module exists to avoid.
        unread = (data.get("unreadable") or {}).get("unread_files")
        why = (tr(en="this corpus holds no instance, or holds some that could not be read "
                     "(see below)",
                  zh="是这台机器的语料里没有相应的实例，或者有、但没能读取（见下）") if unread
               else tr(en="this corpus holds no instance",
                       zh="是这台机器的语料里没有相应的实例"))
        lines += ["", tr(en="## Untestable on this machine",
                         zh="## 这台机器判断不了的那些"), "",
                  tr(en=f"*This is not a pass — {why}, so this machine says nothing about "
                        "these.*",
                     zh=f"*这不是通过 —— {why}，所以它对这几条无话可说。*"), ""]
        for check in unknown:
            fallback = tr(en="no instance in this corpus", zh="语料里没有实例")
            lines.append(f"- {check['claim']}（{check['note'] or fallback}）")
    # Not folded into the list above: "could not read the files" is a different answer from
    # "has no such records", and only one of the two has a fix the reader can act on.
    blocked = data.get("unreadable") or {}
    if blocked.get("install"):
        lines += ["", tr(en="## Some of that was unreadable, not absent",
                         zh="## 其中有一部分是读不了，不是没有"), "",
                  tr(en=f"{blocked['unread_files']} session file(s) on this machine could "
                        f"not be read: Codex's compressed sessions need the "
                        f"`{blocked['install']}` command to decompress, and it was not "
                        "found (macOS does not ship it).",
                     zh=f"这台机器上有 {blocked['unread_files']} 个会话文件没能读取："
                        f"Codex 的压缩会话要靠 `{blocked['install']}` 命令解压，而系统上找不到"
                        "它（macOS 不自带）。"),
                  "",
                  "```", f"brew install {blocked['install']}", "```", "",
                  tr(en="Install it and re-run, and whichever of the rows above do have "
                        "data become testable. **Until then, do not read them as \"this "
                        "machine is fine\".**",
                     zh="装上重跑，上面那几条里本来有数据的就能测出来了。**在那之前，别把它们"
                        "读成「这台机器没问题」。**")]
    lines += ["", "---", "",
              tr(en="Read-only, zero tokens, nothing sent anywhere.",
                 zh="只读，零 token，不往外发任何东西。")]
    return "\n".join(lines)

"""The readable answer to "how is my agent environment?"

Seven stages produced evidence scattered across a dozen JSON artifacts. This assembles it
into one report a person can read, and it is the first place in the project where the
evidence discipline is genuinely at risk: prose invites confident sentences, and a summary
is exactly where "unproven" quietly becomes "fine".

So three rules are enforced structurally rather than by good intentions:

1. **Every number carries its strength.** Observed figures and Estimated projections are
   labelled inline, never merged into one total.
2. **The report states what it cannot answer**, and that section is generated from real
   link coverage plus the confirmed blind spots — not hand-written, so it cannot drift out
   of date while the data changes underneath it.
3. **No component gets a verdict.** The closing section lists *measurements that would
   close an evidence gap*, which is not the same as advice about changing the stack. Advice
   needs the `useful` link, which now exists for instruction files and for nothing else —
   and even where it exists it is a floor on one file's influence, not a valuation of it.

The rendered text comes out in the reader's language — Chinese for a reader who reads it,
English for anyone who clones the repository — resolved by `i18n` from `--lang` or the
environment. Code, identifiers and comments stay English like the rest of the project.
Evidence strings and assumptions produced by earlier stages are structured data shared
across all of them, so they are quoted verbatim rather than translated — a quoted figure
should read the same here as in the artifact it came from.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .i18n import count, tr
from .score import (
    EVIDENCE_LINKS,
    INSTRUCTION_KINDS,
    NO_EVENT_EMITTED,
    PARTIAL,
    PROVEN,
    SCORED_LINKS,
    UNPROVEN,
    EMPTY_TEXT,
    NO_RULE,
    NO_TEXT,
    _coverage,
    _measurability_totals,
    _next_measurement,
    useful_measurability,
    _stack_score,
    build_chain,
    codex_skill_usage,
    sensitivity,
    unmatched_measurements,
    stack_cost,
)

REPORT_SCHEMA_VERSION = "0.8"

# `_render` cannot print the report's own total before the rendering it is part of has
# finished, so it writes this and `profile()` substitutes the real figure afterwards. The
# old code read the figure straight from the dict, which meant it structurally excluded
# rendering — and on a synthetic 9,000-component stack that was most of the time it spent.
ELAPSED_PLACEHOLDER = "@@elapsed_ms@@"

# Limitations established by re-testing the project's own claims against real
# transcripts. Each is a confirmed gap in what the harness records, stated
# with the evidence that confirmed it — not an assumption about what might be missing.
#
# One entry was REMOVED after being disproved: "Codex 的指令文件是否载入" sat here on the
# evidence that `session_meta.base_instructions` is empty in every session examined. The
# field is empty and the instructions still enter the prefix — injecting a real AGENTS.md
# moves it by a reproducible amount matching the file's own size. A blind spot that has
# been closed must leave this list, or the
# report keeps telling the user it cannot answer something it just answered.
#
# Each entry also carries `requires`: the component class whose absence makes the gap moot.
# A blind spot about something the user does not have is not a limitation but an invented
# one — and this list was reporting "hook 是否执行过" into an environment with **zero** hooks
# configured, manufacturing a gap out of nothing.


def _confirmed_blind_spots():
    return (
    {
        # Narrowed after testing: an injected hook *does* fire, and its own side effect
        # proves it without any harness record (SessionStart and UserPromptSubmit both
        # fired; a no-hooks control produced nothing). So the gap is retrospective only —
        # what already happened in sessions that recorded nothing is unrecoverable, while
        # whether a hook fires is directly testable.
        "what": tr(en="whether a hook ever fired in a past session",
                   zh="历史会话里 hook 是否执行过"),
        "evidence": tr(
            en="every transcript was scanned and no hook-related field appears at any "
               "level; an injected hook does fire (its own side effect proves it), so what "
               "is missing is the historical record, not the observability",
            zh="扫过全部 transcript，任意层级都没有 hook 相关字段；"
               "注入的 hook 确实会执行（靠它自己的副作用即可证明），"
               "所以缺的是历史记录，不是可观测性"),
        "requires": "hooks",
    },
    {
        # Narrowed, not removed. Codex genuinely emits no tool call naming a skill — that
        # part held up. What was wrong was concluding nothing is observable: Codex uses a
        # skill by reading its SKILL.md, and those reads are in the rollouts. They are
        # attributed by installed path and sessions that also wrote to the file are set
        # aside, which is what the earlier "544 hits were development" finding was about.
        # So the gap is now the narrow one: a read is not an invocation.
        "what": tr(en="on the Codex side, \"read a skill\" and \"was editing that skill\" "
                      "cannot be separated perfectly",
                   zh="Codex 侧「读了 skill」和「在改这个 skill」分不到 100% 干净"),
        "evidence": tr(
            en="Codex emits no tool call naming a skill — using one means cat-ing its "
               "SKILL.md; after attributing by installed path and setting aside sessions "
               "that also wrote to the file, a residue remains: editing an installed skill "
               "also reads it",
            zh="Codex 不发出指名 skill 的工具调用，用 skill 就是 cat 它的 SKILL.md；"
               "按安装路径归因、并把同一会话里写过该文件的剔除之后仍有残留："
               "改一个已安装的 skill 也会读它"),
        "requires": "codex_skills",
    },
    {
        # Narrowed by measurement, not by argument. It sat here on "no record shows the body
        # being loaded", which is absence of evidence. The listing stage is measurable, and
        # where it has been measured a skill's per-request cost comes out an order of
        # magnitude below its body, so at listing time the body is not in the prefix and the
        # description is. What remains unobserved is the other half: whether a skill matched
        # by description later has its body read in, which no record reports either way.
        #
        # The figures stay out of this string on purpose. It is printed on whatever machine
        # runs the report, so a measurement from another stack quoted here would read as
        # this stack's own — the overclaim this whole list exists to prevent. `measure-skills`
        # puts the reader's own numbers in the cost section.
        "what": tr(en="for a skill matched by its description, whether the body enters the "
                      "context afterwards",
                   zh="被描述匹配中的 skill，正文是不是随后才进上下文"),
        "evidence": tr(
            en="the listing stage is measurable, and where it has been measured a skill "
               "costs an order of magnitude less per request than its body is long — so "
               "what enters the listing is the description, not the body; but whether the "
               "body is read in after a match is recorded nowhere",
            zh="列表阶段是可测的，而在已测过的地方，一个 skill 每次请求的开销比它正文的长度"
               "低一个数量级——所以列表里进去的是描述不是正文；"
               "但「命中之后正文有没有被读进来」没有任何记录"),
        "requires": "skills",
    },
    {
        # Verified, not assumed: every `Skill` call in the corpus carries only
        # `{skill, args}`, and the listing's `digests` could not be reproduced from the
        # files on disk with six hash algorithms over several input forms. The files under
        # the two roots mostly differ, so a digest *would* distinguish them if the
        # algorithm were known — this is a closed door, not a missing fact.
        "what": tr(en="when one skill name is installed under two roots, which copy was "
                      "invoked",
                   zh="同名 skill 装在两个 root 下时，调用的是哪一份"),
        "evidence": tr(
            en="a Skill call records only the name (every single one in the corpus carries "
               "just {skill, args}); the harness's own listing shows each name once, which "
               "means it has already deduplicated the two roots; the listing's `digests` "
               "field could not be matched to the files on disk with 6 hash algorithms",
            zh="Skill 调用只记名字（全语料每一次都只有 {skill, args}）；"
               "宿主程序自己的列表里每个名字只出现一次，说明它已经把两个 root 去重了；"
               "列表里的 digests 字段试了 6 种哈希也对不上磁盘文件"),
        "requires": "duplicate_skills",
    },
    )


def _present_classes(scan: dict, components: list[dict]) -> set[str]:
    """Which component classes this environment actually has.

    Read from the inventory rather than assumed, so a gap is only reported for something
    that exists to have a gap about.
    """
    by_kind = (scan.get("observed") or {}).get("by_kind") or {}
    present = set()
    if by_kind.get("hooks") or any(
            "hooks" in ((f.get("features") or []) if isinstance(f, dict) else [])
            for f in scan.get("files") or []):
        present.add("hooks")
    if by_kind.get("skills"):
        present.add("skills")
    if any(c["kind"] == "skills" and "codex" in (c.get("harnesses") or [])
           for c in components):
        present.add("codex_skills")
    # Only a stack that actually installs one name twice has the attribution problem.
    seen = Counter((c.get("kind"), c.get("name")) for c in components if c.get("name"))
    if any(n > 1 for n in seen.values()):
        present.add("duplicate_skills")
    return present


# Functions rather than module constants: a `tr()` evaluated at import time would bake in
# whichever language the importing process happened to default to, before the CLI has had a
# chance to call `set_lang()`. The same applies to every label table below.
def _link_labels():
    return {
        "installed": tr(en="installed", zh="装了"),
        "loaded": tr(en="loaded", zh="载入过"),
        "used": tr(en="used", zh="用过"),
        "useful": tr(en="useful", zh="有用"),
        "cost": tr(en="cost", zh="成本"),
        "recommendation": tr(en="verdict", zh="结论"),
    }


def _status_labels():
    return {PROVEN: tr(en="proven", zh="已证明"),
            PARTIAL: tr(en="partial", zh="部分"),
            UNPROVEN: tr(en="unproven", zh="未证明")}


# The compliance checks, in the report's own language. The artifact keeps its English
# identifiers — a quoted figure should read the same here as in the file it came from — but
# the *verdict* section is the one place a reader goes for a conclusion, and it was rendering
# a three-line English paragraph into Chinese prose. An unreadable conclusion is a conclusion
# nobody acts on, which is the same outcome as not having one.
def _rule_labels():
    """Readable names for the rule kinds, with no limit or spelling baked in.

    An earlier version wrote the numbers into these labels ("within 50 characters"), which
    were one author's limits copied into the tool. A limit belongs to the file being
    measured, so where a verdict needs one it comes from the measurement.
    """
    return {
        "subject_ends_with_reference": tr(
            en="the subject line ends with an issue reference",
            zh="标题末尾带议题编号"),
        "subject_length_limit": tr(en="the subject stays within its stated length",
                                   zh="标题不超过它自己写明的长度"),
        "body_is_itemised": tr(en="the body is written as one item per line",
                               zh="正文一行一件事"),
        "body_line_length_limit": tr(en="each body line stays within its stated length",
                                     zh="正文每行不超过它自己写明的长度"),
    }


def _useful_in_words(link: dict) -> str:
    """Render a proven `useful` link as the sentence a person would say.

    Built from the structured fields rather than by translating the evidence string, so it
    cannot drift from the numbers: `rules_proven` names the rules, and the separation is
    what "proven" means here — every valid run with the file, none without it.
    """
    rules = [_rule_labels().get(r, r) for r in link.get("rules_proven") or []]
    if not rules:
        return link["evidence"]
    listed = tr(en=", ".join(f"“{r}”" for r in rules),
                zh="、".join(f"「{r}」" for r in rules))
    return tr(en=f"Across 2 unrelated tasks, the arm carrying this file did {listed} every "
                 f"time and the arm without it never did. A control question (one piece of "
                 f"arithmetic) was answered correctly by both arms, which is what shows the "
                 f"two were comparable rather than one of them being broken.",
              zh=f"在 2 个互不相关的任务上，带这个文件的那一组{listed}每次都做到，"
                 f"不带的那一组一次都没做到。对照组（一道算术题）两组都答对，"
                 f"说明两组本身是可比的，不是有一组坏了。")


# The actionable section is the report's own advice, so its method names are localized.
# Unknown methods fall through verbatim rather than being dropped.
def _method_labels():
    return {
        "collect traces for this root": tr(en="collect traces for this root",
                                           zh="采集这个路径的会话记录"),
        "collect traces for the declaring harness": tr(
            en="collect traces for the harness that declares it",
            zh="采集声明它的那个宿主程序的会话记录"),
        "stage 6 paired probe": tr(en="run the Stage 6 paired probe (`evaluate --measure`)",
                                   zh="跑 Stage 6 的成对探针（`evaluate --measure`）"),
    }


def _collect(root: str, limit: int | None, evaluation_path: str | None,
             *, whole_environment: bool = False, exclusions: dict | None = None,
             useful_path: str | None = None, use_cache: bool = True
             ) -> tuple[dict, dict, dict | None, dict | None, dict, dict | None]:
    """Run the deterministic stages once each, so nothing is collected twice."""
    from .core import inspect as inspect_stack
    from .events import codex_rollups
    from .score import codex_skill_paths
    from .traces import collect_root

    scan = inspect_stack(root, whole_environment=whole_environment, exclusions=exclusions)
    traces = collect_root(root, limit, whole_environment=whole_environment,
                          use_cache=use_cache)
    # Codex is not covered by the Claude collector. Its rollouts carry two separate things
    # this report needs — the one harness-recorded shared-context cost in the project, and
    # the skill-body reads that are the only evidence Codex leaves of using a skill — and
    # both come from the same events. They used to be collected by two calls, which parsed
    # the whole Codex corpus twice — tens of seconds a pass on a corpus of any size —
    # directly against what this function's docstring says. One pass now feeds both.
    shared, codex_usage = codex_rollups(codex_skill_paths(scan),
                                        None if whole_environment else Path(root),
                                        use_cache=use_cache)
    evaluation = None
    if evaluation_path:
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    useful = None
    if useful_path:
        useful = json.loads(Path(useful_path).read_text(encoding="utf-8"))
    return scan, traces, evaluation, useful, shared, codex_usage


def _cost_lines(traces: dict, evaluation: dict | None,
                shared: dict | None = None) -> list[dict]:
    """Spend, with observed totals and projections kept apart.

    A projection carries the corpus it was computed over. If that is not the corpus this
    report just collected, saying "on this corpus" would be false — most easily hit when a
    project has moved and its history is filed elsewhere.
    """
    observed = traces["observed"]
    sessions = observed["sessions_collected"]
    spent_label = tr(en="tokens actually spent", zh="实际消耗的 token")
    if sessions:
        lines = [{
            "label": spent_label,
            "value": f"{observed['tokens']['total_tokens']:,}",
            "strength": "observed",
            "detail": tr(
                en=f"{count(sessions, 'session')}, "
                   f"{count(observed['assistant_messages'], 'model response')}; "
                   f"counted once per message.id and cross-checked against the harness's "
                   f"own accounting",
                zh=f"{sessions} 个会话，{observed['assistant_messages']} 次模型响应；"
                   f"按 message.id 去重后统计，与宿主程序自己的账目交叉验证过"),
        }]
    else:
        lines = [{
            "label": spent_label,
            "value": tr(en="cannot be counted", zh="无法统计"),
            "strength": "observed",
            "detail": tr(
                en="no session record was collected for this path, so this is \"no data\", "
                   "not \"spent nothing\"",
                zh="这个路径下没有采集到任何会话记录，所以不是「花了 0」，而是「没有数据」"),
        }]

    # Shared context blocks. Both of these are real recurring costs that were being
    # collected and never shown: a reader saw an empty cost section and concluded nothing
    # was known, when two figures were sitting in the data. Neither is divided among the
    # components it covers, so each says outright that it is a joint cost.
    shared_chars = observed.get("shared_context_chars") or {}
    listing_chars = shared_chars.get("skill_listing")
    if listing_chars and sessions:
        # This figure is a **cumulative** total across every collected session, not the size
        # of one listing. Printing it beside a per-request number would read as though one
        # listing costs 1.8 million characters, which is the aggregation trap this project
        # has already fallen into with tokens and with dollars. Both forms are given, each
        # labelled with what it is.
        per_session = round(listing_chars / sessions)
        lines.append({
            "label": tr(en="Claude Code's skill listing (shared)",
                        zh="Claude Code 的 skill 列表（共享）"),
            "value": tr(en=f"about {per_session:,} characters per session",
                        zh=f"每个会话约 {per_session:,} 字符"),
            "strength": "observed",
            "detail": tr(
                en=f"read straight out of the attachment records, free. {listing_chars:,} "
                   f"characters across {count(sessions, 'session')} in total; the "
                   f"figure above is "
                   f"the average — a cumulative total is not a per-request cost. This is "
                   f"one shared block covering every skill in the listing and is not "
                   f"divided per skill, because dividing it would invent an attribution. "
                   f"Reported in characters rather than tokens: no tokenizer runs here",
                zh=f"从 attachment 记录直接读出，免费。{sessions} 个会话累计 "
                   f"{listing_chars:,} 字符，上面那个是平均值 —— 累计数不是单次成本。"
                   "这是一整块共享上下文，覆盖列表里的所有 skill，没有按 skill 拆分，"
                   "拆就等于凭空造一个归因。报的是字符数而不是 token，这里没有跑分词器"),
        })

    if shared and shared.get("skill_listing_tokens"):
        tokens = shared["skill_listing_tokens"]
        covered = len(shared.get("skills_covered") or [])
        value = (tr(en=f"{tokens[0]:,} tokens", zh=f"{tokens[0]:,} token")
                 if len(tokens) == 1
                 else tr(en=f"{min(tokens):,}–{max(tokens):,} tokens",
                         zh=f"{min(tokens):,}–{max(tokens):,} token"))
        lines.append({
            "label": tr(en="Codex's skill listing (shared)", zh="Codex 的 skill 列表（共享）"),
            "value": tr(en=f"{value}/request", zh=f"{value}/次"),
            "strength": "observed",
            "detail": tr(
                en=f"measured by the harness itself and recorded in "
                   f"state.host_skills.usage — free and exact, covering "
                   f"{count(covered, 'skill')}; "
                   f"also one shared block, not divided per skill. Note: only "
                   f"{shared['sessions_carrying_record']} of "
                   f"{shared['sessions_scanned']} Codex sessions carry this record, so it "
                   f"describes those sessions rather than the whole history",
                zh=f"宿主程序自己量好记在 state.host_skills.usage 里的，免费且精确，"
                   f"覆盖 {covered} 个 skill；同样是共享的一块，不按 skill 拆分。"
                   f"注意：{shared['sessions_scanned']} 个 Codex 会话里只有 "
                   f"{shared['sessions_carrying_record']} 个带这条记录，"
                   f"所以它描述的是这几个会话，不是全部历史"),
        })

    eval_corpus = (evaluation or {}).get("corpus") or {}
    eval_sessions = eval_corpus.get("sessions")
    same_corpus = eval_sessions is None or eval_sessions == sessions

    for entry in (evaluation or {}).get("evaluations") or []:
        # A measurement that was attempted and came back with nothing must not look the
        # same as one that was never attempted, so it gets a line of its own.
        schema_label = tr(en=f"MCP `{entry['server']}` tool-schema size",
                          zh=f"MCP `{entry['server']}` 的 tool schema 体积")
        if entry.get("status") in {"measurement_inconclusive", "measurement_failed"}:
            lines.append({
                "label": schema_label,
                "value": tr(en="measured, but nothing came out", zh="测了，但没测出来"),
                "strength": "observed",
                "detail": entry.get("reason") or tr(
                    en="the probe returned no usable sample", zh="探针没有返回可用样本"),
            })
            continue
        if entry.get("status") != "measured":
            continue
        projection = entry["projection"]
        lines.append({
            "label": schema_label,
            "value": tr(en=f"{projection['schema_tokens']:,} tokens",
                        zh=f"{projection['schema_tokens']:,} token"),
            "strength": "observed",
            "detail": tr(en="measured directly by paired probes, consistent across runs",
                         zh="成对探针直接实测，多次运行一致"),
        })
        scope = (tr(en="cumulative cost on this corpus", zh="在本语料上的累计成本")
                 if same_corpus
                 else tr(en="cumulative cost (a different corpus)",
                         zh="累计成本（另一批语料）"))
        detail = tr(en="; ".join(projection["assumptions"]),
                    zh="；".join(projection["assumptions"]))
        if not same_corpus:
            detail = tr(
                en=f"this figure was computed over a corpus of "
                   f"{count(eval_sessions, 'session')} "
                   f"while only {sessions} were collected here; they are not the same "
                   f"corpus, so it is not this path's cost. Other assumptions: {detail}",
                zh=f"这个数字是在 {eval_sessions} 个会话的语料上算的，"
                   f"而本次只采集到 {sessions} 个，两者不是同一批，不能当作本路径的成本。"
                   f"　其余假设：{detail}")
        lines.append({
            "label": f"MCP `{entry['server']}` {scope}",
            "value": tr(
                en=f"{projection['total_tokens']:,} tokens ≈ "
                   f"${projection['estimated_usd']}",
                zh=f"{projection['total_tokens']:,} token ≈ "
                   f"${projection['estimated_usd']}"),
            "strength": "estimated",
            "detail": detail,
        })
    return lines


def _unanswerable(components: list[dict], link_coverage: dict) -> list[str]:
    """What the report cannot say, derived from the data rather than written by hand."""
    out: list[str] = []
    useful = link_coverage["useful"]
    joiner = tr(en=", ", zh="、")
    if useful["proven"] == 0:
        out.append(tr(
            en=f"**No component's usefulness is proven** (all {len(components)} are "
               f"unproven). So this report gives no component a keep-or-drop verdict and "
               f"does not rate whether one is worth having — that would need proof that "
               f"\"the work gets done without it\", and this project cannot produce that "
               f"evidence.",
            zh=f"**没有任何组件的「有用」得到证明**（{len(components)} 个组件全部未证明）。"
               "所以这份报告不对任何组件下去留结论，也不评价它值不值 —— 那需要先证明"
               "「缺了它这活照样干得成」，本项目拿不出这个证据。"))
    else:
        # A proven `useful` is the project's newest and most easily over-read result, so the
        # limits travel with the count rather than sitting in a footnote: it covers only
        # instruction files, only the rules those files state, and only tasks written here.
        proven_names = [c["name"] for c in components
                        if c["links"]["useful"]["status"] == PROVEN]
        # "All stuck at the same place" was the sentence this replaces, and it stopped being
        # true once the question was asked per component instead of per kind: some state a
        # rule a machine can check and some cannot stated anything, and those two need
        # different acts from the reader. The reasons are read off disk, so the split costs
        # nothing and does not depend on anything having been run.
        for component in components:
            component.setdefault("useful_measurability",
                                 useful_measurability(component, components))
        totals = _measurability_totals(components)
        reasons = totals["reasons"]
        # Noun phrases, so the sentence stays grammatical whether a reason covers one
        # component or twenty.
        why_en = [(reasons.get(NO_TEXT), "with no text of their own to quote"),
                  (reasons.get(NO_RULE), "stating no rule a machine can check"),
                  (reasons.get(EMPTY_TEXT), "empty")]
        why_zh = [(reasons.get(NO_TEXT), "自己没有可供引用的文本"),
                  (reasons.get(NO_RULE), "文本里没有机器能判定的规则"),
                  (reasons.get(EMPTY_TEXT), "是空文件")]
        split_en = "; ".join(f"{n} {label}" for n, label in why_en if n)
        split_zh = "；".join(f"{n} 个{label}" for n, label in why_zh if n)
        ambiguous = sorted({c["name"] for c in components
                            if (c.get("useful_measurability") or {})
                            .get("result_is_about_injected_copy")})
        copies_en = (f" One of them ({', '.join(ambiguous)}) is installed in copies that "
                     f"differ, and which one the harness loaded is not recorded, so a result "
                     f"would be about the copy that was injected." if ambiguous else "")
        copies_zh = (f"不过 {'、'.join(ambiguous)} 装了内容不同的多份，而哪一份被加载没有记录，"
                     f"所以测出来的结论只属于被注入的那一份。" if ambiguous else "")
        out.append(tr(
            en=f"**Usefulness is proven for {len(proven_names)} only, and in a narrow "
               f"sense** ({', '.join(proven_names)}). What is proven is that without this "
               f"file the agent does not follow the conventions you wrote and with it it "
               f"does — a behavioural cause, judged by machine-checkable checks, with no "
               f"model acting as referee. It does not mean \"these conventions are worth "
               f"following\": the standard is the file's own, and you wrote it. The "
               f"remaining {useful['unproven']} are unproven, and not for one reason. "
               f"{totals['measurable']} "
               f"{'states' if totals['measurable'] == 1 else 'state'} a rule a machine can "
               f"check and could be "
               f"measured the same way — the arms are verified: MCP can load one named "
               f"server (`--strict-mcp-config`), and a skill can be loaded alone "
               f"(`--setting-sources \"\"` clears the user's skills, `--plugin-dir` adds "
               f"one back, confirmed with a nonce).{copies_en} The other "
               f"{totals['not_measurable']} cannot be measured by any method here, and the "
               f"reason is on disk rather than in the schedule: {split_en}. That second "
               f"group is not unfinished work — nothing this tool can run would change it, "
               f"so a reader who treats it as a to-do spends money for no answer.",
            zh=f"**「有用」只证明了 {len(proven_names)} 个，而且是窄义的**"
               f"（{'、'.join(proven_names)}）。"
               "证明的是「不给这个文件，agent 就不按你写的规范做；给了就按」—— 这是行为上的"
               "因果，用机器可校验的检查判定，没有让模型当裁判。但它不等于「这些规范值得遵守」："
               "标准是这个文件自己定的，而那是你写的。"
               f"剩下 {useful['unproven']} 个未证明，但原因不是同一个。"
               f"其中 {totals['measurable']} 个自己写了机器能判定的规则，可以照同样的方式测"
               " —— 实验臂是验证过的：MCP 能只加载指定 server（`--strict-mcp-config`），"
               "skill 能只加载指定一个（`--setting-sources \"\"` 清掉用户 skill，"
               f"`--plugin-dir` 加回一个，已用 nonce 确认）。{copies_zh}"
               f"另外 {totals['not_measurable']} 个用这里的任何方法都测不了，"
               f"而原因写在磁盘上、不在排期上：{split_zh}。"
               "后一组不是没干完的活 —— 这个工具能跑的任何测量都改不了它，"
               "把它当待办去花钱，买不到答案。"))
    unrecordable = sorted({
        c["kind"] for c in components
        if c["links"]["loaded"]["blocker"] == NO_EVENT_EMITTED
    })
    if unrecordable:
        out.append(tr(
            en=f"**Whether {joiner.join(unrecordable)} was loaded cannot be determined**, "
               f"because the harness does not record it. That is a gap in the records, not "
               f"in the analysis.",
            zh=f"**{'、'.join(unrecordable)} 的载入情况无法判断**，因为宿主程序不记录这件事。"
               "这不是分析不到位，是记录里没有。"))
    excluded = [c["name"] for c in components if c.get("excluded_by_request")]
    if excluded:
        out.append(tr(
            en=f"**{joiner.join(excluded)} has no measured cost**, because you asked this "
               f"tool not to contact it — no probe was started and nothing was sent to it. "
               f"Its tool schema still occupies context in every session that loads it, so "
               f"that cost is real; it simply was not measured here.",
            zh=f"**{'、'.join(excluded)} 的成本没测**，因为你要求这个工具不去接触它 —— "
               "没有发起任何探针，"
               "也没有向它发送任何东西。它的 tool schema 在每个载入它的会话里仍然占着上下文，"
               "这笔开销是真实存在的，只是这里没有量。"))
    partial_used = [c["name"] for c in components if c["links"]["used"]["status"] == PARTIAL]
    if partial_used:
        more = len(partial_used) > 4
        out.append(tr(
            en=f"**\"Unused\" is not a fact for "
               f"{count(len(partial_used), 'skill')}** "
               f"({', '.join(partial_used[:4])}{' and more' if more else ''}). An explicit "
               f"call is recorded, so use is hard evidence; but loading by description "
               f"match leaves no record, so zero calls does not mean it did nothing.",
            zh=f"**{len(partial_used)} 个 skill 的「没用过」不能当事实**"
               f"（{'、'.join(partial_used[:4])}{' 等' if more else ''}）。"
               "显式调用有记录，所以用过是硬证据；但描述匹配自动载入不留记录，"
               "所以零次调用不等于没起作用。"))
    untraced = [c["name"] for c in components
                if c["links"]["used"]["blocker"] == "harness_not_traced"]
    if untraced:
        more = len(untraced) > 4
        out.append(tr(
            en=f"**{count(len(untraced), 'component is', 'components are')} declared to "
               f"a harness that was not analysed** "
               f"({', '.join(untraced[:4])}{' and more' if more else ''}), so their "
               f"use was not examined this time.",
            zh=f"**{len(untraced)} 个组件声明给了没被分析的宿主程序**"
               f"（{'、'.join(untraced[:4])}{' 等' if more else ''}），"
               "它们的使用情况这次没查。"))
    return out


def _next_steps(components: list[dict]) -> list[dict]:
    """Measurements that would close a gap — never advice about changing the stack."""
    steps: dict[str, dict] = {}
    for component in components:
        nxt = component.get("next_measurement")
        if not nxt or not nxt.get("achievable"):
            continue
        key = f"{nxt['link']}:{nxt['method']}"
        entry = steps.setdefault(key, {
            "link": nxt["link"], "method": nxt["method"],
            "estimated_usd": nxt["estimated_usd"], "note": nxt["note"], "components": [],
        })
        entry["components"].append(component["name"])
    return sorted(steps.values(), key=lambda s: (s["estimated_usd"] or 0.0, s["link"]))


def _verdicts(components: list[dict], profile: str) -> list[dict]:
    """The chain's last link, for the components that actually reached it.

    The report computed scores and rendered none of them, which was defensible only while
    no component was scorable. A verdict is emitted only where `used`, `useful` and `cost`
    are all present, and it says what it rests on — a proven `useful` is the behavioural
    evidence, and the cost is the measured per-request figure, in its own unit.

    Only "keep" can be reached from this evidence. Recommending removal would need the
    usefulness probe to have run its controls and found *no* effect, which is a state no
    component is in: unproven here means not measured or refused, never disproven.
    """
    out = []
    for component in components:
        if not component["coverage"]["scorable"]:
            continue
        scored = component["sensitivity"]["profiles"].get(profile)
        if not scored or not scored["scored"]:
            continue
        links = component["links"]
        cost = links["cost"]
        out.append({
            "name": component["name"],
            "kind": component["kind"],
            "verdict": "keep",
            "score": scored["score"],
            "signals": scored["signals"],
            "stable_across_profiles": component["sensitivity"].get("stable"),
            "cost_value": cost["value"],
            "cost_unit": cost.get("unit"),
            "cost_status": cost["status"],
            # One field, in the reader's language. This used to be two — `basis` plus a
            # `basis_zh` bolted on when the verdict section was the only place rendering an
            # English paragraph into Chinese prose. With a language layer the patch is the
            # bug: two fields invite the renderers to disagree about which one is current.
            "basis": _useful_in_words(links["useful"]),
            # The upstream evidence string, kept verbatim for a reader checking the
            # artifact rather than reading the report.
            "basis_evidence": links["useful"]["evidence"],
            # Worded to avoid the banned substrings in both languages rather than to
            # explain them away: the guard greps, so it cannot tell a negation from a
            # claim, and this project's rule is that the wording changes, not the test.
            "limit": tr(
                en="this verdict can only say \"keep\". The opposite would need the "
                   "usefulness experiment to run its controls and measure an absence of "
                   "effect — no component is in that state, and unproven here means not "
                   "measured, or measured and rejected.",
                zh="这条结论只说「保留」。相反的结论需要有用性实验跑完对照后测出没有效果 —— "
                   "没有任何组件处于这个状态，这里的未证明指的是没测，或者测了但样本被判无效。"),
        })
    return sorted(out, key=lambda v: -v["score"])


# Which installed roots each harness reads. `shared` marks `~/.agents/skills`, read by both,
# which is why ambiguity has to be judged per harness: Claude Code sees `.claude` + `.agents`
# and Codex sees `.codex` + `.agents`, so two copies under *different* harnesses are not two
# candidates for one listing. Judging this globally put `demo-skill` on the list on the strength
# of a Codex copy that Claude Code never sees.
HARNESS_VIEWS = {"claude_code": ("claude_code", "shared"), "codex": ("codex", "shared")}


def _diverged_duplicates(components: list[dict]) -> list[dict]:
    """One name, one harness, two installed files with **different content**.

    The only finding in this report that is both actionable and needs no `useful` link,
    because it is not a judgment about value at all. It states the environment's
    consistency, from three facts each established elsewhere in this project:

        the harness lists this name once, so one of these files is what loads
        the files differ
        which one loaded cannot be recovered — a `Skill` call carries only {skill, args},
        and the listing's digests did not reproduce from disk with six hash algorithms

    Together: the owner cannot know which version of their own instruction is in effect.
    Staleness is what makes that matter rather than merely untidy, and copies months apart
    do occur in practice.

    Where the copies differ in *what they instruct*, that is said too, from the stated rules
    each one carries. Dates and byte counts only establish that the files are not the same;
    a reader still has to open all of them to learn whether the difference matters. When one
    copy states three checkable rules and another states none, the ambiguity is not
    untidiness — it decides whether the agent was told the rule at all. This costs nothing:
    it is the same free classification the `useful` link uses, read from text on disk.

    Identical copies are excluded: whichever loads, the bytes are the same. That is
    redundancy, not ambiguity, and reporting it would be inventing a problem.

    No specific edit is recommended. The real configuration is the object under measurement
    and stays read-only; the report's job is to say the ambiguity exists.
    """
    from .score import stated_rules
    out = []
    for harness, roots in HARNESS_VIEWS.items():
        by_name: dict[tuple, set] = {}
        for component in components:
            if component.get("harness") not in roots or not component.get("path"):
                continue
            if not component.get("name"):
                continue
            by_name.setdefault((component["kind"], component["name"]),
                               set()).add(component["path"])
        for (kind, name), paths in sorted(by_name.items()):
            if len(paths) < 2:
                continue
            copies = []
            for path in sorted(paths):
                file = Path(path)
                try:
                    stat = file.stat()
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                except OSError:
                    # A path the inventory saw and the reader cannot is not something to
                    # guess about, so the whole name is skipped rather than half-reported.
                    copies = []
                    break
                copies.append({"path": path, "bytes": stat.st_size, "mtime": stat.st_mtime,
                               "modified": time.strftime("%Y-%m-%d",
                                                         time.localtime(stat.st_mtime)),
                               "digest": digest,
                               # Read per copy, not per name: the whole point is that these
                               # files may instruct differently, so one lookup on the
                               # component would answer for an unidentified copy.
                               "rules": stated_rules({"path": path})[1]})
            if len(copies) < 2 or len({c["digest"] for c in copies}) == 1:
                continue
            days = int((max(c["mtime"] for c in copies)
                        - min(c["mtime"] for c in copies)) // 86400)
            # Differing bytes are a given here; differing *instructions* are the part a
            # reader cannot see without opening every copy.
            rule_sets = {tuple(sorted(c["rules"])) for c in copies}
            out.append({
                "kind": kind, "name": name, "harness": harness, "copies_total": len(copies),
                "copies": [{k: v for k, v in c.items() if k != "mtime"} for c in copies],
                "days_apart": days,
                "rules_differ": len(rule_sets) > 1,
                # The sharpest case: one copy states a checkable rule and another states
                # none, so which file loaded decides whether the rule was given at all.
                "some_copy_states_no_rule": (len(rule_sets) > 1
                                             and any(not r for r in rule_sets)),
                # Only meaningful once the copies are at least a day apart: the report prints
                # dates, so a label derived from a sub-day mtime could not be checked against
                # what the reader is shown.
                "stale": min(copies, key=lambda c: c["mtime"])["path"] if days else None,
            })
    return sorted(out, key=lambda d: -d["days_apart"])


def _duplicate_lines(diverged: list[dict]) -> list[str]:
    if not diverged:
        return []
    lines = [tr(en="## One thing to sort out: the same skill is installed twice with "
                   "differing content",
                zh="## 该收拾的一件事：同一个 skill 装了两份，内容还不一样"), "",
             tr(en="This section has nothing to do with whether a component is worth "
                   "having, so it can be stated without the `useful` link. The harness "
                   "lists each name once, so it loads only one of the copies; the copies "
                   "differ; and which one was loaded **cannot be recovered** (a call "
                   "records the name, not the path). Together: you cannot be sure which "
                   "version is in effect.",
                zh="这一节跟「值不值得装」无关，所以不需要「有用」那一环就能说。"
                   "宿主程序对每个名字只列一次，也就是只加载其中一份；两份内容不同；"
                   "而哪一份被加载**查不出来**（调用记录只有名字，没有路径）。"
                   "合起来就是：你不能确定自己现在生效的是哪个版本。"), ""]
    for item in diverged:
        if item["days_apart"]:
            how = tr(en=f"**{count(item['days_apart'], 'day')} apart**",
                     zh=f"相差 **{item['days_apart']} 天**")
        else:
            how = tr(en="edited the same day, but the contents differ",
                     zh="同一天改过，但内容不同")
        lines.append(tr(en=f"- **{item['name']}** ({item['harness']} side, {how})",
                        zh=f"- **{item['name']}**（{item['harness']} 侧，{how}）"))
        for copy in sorted(item["copies"], key=lambda c: c["modified"], reverse=True):
            mark = (tr(en="older: ", zh="较旧：") if copy["path"] == item["stale"] else "")
            # The rule count goes beside each path only when the copies actually disagree.
            # Printing it on every row would read as a quality score for each file, which
            # it is not — a file may legitimately state nothing a machine can check.
            rules = ""
            if item.get("rules_differ"):
                found = len(copy.get("rules") or [])
                rules = (tr(en=", states no checkable rule", zh="，没有声明可判定的规则")
                         if not found else
                         tr(en=f", states {count(found, 'checkable rule')}",
                            zh=f"，声明了 {found} 条可判定的规则"))
            lines.append(tr(
                en=f"  - {mark}`{copy['path']}` ({copy['modified']}, "
                   f"{copy['bytes']:,} bytes{rules})",
                zh=f"  - {mark}`{copy['path']}`"
                   f"（{copy['modified']}，{copy['bytes']:,} 字节{rules}）"))
        if item.get("some_copy_states_no_rule"):
            lines.append(tr(
                en="  - → the copies do not instruct the same thing: one states a rule a "
                   "machine can check and another states none, so which file loaded decides "
                   "whether the rule was given at all.",
                zh="  - → 这几份拷贝给的指令不一样：一份声明了机器可判定的规则，另一份一条都"
                   "没有 —— 所以加载的是哪份，决定了这条规则到底有没有被下达。"))
        elif item.get("rules_differ"):
            lines.append(tr(
                en="  - → the copies state different rules, so which one loaded changes what "
                   "the agent was told to do.",
                zh="  - → 这几份拷贝声明的规则不同，所以加载哪一份会改变 agent 被要求做什么。"))
    lines += ["",
              tr(en="*How to sort it out is your call — this tool does not modify your real "
                    "configuration. Byte-identical copies are not listed: those are merely "
                    "duplicated, not ambiguous. The two harnesses read different "
                    "directories, so this is judged per harness.*",
                 zh="*怎么收拾由你定 —— 本工具不改你的真实配置。字节完全相同的副本没有列进来："
                    "那只是重复，不构成歧义。两个宿主程序各自读的目录不同，所以这里按宿主程序"
                    "分开判断。*"),
              ""]
    return lines


def _always_on_but_uncalled(components: list[dict]) -> list[dict]:
    """Components with a measured per-request cost, ordered by how often they were called.

    This is the one actionable thing this tool can produce without the `useful` link, and it
    works by being narrower than a verdict. It asserts two facts and stops:

        this costs N tokens on every request   (measured, per-request, not projected)
        you called it M times in S sessions    (counted from the traces)

    It does **not** say the component is unused. A skill matched by description auto-loads
    without leaving any record, so zero calls is not zero use — that limit is carried on
    every row rather than mentioned once. And it does not say "remove", because the evidence
    for that does not exist here and never did.

    Three structural rules:

    * **Only callable kinds.** An instruction file is never invoked, so a call count for it
      is not a low number — it is a category error. They are excluded, not reported as 0.
    * **One row per listed name.** A skill installed under two roots is two components and
      one listed skill; two rows would double it, exactly as the cost floor once did. The
      lower figure is kept, like the floor, because both are a bound.
    * **Other harnesses are named.** `demo-skill` is never explicitly invoked under Claude
      Code *and* its Codex copy was read 3 times. A row showing only the zero would be
      true and misleading.
    """
    rows: dict[tuple, dict] = {}
    elsewhere: dict[str, list[str]] = {}
    for component in components:
        kind, name = component.get("kind"), component.get("name")
        used = component["links"]["used"]
        if kind in INSTRUCTION_KINDS:
            continue
        cost = component["links"]["cost"]
        if cost.get("unit") != "tokens_per_request" or cost.get("value") is None:
            # Not measured here, so there is no "costs N" half of the claim. The usage
            # evidence is still worth keeping, for the row the measured copy will render.
            if used["status"] != UNPROVEN and used.get("value"):
                elsewhere.setdefault(name, []).append({
                    "harness": component["harness"], "calls": used["value"],
                    "status": used["status"],
                })
            continue
        row = rows.get((kind, name))
        if row is None or cost["value"] < row["tokens"]:
            rows[(kind, name)] = {
                "name": name, "kind": kind, "tokens": cost["value"],
                "harness_label": cost.get("harness") or component.get("harness"),
                "calls": used.get("value"), "used_status": used["status"],
                "used_evidence": used["evidence"],
                "ambiguous": "installed under" in (used["evidence"] or ""),
            }
    for row in rows.values():
        row["also"] = elsewhere.get(row["name"]) or []
    # Fewest calls first: that is the whole ordering the reader cares about.
    return sorted(rows.values(), key=lambda r: (r["calls"] if r["calls"] is not None else -1,
                                                -r["tokens"]))


def _always_on_lines(rows: list[dict], sessions: int) -> list[str]:
    if not rows:
        return []
    lines = [tr(en="## Components paid for on every request, ordered by how often you "
                   "called them",
                zh="## 每次请求都在付钱的组件，按你叫过它几次排序"), "",
             tr(en=f"This section says two things only: **how many tokens it takes per "
                   f"request** (measured) and **how many times you called it by name "
                   f"across {count(sessions, 'session')}** (counted). Both are hard "
                   f"facts, and "
                   f"together they still do not amount to a recommendation — that judgment "
                   f"is yours, because only you know why you installed it.",
                zh="这一节只说两件事：**它每次请求占多少 token**（实测），"
                   f"以及**你在 {sessions} 个会话里显式叫过它几次**（数出来的）。"
                   "两件都是硬的，合起来仍然不构成一条处置建议 —— "
                   "判断得你来做，因为只有你知道当初为什么装它。"), ""]
    for row in rows:
        calls = row["calls"]
        if calls == 0:
            headline = tr(en="**0 explicit calls**", zh="**显式调用 0 次**")
        elif calls is None:
            headline = tr(en="call count unknown", zh="调用次数不详")
        else:
            headline = tr(en=count(calls, "explicit call"),
                          zh=f"显式调用 {calls} 次")
        suffix = (tr(en=" (this name is installed under two roots, so these calls cannot be "
                        "attributed to either copy)",
                     zh="（这个名字装在两个 root 下，这些调用分不清是哪一份）")
                  if row["ambiguous"] else "")
        lines.append(tr(
            en=f"- **{row['name']}** ({row['kind']}): {row['tokens']:,} tokens/request, "
               f"{headline}{suffix}",
            zh=f"- **{row['name']}**（{row['kind']}）：{row['tokens']:,} token/次，"
               f"{headline}{suffix}"))
        for also in row["also"]:
            how = (tr(en="read its SKILL.md", zh="读了它的 SKILL.md")
                   if also["harness"] == "codex" else tr(en="called it", zh="调用过"))
            lines.append(tr(
                en=f"  - but the {also['harness']} copy {how} "
                   f"{count(also['calls'], 'time')}, so "
                   f"the skill itself is not unused — it is the {row['tokens']:,} "
                   f"tokens/request on the {row['harness_label']} side that has no "
                   f"explicit call behind it",
                zh=f"  - 但 {also['harness']} 那一份{how} {also['calls']} 次，"
                   "所以这个 skill 本身不是没在用，"
                   f"是它在 {row['harness_label']} 这边的 {row['tokens']:,} token/次"
                   "没有对应的显式调用"))
    lines += ["",
              tr(en="*Zero calls does not mean it did nothing: loading by description match "
                    "leaves no record at all, so this column can only prove \"you never "
                    "asked for it by name\".*",
                 zh="*0 次不代表它没起作用：靠描述匹配自动载入不留任何记录，所以这一列只能证明"
                    "「你没有主动叫过它」。*"), ""]
    return lines


def _verdict_lines(verdicts: list[dict], profile: str) -> list[str]:
    if not verdicts:
        return []
    lines = [tr(en="## Components that can be given a verdict", zh="## 能给出结论的组件"), "",
             tr(en=f"These have used / useful / cost all three, scored under the "
                   f"`{profile}` weights. The weights are this tool's assumption, not "
                   f"preferences you supplied.",
                zh=f"下面是「用过、有用、成本」三环齐备的组件，按 `{profile}` 权重打分。"
                   f"权重是本工具的假设，不是你给的偏好。"), ""]
    for v in verdicts:
        units = {"tokens_per_request": tr(en="tokens/request", zh="token/次"),
                 "usd": tr(en="USD/request", zh="美元/次")}
        unit = units.get(v["cost_unit"], v["cost_unit"])
        cost = (f"{v['cost_value']:,} {unit}" if v["cost_value"] is not None
                else tr(en="unknown", zh="未知"))
        stable = (tr(en="same verdict under all three weightings", zh="三种权重下结论一致")
                  if v["stable_across_profiles"]
                  else tr(en="**flips under a different weighting**",
                          zh="**换一组权重结论就变**"))
        lines += [
            tr(en=f"- **{v['name']}: keep** ({profile} score {v['score']}, {stable})",
               zh=f"- **{v['name']}：建议保留**（{profile} 得分 {v['score']}，{stable}）"),
            tr(en=f"  - Cost: {cost}", zh=f"  - 成本：{cost}"),
            # Falls back to the quoted evidence rather than failing: a verdict without the
            # rendered sentence should still print its basis.
            tr(en=f"  - Basis: {v['basis']}", zh=f"  - 依据：{v['basis']}"),
        ]
    lines += ["", f"*{verdicts[0]['limit']}*", ""]
    return lines


def _next_steps_preamble(data: dict) -> str:
    """Why this section lists measurements rather than advice.

    The phrasing tracks the data through three states, because it has been wrong in two of
    them: it claimed the `useful` link did not exist after it was proven, and then claimed
    those components' cost was unmeasured after it was measured — while the verdict section
    two screens up was already publishing conclusions for them.
    """
    useful_proven = data["observed"]["link_coverage"]["useful"]["proven"]
    verdicts = len(data.get("verdicts") or [])
    if verdicts:
        return tr(
            en=f"The latter needs `useful` and `cost` together; {verdicts} component(s) "
               f"have both and their verdicts are above. What the rest are missing is below.",
            zh=f"后者需要「有用」和「成本」两环同时成立，目前有 {verdicts} 个组件"
               f"同时成立，结论在上面；其余组件缺的环节见下。")
    if useful_proven:
        return tr(
            en=f"The latter needs `useful` and `cost` together; `useful` is proven for "
               f"{useful_proven} so far, and those components' cost is not measured yet.",
            zh="后者需要「有用」和「成本」两环同时成立；"
               f"目前「有用」证明了 {useful_proven} 个，而这些组件的成本还没测出来。")
    return tr(en="The latter needs the `useful` link, which does not exist yet.",
              zh="后者需要「有用」这一环，还没有。")


def _installed_lines(installed: dict) -> list[str]:
    """The inventory, counted in the same units the component table is built from.

    `by_kind` counts files, and a skill is a directory of them: 12 skills appeared here as
    `skills：147` directly above a table of 12 rows. Entity kinds therefore report the entity
    count first and the file count beside it. Harness-bundled files are counted but are not
    the user's components, which is said rather than implied by an exclusion claim.
    """
    lines = [
        tr(en=f"- {count(installed['files'], 'config file')}, "
              f"{installed['bytes']:,} bytes "
              f"(session transcripts, runtime state, caches and dependency trees are not "
              f"counted)",
           zh=f"- 配置文件 {installed['files']} 个，{installed['bytes']:,} 字节"
              f"（已排除会话记录、运行时状态、缓存与依赖树）"),
    ]
    if installed.get("bundled_files"):
        lines.append(tr(
            en=f"  - {count(installed['bundled_files'], 'of them ships', 'of them ship')}"
               f" with the harness: counted in the file total, but not your components",
            zh=f"  - 其中 {installed['bundled_files']} 个是宿主程序自带的，"
               f"计在文件数里，但不算你的组件"))
    # `n`, not `count`: the loop variable would shadow the pluralization helper.
    for kind, n in sorted(installed["by_kind"].items()):
        entities = (installed.get("entities") or {}).get(kind)
        if entities is None:
            lines.append(tr(en=f"- {kind}: {count(n, 'file')}",
                            zh=f"- {kind}：{n} 个文件"))
            continue
        declared = installed["entity_files"][kind]
        extra = (tr(en=f", plus {n - declared} shipped with the harness",
                    zh=f"；另有 {n - declared} 个宿主程序自带的")
                 if n > declared else "")
        lines.append(tr(en=f"- {kind}: {entities} ({count(declared, 'file')}{extra})",
                        zh=f"- {kind}：{entities} 个（{declared} 个文件{extra}）"))
    return lines


def _render(data: dict) -> str:
    """The readable form. Structure mirrors the evidence chain, not a score."""
    o = data["observed"]
    lines = [
        tr(en="# Agent environment check-up", zh="# Agent 环境体检报告"),
        "",
        tr(en=f"Subject: `{data['root']}`", zh=f"对象：`{data['root']}`"),
        tr(en=f"Generated: {data['generated_at']} | read-only, no real configuration was "
              "touched",
           zh=f"生成时间：{data['generated_at']}　|　只读，没有改动任何真实配置"),
        "",
    ]

    # Before anything else, because an orphaned measurement makes the links below read
    # weaker than the evidence the reader actually paid for.
    orphans = data.get("unmatched_measurements") or []
    if orphans:
        lines += [
            tr(en=f"> **Note: {len(orphans)} supplied measurement(s) matched nothing here, "
                  "so they are not counted below.**",
               zh=f"> **注意：传入的 {len(orphans)} 项测量在这里一个都没匹配上，下面没有计入。**"),
            "",
        ]
        for orphan in orphans[:8]:
            lines.append(tr(en=f"> - `{orphan['subject']}` — {orphan['reason']}",
                            zh=f"> - `{orphan['subject']}` —— {orphan['reason']}"))
        if len(orphans) > 8:
            lines.append(tr(en=f"> - ... and {len(orphans) - 8} more",
                            zh=f"> - ...另有 {len(orphans) - 8} 项"))
        lines += [
            ">",
            tr(en="> A measurement is joined to a component by absolute path or server name. "
                  "An artifact recorded on another machine, under a different home "
                  "directory, or with its paths rewritten still parses — it just joins to "
                  "nothing, and the links it should have proven read as never measured.",
               zh="> 测量结果靠绝对路径或服务名跟组件对应。产物如果是在另一台机器、"
                  "另一个 home 目录下生成的，或者路径被改写过，它照样能解析 —— "
                  "只是一个都对不上，本该被证实的链环会显示成从未测量。"),
            "",
        ]

    if data["relocation_candidates"]:
        lines += [tr(en="> **Note: this path has almost no history, but another path does.**",
                     zh="> **注意：这个路径下几乎没有历史记录，但别处有。**"), ""]
        for c in data["relocation_candidates"]:
            lines.append(tr(
                en=f"> - `{c['recorded_cwd']}` has records for {c['sessions']} sessions",
                zh=f"> - `{c['recorded_cwd']}` 下有 {c['sessions']} 个会话的记录"))
        lines += [
            ">",
            tr(en="> Transcripts are filed by working directory, so a project that moved "
                  "leaves its old history behind. The `used` figures below may therefore be "
                  "empty — that is zero records, not genuinely unused.",
               zh="> 会话记录按工作目录归档，所以项目搬过家之后旧记录会留在原处。"
                  "下面的「用过」数据因此可能是空的 —— 这是零记录，不是真的没用过。"),
            "",
        ]

    lines += [
        # Findings come first, then the detail that backs them. The other order was the
        # report's original sin: a reader looking for "is my setup any good" met a coverage
        # table, and the evidence discipline below was enforced for nobody.
        *_overhead_line(data),
        *_finding_lines(data["findings"], data),
        # The verdict next: it is the one place a component gets a conclusion, and on an
        # environment with none this section simply does not exist.
        *_verdict_lines(data.get("verdicts") or [], data.get("profile", "balanced")),
        *_duplicate_lines(data["diverged_duplicates"]),
        *_always_on_lines(data["always_on"], data["sessions"]),
        *_reliability_lines(data["reliability"]),
        tr(en="## What is installed", zh="## 装了什么"),
        "",
        *_installed_lines(o["installed"]),
    ]
    lines += ["", tr(en="## Evidence-chain coverage", zh="## 证据链覆盖情况"), "",
              tr(en=f"*The detail behind \"what could not be determined\" above: "
                    f"{data['headline']}*",
                 zh=f"*这是上面那句「看不出来的」的明细：{data['headline']}*"), "",
              tr(en="| link | proven | partial | unproven |",
                 zh="| 环节 | 已证明 | 部分 | 未证明 |"),
              "| --- | --- | --- | --- |"]
    for link in SCORED_LINKS:
        c = o["link_coverage"][link]
        lines.append(f"| {_link_labels()[link]} | {c['proven']} | {c['partial']} | {c['unproven']} |")

    lines += ["", tr(en="## Component by component", zh="## 逐个组件"), "",
              tr(en="| component | kind | loaded | used | cost |",
                 zh="| 组件 | 类型 | 载入过 | 用过 | 成本 |"),
              "| --- | --- | --- | --- | --- |"]
    # The same skill name under two harnesses is deliberately two components, which in a
    # table without a harness column reads as a duplicated row with contradictory evidence.
    # Counted rather than re-scanned per component: the nested form was quadratic, and on a
    # synthetic 9,000-component stack it was most of the report's time.
    seen_names = Counter(c["name"] for c in data["components"])
    repeated = {name for name, times in seen_names.items() if times > 1}
    for component in data["components"]:
        links = component["links"]

        def cell(link: str) -> str:
            info = links[link]
            label = _status_labels()[info["status"]]
            if info["value"] not in (None, 0):
                return tr(en=f"{label} ({info['value']})", zh=f"{label}（{info['value']}）")
            if info["status"] == PROVEN and info["value"] == 0:
                return tr(en="0 calls", zh="0 次")
            return label

        cost = links["cost"]
        if cost["value"] is None:
            cost_cell = _status_labels()[cost["status"]]
        elif cost.get("unit") == "tokens_per_request":
            # Tokens, not dollars: this is a prefix size, and it is paid on every request.
            cost_cell = tr(en=f"{cost['value']:,} tokens/req",
                           zh=f"{cost['value']:,} token/次")
            # The rate alone says nothing about this stack's size. The request count that
            # turns it into a total is in the corpus, so the total belongs beside it — with
            # `≥` whenever traffic from another harness is not in the product.
            carried = cost.get("tokens_carried")
            if carried is not None:
                mark = "≥" if cost.get("carried_is_lower_bound") else ""
                cost_cell += tr(en=f" · {mark}{carried:,} carried",
                                zh=f" · 累计 {mark}{carried:,}")
        else:
            cost_cell = f"${cost['value']}"
        # An excluded component must say so here, not only in the JSON: a reader who asked
        # for something to be left alone needs to see that it was, and needs its blank cost
        # cell attributed to their own decision rather than to a gap in the tool.
        if component.get("excluded_by_request"):
            cost_cell = tr(en="excluded on request", zh="按要求排除")
        name = f"`{component['name']}`"
        if component["name"] in repeated:
            name += tr(en=f" ({component['harness']})", zh=f"（{component['harness']}）")
        if component.get("excluded_by_request"):
            name += tr(en=" (left untouched on request)", zh="（按要求不接触）")
        lines.append(
            f"| {name} | {component['kind']} | {cell('loaded')} | "
            f"{cell('used')} | {cost_cell} |"
        )

    lines += ["", tr(en="## What it costs", zh="## 花了多少"), ""]
    for entry in data["cost"]:
        tag = (tr(en="measured", zh="实测") if entry["strength"] == "observed"
               else tr(en="projected", zh="推算"))
        lines += [tr(en=f"- **{entry['label']}: {entry['value']}** ({tag})",
                     zh=f"- **{entry['label']}：{entry['value']}**（{tag}）"),
                  f"  - {entry['detail']}"]

    lines += _render_carried(data.get("carried") or [])

    # The stack-level cost is the one question that never needed per-component attribution,
    # so it is answerable where the per-component column is still empty. It is a floor, and
    # the word appears in the heading rather than in a footnote.
    harnesses = {h: v for h, v in (data.get("stack_cost") or {}).items()
                 if v.get("per_request_tokens_floor")}
    if harnesses:
        lines += ["", tr(en="## Fixed overhead per request (a floor)",
                         zh="## 每次请求的固定开销（下限）"), "",
                  tr(en="This is \"how many tokens this environment carries on every "
                        "request\". It needs no per-component attribution, so it is "
                        "answerable while the per-component column is still empty.",
                     zh="这是「这套环境每次请求要带多少 token」，不需要把成本摊到单个组件头上，"
                        "所以在逐组件那一列还空着的时候就能回答。"), ""]
        for harness, info in sorted(harnesses.items()):
            low = info["per_request_tokens_floor"]
            high = info.get("per_request_tokens_floor_high")
            total = f"{low:,}" + (f"–{high:,}" if high else "")
            lines.append(tr(en=f"- **{harness}: at least {total} tokens/request**",
                            zh=f"- **{harness}：至少 {total} token/次**"))
            for item in info["contributions"]:
                size = f"{item['tokens']:,}" + (
                    f"–{item['tokens_max']:,}" if item.get("tokens_max") else "")
                tag = (tr(en=f" (shared, covers {item['covers']} skills)",
                          zh=f"（共享，覆盖 {item['covers']} 个 skill）")
                       if item["shared"] else "")
                lines.append(tr(en=f"  - {item['name']}: {size} tokens{tag}",
                                zh=f"  - {item['name']}：{size} token{tag}"))
            for note in info["assumptions"]:
                lines.append(f"  - *{note}*")
        # A harness with no summable figure must say so, or its absence reads as zero.
        silent = [h for h, v in (data.get("stack_cost") or {}).items()
                  if not v.get("per_request_tokens_floor") and v.get("components_total")]
        if silent:
            # The reason has to be read from the data, not asserted. This sentence used to
            # state that Claude Code's per-component prefix cost was unmeasurable; the
            # moment it was measured, the report was telling the user the opposite of its
            # own numbers two sections above.
            listed = tr(en=", ".join(sorted(silent)), zh="、".join(sorted(silent)))
            lines += ["", tr(
                en=f"**{listed} cannot produce this figure**: no component under these "
                   "harnesses has a measured tokens-per-request. The only free figure here "
                   "is the skill listing, and it carries characters rather than tokens (no "
                   "tokenizer runs here) — characters cannot be added to tokens.",
                zh=f"**{listed} 算不出这个数**："
                   "这些宿主程序下没有任何组件测出「每次请求多少 token」。"
                   "这里唯一免费的数字是 skill 列表，而它只有字符数没有 token 数"
                   "（这里不跑分词器），字符不能和 token 相加。")]

    lines += ["", tr(en="## What this report cannot answer",
                     zh="## 这份报告回答不了什么"), ""]
    for item in data["unanswerable"]:
        lines.append(f"- {item}")
    if data["blind_spots"]:
        lines += ["", tr(en="Recording blind spots established by re-testing, not guessed:",
                         zh="已经查实的记录盲区（复验过，不是猜的）："), ""]
        for spot in data["blind_spots"]:
            lines.append(tr(en=f"- **{spot['what']}**: {spot['evidence']}",
                            zh=f"- **{spot['what']}**：{spot['evidence']}"))
    # Stated positively: a gap that does not apply to this environment is one fewer thing
    # the reader has to worry about, and silently dropping it would hide that.
    if data["blind_spots_not_applicable"]:
        lines += ["", tr(en="These known blind spots do not apply to your environment:",
                         zh="下面这些已知盲区在你这个环境里不成立："), ""]
        for spot in data["blind_spots_not_applicable"]:
            lines.append(tr(en=f"- **{spot['what']}**: {spot['why']}",
                            zh=f"- **{spot['what']}**：{spot['why']}"))

    lines += ["", tr(en="## What could be done to get further",
                     zh="## 想往前推一步，可以做什么"), "",
              tr(en=f"What follows is \"which piece of evidence could be closed\", not "
                    f"\"how to change your configuration\" — {_next_steps_preamble(data)}",
                 zh=f"下面是「能补上哪块证据」，不是「该怎么改配置」—— "
                    f"{_next_steps_preamble(data)}"),
              ""]
    if data["next_steps"]:
        for step in data["next_steps"]:
            price = _price(step["estimated_usd"])
            names = tr(en=", ".join(step["components"][:5]),
                       zh="、".join(step["components"][:5]))
            more = len(step["components"]) > 5
            method = _method_labels().get(step["method"], step["method"])
            lines += [tr(en=f"- **close `{_link_labels()[step['link']]}`** ({price}): "
                            f"{method}",
                         zh=f"- **补「{_link_labels()[step['link']]}」**（{price}）：{method}"),
                      tr(en=f"  - covers: {names}{' and more' if more else ''}",
                         zh=f"  - 涉及：{names}{' 等' if more else ''}"),
                      f"  - {step['note']}"]
    else:
        lines.append(tr(
            en="- the available methods are exhausted; the remaining gaps need methods this "
               "tool does not have.",
            zh="- 现有手段已经用尽；剩下的缺口都要靠本工具还没有的方法。"))

    cost = data["profiler_cost"]
    lines += ["", "---", "",
              tr(en=f"This report's own cost: {ELAPSED_PLACEHOLDER} ms, LLM tokens "
                    f"{cost['llm_input_tokens']}/{cost['llm_output_tokens']}, "
                    f"${cost['cost_usd']}.",
                 zh=f"本报告自身开销：{ELAPSED_PLACEHOLDER} ms，"
                    f"LLM token {cost['llm_input_tokens']}/{cost['llm_output_tokens']}，"
                    f"${cost['cost_usd']}。")]
    return "\n".join(lines)


# The four actions a finding may carry, in the order they are shown. The order is *not* an
# invented severity score — this project does not get to manufacture an impact number any
# more than it gets to manufacture a saving. It is "how ready is this to be acted on":
#
#   fix          something in the environment is inconsistent, and nothing about value is
#                claimed. The most actionable thing the tool can produce.
#   investigate  a measured cost with no evidence of use. Actionable, but only the owner
#                knows why it is installed.
#   ask          a measured cost with heavy use and no measurable value. The tool is out of
#                methods and needs something from the owner to go further.
#   keep         proven to change behaviour. Reassurance, not a task.
#
# Within a class, findings are ordered by magnitude, which is measured rather than assigned.
FINDING_ORDER = ("fix", "test", "investigate", "ask", "keep")

def _action_labels():
    return {
        "fix": tr(en="fix", zh="去收拾"),
        "test": tr(en="can be tested", zh="可以去测"),
        "investigate": tr(en="look into", zh="去查一下"),
        "ask": tr(en="needs a standard from you", zh="需要你给一个判定标准"),
        "keep": tr(en="keep", zh="留着"),
    }

# How many findings the first screen shows. A cap is the point of the section: a reader who
# is handed nineteen rows reads none of them. Everything beyond the cap is still in the
# detail sections below, and the section says how many were held back.
FIRST_SCREEN = 5
# A headline is one line. Seven names in it is a list pretending to be a sentence, so the
# rest go to the evidence bullets, which is where a reader looks for them anyway.
HEADLINE_NAMES = 3


def _price(usd: float | None) -> str:
    return tr(en="free", zh="免费") if not usd else tr(en=f"about ${usd}", zh=f"约 ${usd}")


def _step_line(step: dict) -> str:
    """One measurable gap, as the reader needs it: what it closes, what it costs, how."""
    method = _method_labels().get(step["method"], step["method"])
    return tr(
        en=f"close `{_link_labels()[step['link']]}` "
           f"({_price(step.get('estimated_usd'))}): {method} → "
           f"{_name_list(step['components'])}",
        zh=f"补「{_link_labels()[step['link']]}」（{_price(step.get('estimated_usd'))}）："
           f"{method} → {_name_list(step['components'])}")


def _name_list(names: list[str]) -> str:
    shown = names[:HEADLINE_NAMES]
    text = tr(en=", ".join(f"`{n}`" for n in shown),
              zh="、".join(f"`{n}`" for n in shown))
    if len(names) > len(shown):
        text += tr(en=f" and {len(names) - len(shown)} more ({len(names)} total)",
                   zh=f" 等 {len(names)} 个")
    return text


# A server whose calls fail this often is not working, whatever it is worth. The threshold
# is deliberately high: the corpus-wide error rate across all tools is a few percent, and a
# handful of failures inside hundreds of calls is ordinary agent work (a wrong argument, a
# race, a retry). This is a "something is broken" line, not a quality metric.
ERROR_RATE_SUSPICIOUS = 0.20
# Below this many calls a rate is not a rate. Two failures out of three looks like 67% and
# means nothing.
ERROR_RATE_MIN_CALLS = 10


def _reliability(traces: dict) -> list[dict]:
    """Per-MCP-server call and error counts, which the report was not surfacing at all.

    Free: both numbers are already in the collected observations. This was found by an
    outside audit of the report, which asked what else could be said without a new
    measurement — and the answer was "whether the thing even works", a question the report
    had never asked.

    What this is **not** is evidence about value. A server can succeed at doing something
    useless, and a failing server can be indispensable. It answers "is it working", and the
    finding it produces is `fix`, never `keep` or its opposite.

    Attribution note: only calls the harness emits as tool calls are counted. Codex writes
    MCP calls *inside* the programs it runs (`tools.mcp__server__tool(...)`), where there is
    no per-call result record, so a Codex-only server has usage this cannot see. Such a
    server is reported with `calls_observed_here: 0` rather than as reliable.
    """
    observed = traces.get("observed") or {}
    calls_by_tool = observed.get("calls_by_tool") or {}
    errors_by_tool = observed.get("errors_by_tool") or {}

    servers: dict[str, dict] = {}
    for tool, calls in calls_by_tool.items():
        if not tool.startswith("mcp__"):
            continue
        parts = tool.split("__", 2)
        if len(parts) < 3:
            continue
        entry = servers.setdefault(parts[1], {"calls": 0, "errors": 0, "tools": []})
        errors = errors_by_tool.get(tool, 0)
        entry["calls"] += calls
        entry["errors"] += errors
        entry["tools"].append((parts[2], calls, errors))

    out = []
    for name, entry in sorted(servers.items()):
        rate = entry["errors"] / entry["calls"] if entry["calls"] else 0.0
        out.append({
            "name": name,
            "calls": entry["calls"],
            "errors": entry["errors"],
            "error_rate": rate,
            "suspicious": (entry["calls"] >= ERROR_RATE_MIN_CALLS
                           and rate >= ERROR_RATE_SUSPICIOUS),
            "worst_tools": sorted((t for t in entry["tools"] if t[2]),
                                  key=lambda t: -t[2])[:3],
        })
    return sorted(out, key=lambda r: -r["error_rate"])


def _reliability_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = [tr(en="## Are the MCP servers working", zh="## MCP server 工作正常吗"), "",
             tr(en="This section answers \"is it working\", not \"is it worth having\" — a "
                   "component can succeed brilliantly at something not worth doing. Free; "
                   "the data is already in the session records.",
                zh="这一节只回答「它有没有在正常工作」，不回答「它值不值」—— "
                   "一个组件可以把一件不值得做的事做得很成功。免费，数据就在会话记录里。"), "",
             tr(en="| server | calls | failures | failure rate |",
                zh="| server | 调用 | 失败 | 失败率 |"),
             "| --- | --- | --- | --- |"]
    for row in rows:
        mark = " ⚠" if row["suspicious"] else ""
        lines.append(f"| `{row['name']}` | {row['calls']:,} | {row['errors']:,} | "
                     f"{row['error_rate']:.1%}{mark} |")
    lines += ["",
              tr(en="*Only calls the harness issued as tool calls are counted. Codex writes "
                    "MCP calls inside the body of the programs it runs, where there is no "
                    "per-call result record, so a server used only on the Codex side shows "
                    "0 here — that is invisibility, not an absence of calls.*",
                 zh="*只统计宿主程序当成工具调用发出去的那些。Codex 把 MCP 调用写在它运行的"
                    "程序正文里，那里没有逐次的结果记录，所以只在 Codex 侧用的 server "
                    "在这张表里会是 0 次 —— 那是看不到，不是真的一次都没调用。*"), ""]
    return lines


def _findings(data: dict) -> list[dict]:
    """Discrete, ranked, actionable findings — the report's top-level output.

    Before this the report's first screen was coverage statistics, then (one round later)
    five paragraphs of prose. The prose was right about *content* and still wrong about
    *shape*: a paragraph cannot be counted, ranked against a paragraph of a different kind,
    or carried forward to a later run. A finding has to be an object.

    **One finding per class, never per instance.** The first version of this emitted one per
    duplicated skill, and four identically-worded `fix` rows filled the whole first screen
    and pushed out every other kind — the same overload the section exists to prevent, in a
    new costume. Four copies of one problem is one finding with four instances, which is
    also how a person would say it.

    Every finding is built from evidence already computed elsewhere in the same report.
    Nothing is measured here and nothing is inferred here — a number in this section appears
    identically in the detail below.

    Two rules hold the discipline in place, both enforced by tests:

    1. **Actions attach to findings, not to components.** Hanging an action on all nineteen
       components would put `investigate` on seventeen of them, and a column that says the
       same thing seventeen times carries no information.
    2. **No finding may recommend removal**, because that needs a measured *absence* of
       effect and no component is in that state. `fix` is about consistency and says nothing
       about value; `keep` is an abstention.
    """
    out: list[dict] = []

    # -- fix: the environment contradicts itself, and value is not the question ----------
    diverged = data.get("diverged_duplicates") or []
    if diverged:
        worst = diverged[0]
        detail = (tr(en=f"the widest gap is `{worst['name']}`, "
                        f"{count(worst['days_apart'], 'day')} apart",
                     zh=f"最久的 `{worst['name']}` 两份相差 {worst['days_apart']} 天")
                  if worst["days_apart"]
                  else tr(en=f"e.g. `{worst['name']}`", zh=f"比如 `{worst['name']}`"))
        evidence = []
        for item in diverged:
            gap = (tr(en=f"{count(item['days_apart'], 'day')} apart",
                      zh=f"相差 {item['days_apart']} 天") if item["days_apart"]
                   else tr(en="edited the same day, contents differ",
                           zh="同一天改过，内容不同"))
            # Fullwidth in Chinese, ASCII in English: a CJK glyph in an English line is the
            # same half-translated tell as a missed `tr()`, just harder to grep for.
            paths = tr(en=" | ", zh=" ／ ").join(c["path"] for c in sorted(
                item["copies"], key=lambda c: c["modified"], reverse=True))
            evidence.append(tr(
                en=f"`{item['name']}` ({item['harness']} side, {gap}): {paths}",
                zh=f"`{item['name']}`（{item['harness']} 侧，{gap}）：{paths}"))
        out.append({
            "id": "duplicate-skills",
            "action": "fix",
            "magnitude": worst["days_apart"],
            "instances": len(diverged),
            "headline": tr(
                en=f"{count(len(diverged), 'skill is', 'skills are')} installed twice "
                   f"with differing content ({detail})",
                zh=f"{len(diverged)} 个 skill 装了两份、内容还不一样（{detail}）"),
            "detail": tr(
                en="the harness lists each name once, so it loads only one of the copies; "
                   "the copies differ; and which one was loaded cannot be recovered (a call "
                   "records the name, not the path). Together: you do not know which "
                   "version is in effect.",
                zh="宿主程序对每个名字只列一次，也就是只加载其中一份；两份内容不同；"
                   "而哪一份被加载查不出来（调用记录只有名字，没有路径）。"
                   "合起来就是：你不确定自己生效的是哪个版本。"),
            "evidence": evidence,
            "next": tr(
                en="nothing further needs measuring. The paths are above; which copy to "
                   "keep is your call — this tool does not modify your real configuration.",
                zh="这条不需要再测什么。路径都在上面，留哪份是你的决定 —— "
                   "本工具不改你的真实配置。"),
        })

    # -- fix: a server that is declared and failing is broken regardless of its value ----
    broken = [r for r in data.get("reliability") or [] if r["suspicious"]]
    if broken:
        worst_rate = max(r["error_rate"] for r in broken)
        out.append({
            "id": "failing-mcp",
            "action": "fix",
            "magnitude": worst_rate * 1000,
            "instances": len(broken),
            "headline": _name_list([r["name"] for r in broken]) + tr(
                en=f" has an elevated call failure rate (worst {worst_rate:.0%})",
                zh=f" 的调用失败率偏高（最高 {worst_rate:.0%}）"),
            "detail": tr(
                en="a high failure rate means it is not working, which is a separate "
                   "question from whether it is worth having — a broken component should be "
                   "fixed whatever its value.",
                zh="失败率高说明它没在正常工作，这跟「值不值得装」是两件事 —— "
                   "一个坏掉的组件不论价值如何都该修。"),
            "evidence": [tr(en=f"`{r['name']}`: {count(r['calls'], 'call')}, "
                               f"{count(r['errors'], 'failure')} "
                               f"({r['error_rate']:.1%})",
                            zh=f"`{r['name']}`：{r['calls']:,} 次调用，{r['errors']:,} 次失败"
                               f"（{r['error_rate']:.1%}）") for r in broken],
            "next": tr(
                en="this tool does not read call arguments (they may carry credentials), so "
                   "the cause is something you need to look at in those calls yourself.",
                zh="本工具不读取调用参数（可能含凭据），所以具体原因要你自己看那几次调用。"),
        })

    # -- test: the tool can still close this gap itself, for a known price ---------------
    # This was already computed and buried in a closing section, which made the report's
    # most useful sentence — "here is what I could measure next, and what it costs" — the
    # last thing a reader reached. It is a finding: actionable now, by the tool, at a price.
    steps = data.get("next_steps") or []
    priced = [s for s in steps if s.get("estimated_usd")]
    free = [s for s in steps if not s.get("estimated_usd")]
    if steps:
        cheapest = min(steps, key=lambda s: s.get("estimated_usd") or 0.0)
        covered = sorted({name for s in steps for name in s["components"]})
        price = _price(cheapest.get("estimated_usd"))
        out.append({
            "id": "measurable-gaps",
            "action": "test",
            # Ranked by how many components it would unblock, which is counted, not assigned.
            "magnitude": len(covered),
            "instances": len(steps),
            "headline": tr(
                en=f"{count(len(covered), 'component has', 'components have')} evidence "
                   f"gaps that can still be closed ({len(free)} free, "
                   f"{len(priced)} paid, cheapest {price})",
                zh=f"还有 {len(covered)} 个组件的证据缺口是能补上的"
                   f"（{len(free)} 项免费，{len(priced)} 项要花钱，最便宜的{price}）"),
            "detail": tr(
                en="these are not \"unknowable\" but \"not yet measured\" — the difference "
                   "is that a method exists. Sorted by price below; the free ones are "
                   "simply not collected or not run yet.",
                zh="这些不是「不知道」，是「还没测」—— 区别在于有现成的方法。"
                   "下面按价格排序，免费的那些只是还没采集或还没跑。"),
            "evidence": [_step_line(step) for step in steps],
            "next": tr(
                en="say which one and it gets run; the paid ones are quoted before "
                   "anything is spent.",
                zh="要哪一项我就去跑；花钱的那几项我会先报价再动手。"),
        })

    # -- investigate: a measured cost with nothing showing it was asked for --------------
    uncalled = [r for r in data.get("always_on") or [] if r["calls"] == 0]
    if uncalled:
        evidence = []
        for row in uncalled:
            also = tr(en="; ".join(f"the {a['harness']} copy was used {a['calls']} times"
                                   for a in row.get("also") or []),
                      zh="；".join(f"{a['harness']} 那份被用过 {a['calls']} 次"
                                   for a in row.get("also") or []))
            evidence.append(tr(
                en=f"`{row['name']}`: {row['tokens']:,} tokens/request, 0 explicit "
                   f"calls across {count(data['sessions'], 'session')}"
                   + (f" (but {also})" if also else ""),
                zh=f"`{row['name']}`：{row['tokens']:,} token/次，"
                   f"{data['sessions']} 个会话里显式调用 0 次"
                   + (f"（但 {also}）" if also else "")))
        out.append({
            "id": "uncalled-but-paid",
            "action": "investigate",
            "magnitude": max(r["tokens"] for r in uncalled),
            "instances": len(uncalled),
            "headline": _name_list([r["name"] for r in uncalled]) + tr(
                en=" costs tokens on every request, and you have never called it by name",
                zh=" 每次请求都在付钱，但你从没点过它的名"),
            "detail": tr(
                en="this does not mean it did nothing: loading by description match leaves "
                   "no record at all, so what is proven here is only \"no explicit call\".",
                zh="这不代表它没起作用：靠描述匹配自动载入不留任何记录，"
                   "所以这里只能证明「没有显式调用」。"),
            "evidence": evidence,
            "next": tr(
                en="you know why you installed it better than this report does. If it fires "
                   "on a description match, zero calls is normal; if you cannot remember "
                   "what it does, that is the signal.",
                zh="你比这份报告更清楚当初为什么装它。如果它是靠描述自动触发的，"
                   "零调用是正常的；如果你想不起它干什么，那才是信号。"),
        })

    # -- ask: out of methods, and honest about needing the owner -------------------------
    stuck = []
    for component in components_of(data):
        links = component["links"]
        cost, used, useful = links["cost"], links["used"], links["useful"]
        if useful["status"] != UNPROVEN or cost.get("unit") != "tokens_per_request":
            continue
        if cost.get("value") is None or not used.get("value"):
            continue
        if (component.get("next_measurement") or {}).get("achievable"):
            # A real priced measurement exists, so this belongs in the next-steps section
            # rather than being turned into a question for the owner.
            continue
        stuck.append((component, used, cost))
    # One row per *listed* name. Two installed copies of a skill are two components and one
    # thing the harness loads, and listing both put `second-skill` in this finding twice. That
    # is the aggregation trap for the seventh time, in a seventh unit — so the same rule the
    # cost floor uses applies here: collapse by name, keep the lower figure.
    by_name: dict[tuple, tuple] = {}
    for component, used, cost in stuck:
        key = (component["kind"], component["name"])
        kept = by_name.get(key)
        if kept is None or cost["value"] < kept[2]["value"]:
            by_name[key] = (component, used, cost)
    stuck = list(by_name.values())
    if stuck:
        stuck.sort(key=lambda t: -t[1]["value"])
        out.append({
            "id": "value-unmeasurable",
            "action": "ask",
            "magnitude": max(c["value"] for _, _, c in stuck),
            "instances": len(stuck),
            "headline": _name_list([c["name"] for c, _, _ in stuck]) + tr(
                en=": used heavily, cost known, worth unmeasurable",
                zh="：用得很多、成本已知，但值不值测不出来"),
            "detail": tr(
                en="what is missing is not the experiment — the arms exist and are "
                   "validated. What is missing is **a task whose success a machine can "
                   "judge**. Instruction files are measurable because they state their own "
                   "rules; these have no such standard.",
                zh="缺的不是实验手段 —— 实验臂已经有并且验证过了。"
                   "缺的是**一个成败能用机器判定的任务**。"
                   "规范文件能测，是因为它自己写明了规则；这些没有这种判定标准。"),
            "evidence": [tr(en=f"`{c['name']}`: {count(u['value'], 'call')}, "
                               f"{k['value']:,} tokens/request (measured)",
                            zh=f"`{c['name']}`：{u['value']} 次调用，"
                               f"{k['value']:,} token/次（实测）") for c, u, k in stuck],
            "next": tr(
                en="name a task whose outcome you can judge at a glance (\"did it pull the "
                   "right price off this page?\") and the arms can be built for it. "
                   "Inventing a proxy metric is the worse option — this project tried twice "
                   "and both times it was self-deception.",
                zh="说出一个任务，你看一眼就知道它成没成（比如「它有没有从这页里取对价格」），"
                   "我就能给它搭实验臂。编一个代理指标是更糟的选择 —— "
                   "这个项目试过两次，两次都是自欺。"),
        })

    # -- keep: proven to change behaviour ------------------------------------------------
    verdicts = data.get("verdicts") or []
    if verdicts:
        unstable = [v["name"] for v in verdicts if not v["stable_across_profiles"]]
        out.append({
            "id": "proven-useful",
            "action": "keep",
            "magnitude": max(v["cost_value"] or 0 for v in verdicts),
            "instances": len(verdicts),
            "headline": _name_list([v["name"] for v in verdicts]) + tr(
                en=" does change what the agent does", zh=" 确实在起作用"),
            "detail": tr(
                en="without these files the agent does not follow the conventions you "
                   "wrote; with them it does. That is a measured behavioural difference, "
                   "not \"looks useful\".",
                zh="不给这些文件，agent 就不按你写的规范做；给了就按。"
                   "这是实测的行为差异，不是「看起来有用」。")
                + (tr(en=f" Note that {', '.join(unstable)} flips under a different set of "
                         "weights.",
                      zh=f"注意 {'、'.join(unstable)} 换一组权重结论就变。")
                   if unstable else ""),
            "evidence": [tr(en=f"`{v['name']}`: {v['cost_value']:,} tokens/request — "
                               f"{v['basis']}",
                            zh=f"`{v['name']}`：{v['cost_value']:,} token/次 —— "
                               f"{v['basis']}") for v in verdicts],
            "next": verdicts[0]["limit"],
        })

    rank = {action: i for i, action in enumerate(FINDING_ORDER)}
    return sorted(out, key=lambda f: (rank[f["action"]], -(f["magnitude"] or 0)))


def _finding_lines(findings: list[dict], data: dict) -> list[str]:
    """The first screen: a ranked list, capped, with the coverage gap kept in one line."""
    total = len(components_of(data))
    scorable = sum(1 for c in components_of(data) if c["coverage"]["scorable"])
    lines = [tr(en="## What was found", zh="## 发现了什么"), ""]
    if not findings:
        lines += [tr(en="nothing to report this round — no inconsistency was found and no "
                        "component's usefulness was proven. That does not mean the "
                        "environment is fine; see the coverage below.",
                     zh="这轮没有可报的发现 —— 既没有查出不一致，也没有任何组件的「有用」"
                        "得到证明。这不等于环境没问题，见下面的覆盖情况。"), ""]
        return lines

    shown = findings[:FIRST_SCREEN]
    held = len(findings) - len(shown)
    lines += [tr(en=f"Ranked by \"can be acted on now\", top {len(shown)}"
                    + (f" ({held} more of the same kinds in the sections below)"
                       if held else "") + ":",
                 zh=f"按「现在就能动手」排序，前 {len(shown)} 条"
                    + (f"（另有 {held} 条同类的在下面小节里）" if held else "") + "："), ""]
    for index, finding in enumerate(shown, 1):
        lines += [f"**{index}. [{_action_labels()[finding['action']]}] {finding['headline']}**",
                  "", f"   {finding['detail']}", ""]
        for item in finding["evidence"]:
            lines.append(f"   - {item}")
        lines += ["", f"   → {finding['next']}", ""]

    # Demoted from the headline, but never removed: a reader who finishes five findings and
    # concludes "so my environment is fine" has made a worse error than one who drowned in
    # tables. Conclusion strength may not exceed evidence strength, and that is the whole
    # reason this tool exists.
    lines += [tr(
        en=f"Beyond these, for {total - scorable} of {total} components this report **cannot "
           "say whether they are worth their keep** — not that they are fine, but that it "
           "was not measured. What is missing is not the experiment but a task whose success "
           "a machine can judge. See the coverage below.",
        zh=f"除此之外，{total} 个组件里有 {total - scorable} 个这份报告**给不出"
           "「值不值」的判断** —— 不是说它们没问题，是没测出来。"
           "缺的不是实验手段，是一个成败能用机器判定的任务。详情见下面的覆盖情况。"), ""]
    return lines


def _overhead_line(data: dict) -> list[str]:
    """The one number from the old prose first screen that is not a finding.

    "Every request carries at least N tokens" is not something to act on — it is the
    baseline the findings sit against. It leads, and then the findings follow.
    """
    floors = [(h, e) for h, e in sorted((data["stack_cost"] or {}).items())
              if e.get("per_request_tokens_floor")]
    if not floors:
        return []
    parts = []
    for harness, entry in floors:
        low = entry["per_request_tokens_floor"]
        high = entry.get("per_request_tokens_floor_high")
        span = f"{low:,}–{high:,}" if high and high != low else f"{low:,}"
        parts.append(tr(en=f"{harness} **at least {span} tokens**",
                        zh=f"{harness} **至少 {span} token**"))
    return [tr(en="**What every request carries before you say anything**: "
                  f"{'; '.join(parts)}. This is measured fixed-prefix blocks added "
                  "together, and it is a floor — whatever was not measured can only push "
                  "it up.",
               zh=f"**每说一句话，先背多少行李**：{'；'.join(parts)}。"
                  "这是实测的几块固定前缀相加，是下限 —— 没测到的只会让它更高。"), ""]


def _carried_lines(components: list[dict]) -> list[dict]:
    """What the always-on components actually moved, per harness.

    Deliberately not one grand total. Two installed copies of one skill under different
    scopes are two components with the same measured rate, and adding their carried
    figures would count the same prefix twice. The rows are per component, so the reader
    can see the duplicates.
    """
    out = []
    for component in components:
        cost = component["links"]["cost"]
        carried = cost.get("tokens_carried")
        if carried is None:
            continue
        out.append({
            "name": component["name"],
            "scope": component["scope"],
            # Where the component is installed, which is its identity. The harness the
            # rate was *measured* under is a different fact and is kept separately: one
            # skill installed under two scopes was measured under claude_code both times,
            # so labelling the rows by the measured harness made them indistinguishable.
            "harness": component["harness"],
            "measured_under": cost.get("harness") or component["harness"],
            "path": component.get("path"),
            "tokens_per_request": cost.get("value"),
            "requests": cost.get("requests_observed"),
            "tokens_carried": carried,
            "lower_bound": bool(cost.get("carried_is_lower_bound")),
            "excludes": cost.get("carried_excludes") or [],
        })
    # A name that repeats gets its install directory, because two copies of one skill with
    # two different measured rates are otherwise the same row printed twice.
    seen = Counter(r["name"] for r in out)
    home = str(Path.home())
    for r in out:
        if seen[r["name"]] > 1 and r["path"]:
            parent = str(Path(r["path"]).parent.parent)
            r["where"] = parent[len(home) + 1:] if parent.startswith(home) else parent
    return sorted(out, key=lambda r: -r["tokens_carried"])


def _render_carried(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = [tr(en="## What the always-on components carried", zh="## 常驻组件累计搬了多少"), ""]
    lines += [tr(en="| component | measured rate | requests | carried |",
                 zh="| 组件 | 实测速率 | 请求数 | 累计 |"),
              "| --- | --- | --- | --- |"]
    for r in rows:
        mark = "≥" if r["lower_bound"] else ""
        where = r.get("where") or f"{r['scope']}/{r['harness']}"
        label = f"`{r['name']}` ({where})"
        lines.append(f"| {label} | "
                     f"{r['tokens_per_request']:,} | {r['requests']:,} | "
                     f"{mark}{r['tokens_carried']:,} |")
    excluded = sorted({h for r in rows for h in r["excludes"]})
    lines += ["", tr(
        en="Rate times requests, both observed. The request count is this corpus's model "
           "responses with this tool's own probe sessions removed, cross-checked by two "
           "independent walkers. Rows are not summed: one skill installed under two scopes "
           "appears twice with the same prefix, and adding those would count it twice."
           + (f" A `≥` row is carried by {', '.join(excluded)} traffic as well, at a rate "
              "nobody has measured." if excluded else ""),
        zh="速率乘请求数，两边都是实测。请求数是本语料的模型响应数，已剔除本工具自己的"
           "探针会话，并由两个独立遍历器互校。各行**不可相加**：同一个 skill 装在两个"
           "位置就出现两次，加起来等于把同一段前缀算两遍。"
           + (f" 带 `≥` 的行还会被 {'、'.join(excluded)} 的流量搬运，那边的速率没人测过。"
              if excluded else "")), ""]
    lines += [tr(
        en="What these tokens cost in money is not stated: that needs a price per token, "
           "which depends on the model and the cache state of each request, and this tool "
           "does not invent rates.",
        zh="这些 token 折成钱是多少，这里不说：那需要单价，而单价取决于模型和每次请求的"
           "缓存状态，本工具不编造费率。"), ""]
    return lines


def components_of(data: dict) -> list[dict]:
    """The component rows, wherever the report's shape puts them."""
    return data.get("components") or []


def _headline(components: list[dict], link_coverage: dict, stack: dict) -> str:
    """One honest sentence, which on a real stack is mostly about what is missing."""
    proven_links = [_link_labels()[k] for k in SCORED_LINKS if link_coverage[k]["proven"]]
    # A link proved nowhere but measured somewhere is not the same as untouched, and the
    # headline once said 「成本还没有任何组件证明得了」 in the same breath as 「齐备的组件有
    # 1 个」 because it only looked at `proven`.
    missing = [_link_labels()[k] for k in SCORED_LINKS
               if not link_coverage[k]["proven"] and not link_coverage[k]["partial"]]
    partial_only = [_link_labels()[k] for k in SCORED_LINKS
                    if not link_coverage[k]["proven"] and link_coverage[k]["partial"]]
    joiner = tr(en=", ", zh="、")
    if stack["withheld"]:
        scorable = sum(1 for c in components if c["coverage"]["scorable"])
        state = (tr(en=f"**{joiner.join(missing)}** is not yet proven for any component. ",
                    zh=f"**{'、'.join(missing)}** 还没有任何组件证明得了。")
                 if missing else "")
        if partial_only:
            state += tr(en=f"**{joiner.join(partial_only)}** is only partially proven. ",
                        zh=f"**{'、'.join(partial_only)}** 只测到部分证明。")
        if scorable:
            return tr(
                en=f"Of {len(components)} components, **{scorable}** have all three of "
                   f"used / useful / cost and can be given a verdict; the other "
                   f"{len(components) - scorable} cannot. {state}"
                   f"The stack score is still withheld: only "
                   f"{scorable / len(components):.0%} are scorable, and taking that "
                   f"coverage to stand for the whole environment is inferring the many "
                   f"from the few.",
                zh=f"{len(components)} 个组件里，有 **{scorable} 个**「用过、有用、成本」"
                   f"三环齐备，能给出结论；其余 {len(components) - scorable} 个不能。{state}"
                   f"整体总分仍然扣住：可打分的只占 {scorable / len(components):.0%}，"
                   f"拿这点覆盖率去代表整套环境，等于用少数推断全体。")
        return tr(
            en=f"Of {len(components)} components, the links that stand up are "
               f"**{joiner.join(proven_links) or 'none'}**; {state}"
               f"so the score is withheld: scoring needs used / useful / cost together, "
               f"and 0 components have all three. Ranking on whichever links happen to be "
               f"measured would be ranking by what was measurable.",
            zh=f"{len(components)} 个组件里，证据链上已经站得住的环节是"
               f"**{'、'.join(proven_links) or '无'}**；{state}"
               f"所以总分被扣住：打分要求「用过、有用、成本」三环齐备，"
               f"目前齐备的组件有 0 个。拿已测到的那几环去排名，"
               f"等于按「哪项恰好可测」排名。")
    return tr(
        en=f"{len(components)} components, score {stack['value']} ({stack['basis']}, "
           f"strength estimated; the weights are this tool's assumption, not preferences "
           f"you supplied).",
        zh=f"{len(components)} 个组件，总分 {stack['value']}"
           f"（{stack['basis']}，强度 estimated，权重是本工具的假设而非你给的偏好）。")


def report_root(root: str, *, evaluation_path: str | None = None,
                limit: int | None = None, profile: str = "balanced",
                whole_environment: bool = False, exclusions: dict | None = None,
                useful_path: str | None = None, use_cache: bool = True) -> dict:
    """Assemble every stage's evidence into a readable report.

    With `whole_environment` the subject is the environment rather than one directory,
    which is the question this tool exists to answer.
    """
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    started = time.perf_counter()
    scan, traces, evaluation, useful, shared, codex_usage = _collect(
        str(base), limit, evaluation_path, whole_environment=whole_environment,
        exclusions=exclusions, useful_path=useful_path, use_cache=use_cache)
    components = build_chain(scan=scan, traces=traces, evaluation=evaluation,
                             useful=useful, codex_usage=codex_usage,
                             codex_listing=shared)
    orphans = unmatched_measurements(scan, evaluation, useful)
    for component in components:
        component["coverage"] = _coverage(component)
        # Decided before the suggested step is rendered, so the report can separate a link
        # nobody has measured yet from one no method here can measure.
        component["useful_measurability"] = useful_measurability(component, components)
        component["next_measurement"] = _next_measurement(component)
        # Stability across profiles is part of the verdict, not a footnote: a conclusion
        # that only survives one set of invented weights is a property of the weights.
        if component["coverage"]["scorable"]:
            component["sensitivity"] = sensitivity(component)

    link_coverage = {
        link: {
            "proven": sum(1 for c in components if c["links"][link]["status"] == PROVEN),
            "partial": sum(1 for c in components if c["links"][link]["status"] == PARTIAL),
            "unproven": sum(1 for c in components if c["links"][link]["status"] == UNPROVEN),
        }
        for link in SCORED_LINKS
    }
    scorable = [c for c in components if c["coverage"]["scorable"]]
    stack = _stack_score(scorable, components, profile)
    scan_observed = scan["observed"]

    data = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "report",
        "root": str(base),
        "read_only": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "components": components,
        # A supplied measurement that joined to nothing reads downstream exactly like one
        # nobody ran, so it is surfaced rather than silently weakening a link.
        "unmatched_measurements": orphans,
        "relocation_candidates": traces["observed"].get("relocation_candidates") or [],
        "observed": {
            "installed": {
                "files": scan_observed["files_found"],
                "bytes": scan_observed["bytes_found"],
                "by_kind": scan_observed["by_kind"],
                # by_kind counts files, and a skill is a directory of them, so a handful of
                # skills shows up as a file count many times larger — which a reader takes
                # for the number of skills. The rollup is the count the component table
                # below is actually built from.
                "entities": {k: v["entities"] for k, v in scan_observed["components"].items()},
                "entity_files": {k: v["files"] for k, v in scan_observed["components"].items()},
                "bundled_files": scan_observed["by_origin"].get("bundled", 0),
            },
            "sessions": traces["observed"]["sessions_collected"],
            "link_coverage": link_coverage,
            "components_total": len(components),
            "components_scorable": len(scorable),
        },
        "stack_score": stack,
        "sessions": traces["observed"]["sessions_collected"],
        "always_on": _always_on_but_uncalled(components),
        "diverged_duplicates": _diverged_duplicates(components),
        "reliability": _reliability(traces),
        "verdicts": _verdicts(components, profile),
        "profile": profile,
        "cost": _cost_lines(traces, evaluation, shared),
        # Per component, never summed; see `_carried_lines`.
        "carried": _carried_lines(components),
        "stack_cost": stack_cost(components, shared),
        "unanswerable": _unanswerable(components, link_coverage),
        "blind_spots": [dict(spot) for spot in _confirmed_blind_spots()
                        if spot["requires"] in _present_classes(scan, components)],
        "blind_spots_not_applicable": [
            {**spot, "why": tr(
                en=f"this environment has no {spot['requires']}, so the blind spot does "
                   f"not arise",
                zh=f"这个环境里没有 {spot['requires']}，所以不存在这个盲区")}
            for spot in _confirmed_blind_spots()
            if spot["requires"] not in _present_classes(scan, components)],
        "next_steps": _next_steps(components),
        "profiler_cost": {
            "stage": "report",
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "cost_usd": 0.0,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    }
    data["headline"] = _headline(components, link_coverage, stack)
    # Findings are derived from the rest of `data`, so they are computed after it exists
    # rather than being threaded through every helper that feeds them.
    data["findings"] = _findings(data)
    # Rendering is part of what this costs, and on a large stack it was the most expensive
    # part of all — so the figure is stamped after it, not before. A total cannot be printed
    # before the work it totals is done, so `_render` writes a placeholder and the real
    # number is substituted here; only that substitution is left out of its own figure.
    data["text"] = _render(data)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    data["profiler_cost"]["elapsed_ms"] = elapsed
    data["text"] = data["text"].replace(ELAPSED_PLACEHOLDER, f"{elapsed}")
    return data

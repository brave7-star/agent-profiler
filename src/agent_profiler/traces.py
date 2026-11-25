"""Claude Code trace collector.

Stage 2 collects what actually happened in a run. It is deterministic and costs 0 LLM
tokens: everything reported here is read directly out of the harness transcript.

Three properties of the real Claude Code format drive this design, each verified against
live transcripts (see docs/dogfood/trace-schema.md):

1. One API response is written as **one record per content block**, all sharing a
   `message.id` and repeating the same `message.usage`. Summing usage per record
   overcounts tokens; usage is therefore counted once per message id.
2. `api-request-blob` records **replay the request payload**, including the assistant's
   `tool_use` blocks. Counting tool calls from every record double-counts them, so only
   `assistant` records are counted, deduplicated by `tool_use` id.
3. `assistant.requestRef` points at `api-request.id` (not `requestId`), which is what
   makes per-request latency derivable.

Transcripts reach ~75 MB, so parsing streams line by line and never holds a whole file.
"""

from __future__ import annotations

import json
import re
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .core import FileCache, _encode_project_key, _user_tree

# Records that carry token usage. Verified: only `assistant` does.
USAGE_RECORD_TYPE = "assistant"
# Request payload replays. Legitimate raw evidence, never counted as activity.
REPLAY_RECORD_TYPES = {"api-request-blob", "api-request-shape"}
# Types seen in real transcripts. Anything else is reported as schema drift.
KNOWN_RECORD_TYPES = {
    "assistant",
    "user",
    "api-request",
    "api-request-blob",
    "api-request-shape",
    "attachment",
    "atis-latch",
    "last-prompt",
    "ai-title",
    "queue-operation",
    "file-history-snapshot",
    "file-history-delta",
    "cost-state",
    "system",
    "mode",
    "tools-used",
    "summary",
    "compact-boundary",
    "permission-mode",
}
# A model id used for messages the harness synthesises locally (no API call billed).
SYNTHETIC_MODEL = "<synthetic>"
# Caps on retained raw references, so a huge transcript cannot blow up the output.
MAX_REFS = 25
MAX_PARSE_ERRORS = 25
# Above this, counting more than the harness means de-duplication is broken. Real sessions
# show up to ~1% positive variance from accounting differences alone, so the alarm sits
# above that without being wide enough to hide an actual repeated record.
OVER_COUNT_TOLERANCE_PCT = 2.0


# A `Skill` call names the skill it invokes, which is the only record tying a declared
# entity to its use. The name comes from a fixed published list, so unlike a general tool
# argument it cannot carry a credential and is kept rather than digested.
COMPONENT_KEYS = {"Skill": "skill", "SlashCommand": "command"}

# Tools that run a shell command. Codex has no skill-invocation tool at all: it uses a
# skill by *reading the file*, which arrives as a generic shell call whose argument holds
# the path. Looking only at tool names therefore found nothing and the project recorded
# "Codex skill invocation is unobservable" — a false negative in this detector, not a gap
# in the harness. Confirmed by running one: the rollout carries
# `function_call name=exec_command {"cmd":"cat .../skills/build-status/SKILL.md"}`.
# `exec` is Codex's dominant shell tool — the overwhelming majority of its tool calls —
# and leaving it out made the detector blind to almost everything Codex does.
SHELL_TOOLS = ("exec", "exec_command", "shell", "local_shell", "bash", "Bash")
_SHELL_KEYS = ("cmd", "command", "script")
# Only the skill directory layout is matched, and only the name is extracted — never the
# surrounding command, which can hold a credential.
SKILL_BODY_READ = re.compile(r"skills/(?:\.system/)?([A-Za-z0-9._-]+)/SKILL\.md")
# The same match with whatever path led up to it. The name alone cannot tell a user's
# skill from a vendor-bundled one sharing its name, so attribution needs the full path;
# the leading run deliberately stops at quotes and whitespace so no command text is kept.
SKILL_BODY_PATH = re.compile(r"[^\s\"'`]*skills/(?:\.system/)?[A-Za-z0-9._-]+/SKILL\.md")
# A literal the pattern above cannot match without, checked before the pattern runs.
#
# That leading `[^\s"'`]*` is unanchored and greedy, and the text it scans is a whole Codex
# program — one `exec` call can carry a JavaScript file. So the engine retries the run from
# every position in a long line of non-whitespace, and this single pattern was **5.8 of 28
# seconds** of a whole-environment `normalize`, almost all of it on text that could never
# match. `in` on a literal is a memchr; the regex is not.
SKILL_BODY_MARKER = "/SKILL.md"


def _component_of(tool: str | None, tool_input: object) -> str | None:
    """The declared entity a call invokes, when the call names one.

    Only `Skill`-class tools do. This is the one place a declared component can be tied to
    an observed call, so it is extracted rather than left inside the input digest.
    """
    key = COMPONENT_KEYS.get(tool or "")
    if key is None or not isinstance(tool_input, dict):
        return None
    value = tool_input.get(key)
    return value if isinstance(value, str) and value else None


def _shell_command(tool_input: object) -> str | None:
    """The command text of a shell call, whatever key this harness puts it under.

    Codex's `exec` does not carry a command at all: its input is a JavaScript program that
    calls `tools.exec_command({cmd: "..."})`. The whole string is returned rather than the
    `cmd` argument parsed out of it, because callers here search it for a pattern and a
    half-parsed program is worse than the program. Nothing is executed, only matched.
    """
    if isinstance(tool_input, str):
        return tool_input
    if not isinstance(tool_input, dict):
        return None
    for key in _SHELL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return " ".join(value)
    return None


def _skill_body_read(tool: str | None, tool_input: object) -> str | None:
    """The skill whose body a shell call read, which is how Codex uses a skill.

    This is weaker evidence than a tool call naming a skill, and it is deliberately kept
    separate rather than merged into `component`: these same reads also happen while
    *authoring* a skill, and the project already has a documented case of 544 apparent hits
    that were development activity. The caller records how the component was identified so
    a reader can tell hard evidence from a heuristic.
    """
    if (tool or "") not in SHELL_TOOLS:
        return None
    command = _shell_command(tool_input)
    if not command:
        return None
    match = SKILL_BODY_READ.search(command)
    return match.group(1) if match else None


# Writing to a skill's own file is authoring it, not using it. Codex has no single write
# tool — `apply_patch` is one, but most edits arrive as shell redirection inside `exec` —
# so the command text is matched as well. This runs where the input is already parsed, and
# only the boolean leaves: the command itself never enters an event, because it can hold a
# credential.
_WRITE_TOOLS = ("apply_patch", "write_file", "create_file", "Write", "Edit")
_WRITE_COMMAND = re.compile(r"apply_patch|write_file|create_file|\bcat\s*>|\btee\b|>>|\bsed -i\b")


# Codex has no MCP tool call of its own: the model writes `tools.mcp__<server>__<tool>(...)`
# inside the JavaScript program it hands to `exec`. Requiring the `tools.` prefix and the
# open paren is what separates a call from a mention — the model also writes things like
# `ALL_TOOLS.filter(x => /repl_server/.test(x.name))`, which names a server without using it.
MCP_CALL_IN_PROGRAM = re.compile(r"\btools\.mcp__([A-Za-z0-9_-]+)__[A-Za-z0-9_]+\s*\(")
# Same short-circuit as SKILL_BODY_MARKER: a literal the pattern cannot match without.
MCP_CALL_MARKER = "tools.mcp__"


def _mcp_servers_called(tool: str | None, tool_input: object) -> tuple[str, ...]:
    """MCP servers invoked from inside a program this call executed.

    Names are normalized to the underscore form the tool namespace uses, because a server
    declared as `demo-server` is called as `mcp__demo_server__*`.
    """
    text = _path_bearing_text(tool, tool_input)
    if not text or MCP_CALL_MARKER not in text:
        return ()
    return tuple(sorted({m.replace("-", "_") for m in MCP_CALL_IN_PROGRAM.findall(text)}))


def _skill_body_written(tool: str | None, tool_input: object) -> bool:
    """Whether the call that named a skill's body was editing it rather than reading it."""
    if (tool or "") in _WRITE_TOOLS:
        return True
    command = _shell_command(tool_input) if (tool or "") in SHELL_TOOLS else None
    return bool(command) and bool(_WRITE_COMMAND.search(command))


def _path_bearing_text(tool: str | None, tool_input: object) -> str | None:
    """The part of a call that can name a file, for the tools that name one.

    A write tool puts the path in its arguments rather than in a command, so reading only
    `_shell_command` found nothing for `apply_patch` — and a write that cannot be located
    is a write that cannot be told apart from a read.
    """
    if (tool or "") in SHELL_TOOLS:
        return _shell_command(tool_input)
    if (tool or "") not in _WRITE_TOOLS:
        return None
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, dict):
        return " ".join(v for v in tool_input.values() if isinstance(v, str))
    return None


def _skill_body_paths(tool: str | None, tool_input: object) -> tuple[str, ...]:
    """Every installed-skill path a call named, normalized for comparison.

    One command often names several, and the name alone cannot say which skill it was, so
    attribution upstream is by path. Relative paths are dropped rather than resolved
    against a guessed cwd: a wrong guess would credit the use to the wrong component.
    """
    command = _path_bearing_text(tool, tool_input)
    # The marker check is a short-circuit, not a filter: the pattern contains this literal,
    # so text without it has no match to find. See SKILL_BODY_MARKER for why it is worth it.
    if not command or SKILL_BODY_MARKER not in command:
        return ()
    found = []
    for match in SKILL_BODY_PATH.finditer(command):
        text = match.group(0)
        if not text.startswith("/"):
            continue
        resolved = str(Path(text))
        if resolved not in found:
            found.append(resolved)
    return tuple(found)


def _context_loads(obj: dict) -> Iterator[dict]:
    """What an `attachment` record says entered the context window.

    This is the `loaded` link, and it is recorded by the harness after all — the earlier
    claim that nothing observes it came from never reading this record type.

    `chars` is the size of the text that occupied context. It is reported as characters
    rather than tokens because no tokenizer is run here; `shared` says whether that size
    belongs to one component or is a joint cost across several, which matters because a
    skill listing names many skills in one block and its size cannot be split between them.
    """
    attachment = obj.get("attachment")
    if not isinstance(attachment, dict):
        return
    kind = attachment.get("type")

    if kind == "skill_listing":
        names = [n for n in attachment.get("names") or [] if isinstance(n, str)]
        if names:
            yield {
                "load_kind": "skill_listing",
                "components": names,
                "component_kind": "skills",
                "chars": len(attachment.get("content") or ""),
                "shared": True,
                "initial": bool(attachment.get("isInitial")),
                # Verified on real listings: a few thousand characters for a dozen-odd skills, formatted
                # `- name: description`. Being listed means the skill was *offered* as a
                # one-line description, not that its body entered context.
                "depth": "description_only",
            }

    elif kind == "agent_listing_delta":
        built_in = {n for n in (attachment.get("builtInTypes") or [])
                    if isinstance(n, str)}
        built_in.update(n for n in (attachment.get("builtInPluginTypes") or [])
                        if isinstance(n, str))
        # Harness-bundled agents are not the user's stack. Counting them would repeat the
        # Stage 1 false positive where bundled entities inflated the inventory.
        names = [n for n in attachment.get("addedTypes") or []
                 if isinstance(n, str) and n not in built_in]
        if names:
            yield {
                "load_kind": "agent_listing",
                "components": names,
                "component_kind": "agents",
                "chars": len(attachment.get("addedLines") or ""),
                "shared": True,
                "initial": bool(attachment.get("isInitial")),
                "depth": "description_only",
            }

    elif kind == "mcp_instructions_delta":
        names = [n for n in attachment.get("addedNames") or [] if isinstance(n, str)]
        if names:
            yield {
                "load_kind": "mcp_instructions",
                "components": names,
                "component_kind": "mcp",
                "chars": sum(len(b or "") for b in attachment.get("addedBlocks") or []),
                "shared": len(names) > 1,
                "initial": None,
            }

    elif kind == "instructions":
        for entry in attachment.get("files") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            yield {
                "load_kind": "instructions",
                "components": [entry["path"]],
                "component_kind": "instructions",
                "instruction_type": entry.get("type"),
                # One file, one size: this cost is attributable, unlike a shared listing.
                "chars": len(entry.get("content") or ""),
                "shared": False,
                "initial": None,
            }


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Transcripts use ...Z; fromisoformat needs an explicit offset before 3.11.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Values must be sorted."""
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round(pct / 100 * (len(values) - 1)))))
    return values[idx]


# The prompt Stage 6's probes send. It lives here, not in evaluate.py, because the
# collector has to be able to recognise the profiler's own exhaust in the corpus.
SELF_PROBE_PROMPT = "Say: X"
SELF_PROMPT_KEEP = 2

# The harness writes these into the transcript as user content when a run is cut short. A
# killed `claude -p` subprocess leaves one, so treating it as a person typing would hand an
# abandoned probe back to the user.
INTERRUPTION_PREFIX = "[Request interrupted"

# Every stage that calls `claude -p` leaves sessions behind, and each one the collector
# cannot name gets counted as the user's own usage. Stage 6 was recognised from the start;
# the Stage 5 judge and the Stage 8 usefulness probes were not, which put 48 profiler
# sessions into the user's corpus. Signatures are literals rather than imports because
# judge.py reaches this module through events.py, so importing back would be a cycle —
# `test_self_generated_signatures_match_what_the_probes_actually_send` is what keeps them
# honest when a prompt is reworded.
SELF_SIGNATURES = (
    # (produced_by, how, text). "exact" is a whole short prompt; "prefix" survives a prompt
    # whose body is generated, and is anchored at character 0 so user text cannot match it.
    ("agent_profiler_probe", "exact", SELF_PROBE_PROMPT),
    ("agent_profiler_judge", "prefix_after_count",
     " candidates. For each, was the second identical call necessary?"),
    ("agent_profiler_useful_probe", "exact",
     "What is 17 times 23? Reply with the number only."),
    ("agent_profiler_useful_probe", "prefix",
     "Write a git commit message for this change: the timesheet export "),
    ("agent_profiler_useful_probe", "prefix",
     "Write a git commit message for this change: the asset thumbnail cache "),
)


def _signature_match(prompt: str) -> tuple[str, str] | None:
    """Name the profiler stage that sent this prompt, or None for a real user prompt."""
    for produced_by, how, text in SELF_SIGNATURES:
        if how == "exact" and prompt == text:
            return produced_by, f"prompt is exactly {text!r}"
        if how == "prefix" and prompt.startswith(text):
            return produced_by, f"prompt starts with {text!r}"
        # The judge prompt opens with its own candidate count, so the fixed part begins
        # after digits. Requiring the digits keeps a user quoting the sentence from matching.
        if how == "prefix_after_count" and re.match(r"^\d+" + re.escape(text), prompt):
            return produced_by, f"prompt starts with a count and {text!r}"
    return None


def _prompt_text(blocks: object) -> str:
    if isinstance(blocks, str):
        return blocks.strip()[:200]
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"] \
        if isinstance(blocks, list) else []
    return " ".join(parts).strip()[:200]


def self_generated_verdict(produced_by: str | None, evidence: str | None,
                           user_turns: int, later_real_prompts: int) -> dict | None:
    """The shared rule, over facts either walker can supply.

    The collector keeps a bounded list of prompt texts; the normalizer deliberately does
    not, because its event stream is writable to disk with `--events-out` and user prompt
    text is not something this tool should start copying into artifacts. So the rule is
    stated over derived signals rather than over the prompts themselves, and both walkers
    call this one copy of it. A second implementation would drift, and the direction it
    would drift in is handing the profiler's own sessions back to the user.
    """
    if produced_by is None or user_turns > SELF_PROMPT_KEEP or later_real_prompts:
        return None
    return {
        "self_generated": True,
        "produced_by": produced_by,
        "evidence": f"{evidence}, single turn, no tool call",
    }


def _self_generated(acc: "_SessionAccumulator") -> dict | None:
    """Recognise a session this tool created itself.

    Stage 6 measures a schema by running `claude -p`, and the harness records those runs
    like any other session. They then land in the corpus the profiler reads, so without
    this the tool counts its own exhaust as the user's usage, which on a real corpus is a
    large share of it. Reporting them as the user's activity would be wrong twice: it
    inflates the corpus and it attributes the profiler's spend to the person.

    What makes a session the profiler's is that the profiler sent its only prompt. An
    earlier version also required no tool call, which was wrong: the usefulness probe asks
    for a commit message, and the model reaches for `git log` to check the local style on
    its own. Seven probe sessions were handed back to the user that way. Tool use inside a
    single `-p` run is ordinary model behaviour and says nothing about who started it.

    A *second* user prompt is what rules a session out, because that is a person typing.
    An interruption marker is not a person typing.
    """
    # Only SELF_PROMPT_KEEP prompts are retained, so a longer conversation is declined
    # rather than judged on the part of it that happens to be in memory.
    if not acc.prompts:
        return None
    match = _signature_match(acc.prompts[0])
    produced_by, evidence = match if match else (None, None)
    later_real = sum(1 for p in acc.prompts[1:] if not p.startswith(INTERRUPTION_PREFIX))
    return self_generated_verdict(produced_by, evidence, acc.user_turns, later_real)


class _SessionAccumulator:
    """Mutable per-session tallies, kept bounded while streaming one transcript."""

    def __init__(self) -> None:
        self.record_types: Counter = Counter()
        self.unknown_types: Counter = Counter()
        self.parse_errors: list[dict] = []
        self.parse_error_count = 0
        self.lines = 0

        # Usage is accumulated once per message id; the set bounds re-counting.
        self.seen_message_ids: set[str] = set()
        self.duplicate_usage_records = 0
        self.tokens: Counter = Counter()
        self.tokens_by_model: dict[str, Counter] = {}
        self.synthetic_messages = 0
        # Responses with no stop_reason were interrupted; the harness leaves them out of
        # its cost accounting, so they are tracked apart to keep the comparison honest.
        self.incomplete_messages = 0
        self.incomplete_tokens: Counter = Counter()

        # tool_use id -> name, so tool_result outcomes can be attributed to a tool.
        self.tool_names: dict[str, str] = {}
        self.tool_calls: Counter = Counter()
        self.components_invoked: Counter = Counter()
        self.components_loaded: Counter = Counter()
        self.instructions_loaded: dict[str, dict] = {}
        self.shared_context_chars: Counter = Counter()
        self.duplicate_tool_use_records = 0
        self.tool_ok: Counter = Counter()
        self.tool_error: Counter = Counter()
        self.unmatched_results = 0
        self.error_refs: list[dict] = []

        self.user_turns = 0
        # Only the first couple are kept: they are all a self-session fingerprint needs,
        # and a transcript's prompts are not something to hold in memory wholesale.
        self.prompts: list[str] = []
        self.assistant_messages = 0
        self.thinking_blocks = 0
        self.request_ts: dict[str, datetime] = {}
        self.latencies: list[float] = []
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None

        self.versions: Counter = Counter()
        self.models: Counter = Counter()
        self.cwds: Counter = Counter()
        self.git_branches: Counter = Counter()
        self.sidechain_records = 0
        self.harness_cost: dict | None = None
        self.harness_start: datetime | None = None
        self.harness_tools_used: set[str] = set()
        self.compact_boundaries = 0

    # -- per-record handling ------------------------------------------------------
    def add(self, obj: dict, line_no: int) -> None:
        rtype = obj.get("type") or "__missing_type__"
        self.record_types[rtype] += 1
        if rtype not in KNOWN_RECORD_TYPES:
            self.unknown_types[rtype] += 1

        ts = _parse_ts(obj.get("timestamp"))
        if ts is not None:
            if self.first_ts is None or ts < self.first_ts:
                self.first_ts = ts
            if self.last_ts is None or ts > self.last_ts:
                self.last_ts = ts

        if obj.get("isSidechain"):
            self.sidechain_records += 1
        for field, counter in (("version", self.versions), ("cwd", self.cwds), ("gitBranch", self.git_branches)):
            value = obj.get(field)
            if isinstance(value, str) and value:
                counter[value] += 1

        if rtype == "api-request":
            if isinstance(obj.get("id"), str) and ts is not None:
                self.request_ts[obj["id"]] = ts
            return
        if rtype == "cost-state":
            self.harness_cost = obj
            start = obj.get("startTime")
            if isinstance(start, (int, float)):
                self.harness_start = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
            return
        if rtype == "tools-used":
            tools = obj.get("tools")
            if isinstance(tools, list):
                self.harness_tools_used.update(t for t in tools if isinstance(t, str))
            return
        if rtype == "compact-boundary":
            self.compact_boundaries += 1
            return
        if rtype in REPLAY_RECORD_TYPES:
            # Replays are evidence of what was sent, not of new activity.
            return
        if rtype == "attachment":
            self._add_attachment(obj)
            return

        if rtype == "user":
            self._add_user(obj, line_no)
        elif rtype == USAGE_RECORD_TYPE:
            self._add_assistant(obj, line_no, ts)

    def _add_user(self, obj: dict, line_no: int) -> None:
        message = obj.get("message")
        blocks = message.get("content") if isinstance(message, dict) else None

        # A real prompt carries text; a tool_result-only record is the harness feeding
        # results back, not the person taking another turn.
        is_prompt = False
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                is_prompt = True
            elif btype == "tool_result":
                self._add_tool_result(block, line_no)
        if isinstance(blocks, str) and blocks.strip():
            is_prompt = True
        if is_prompt:
            self.user_turns += 1
            if len(self.prompts) < SELF_PROMPT_KEEP:
                self.prompts.append(_prompt_text(blocks))

    def _add_tool_result(self, block: dict, line_no: int) -> None:
        use_id = block.get("tool_use_id")
        name = self.tool_names.get(use_id) if isinstance(use_id, str) else None
        if name is None:
            # The matching tool_use may appear later, or in a compacted-away prefix.
            self.unmatched_results += 1
            name = "__unattributed__"
        if block.get("is_error"):
            self.tool_error[name] += 1
            if len(self.error_refs) < MAX_REFS:
                self.error_refs.append({"line": line_no, "tool": name, "tool_use_id": use_id})
        else:
            self.tool_ok[name] += 1

    def _add_assistant(self, obj: dict, line_no: int, ts: datetime | None) -> None:
        message = obj.get("message")
        if not isinstance(message, dict):
            return

        model = message.get("model") if isinstance(message.get("model"), str) else "__unknown__"
        msg_id = message.get("id")
        usage = message.get("usage")

        # Usage repeats across the blocks of one response; count it once per message.
        if isinstance(usage, dict):
            if isinstance(msg_id, str) and msg_id in self.seen_message_ids:
                self.duplicate_usage_records += 1
            else:
                if isinstance(msg_id, str):
                    self.seen_message_ids.add(msg_id)
                self.assistant_messages += 1
                self.models[model] += 1
                if model == SYNTHETIC_MODEL:
                    self.synthetic_messages += 1
                incomplete = message.get("stop_reason") is None
                if incomplete:
                    self.incomplete_messages += 1
                self._add_usage(usage, model, incomplete=incomplete)
                # Latency belongs to the response as a whole, so take it once per message.
                ref = obj.get("requestRef")
                started = self.request_ts.get(ref) if isinstance(ref, str) else None
                if started is not None and ts is not None:
                    delta = (ts - started).total_seconds()
                    if delta >= 0:
                        self.latencies.append(delta)

        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "thinking":
                self.thinking_blocks += 1
            elif btype == "tool_use":
                use_id = block.get("id")
                name = block.get("name") if isinstance(block.get("name"), str) else "__unknown__"
                if isinstance(use_id, str):
                    if use_id in self.tool_names:
                        self.duplicate_tool_use_records += 1
                        continue
                    self.tool_names[use_id] = name
                self.tool_calls[name] += 1
                # A Skill-class call names the declared entity it invokes. That name is the
                # only observable link from a declared component to its use.
                component = _component_of(name, block.get("input"))
                if component:
                    self.components_invoked[component] += 1

    def _add_attachment(self, obj: dict) -> None:
        """Record what entered the context window, which is the `loaded` link.

        A shared block (a skill listing names many skills at once) has its size kept apart
        from the per-component tally, because that size cannot be split between the names
        it mentions without inventing an attribution.
        """
        for load in _context_loads(obj):
            if load["load_kind"] == "instructions":
                path = load["components"][0]
                entry = self.instructions_loaded.setdefault(
                    path, {"type": load.get("instruction_type"), "loads": 0, "chars": 0})
                entry["loads"] += 1
                entry["chars"] = max(entry["chars"], load["chars"])
                continue
            for name in load["components"]:
                self.components_loaded[name] += 1
            if load["shared"]:
                self.shared_context_chars[load["load_kind"]] += load["chars"]

    def _add_usage(self, usage: dict, model: str, *, incomplete: bool = False) -> None:
        per_model = self.tokens_by_model.setdefault(model, Counter())
        for field, key in (
            ("input_tokens", "input"),
            ("cache_read_input_tokens", "cache_read"),
            ("cache_creation_input_tokens", "cache_creation"),
            ("output_tokens", "output"),
        ):
            value = usage.get(field)
            if isinstance(value, int):
                self.tokens[key] += value
                per_model[key] += value
                if incomplete:
                    self.incomplete_tokens[key] += value


def _usage_block(acc: _SessionAccumulator) -> dict:
    tokens = acc.tokens
    # Cached and fresh input are both billed input, at different rates; keep them apart
    # and report the total so neither is mistaken for the whole.
    effective_input = tokens["input"] + tokens["cache_read"] + tokens["cache_creation"]
    return {
        "input_tokens": tokens["input"],
        "cache_read_input_tokens": tokens["cache_read"],
        "cache_creation_input_tokens": tokens["cache_creation"],
        "effective_input_tokens": effective_input,
        "output_tokens": tokens["output"],
        "total_tokens": effective_input + tokens["output"],
        "by_model": {
            model: {
                "input_tokens": c["input"],
                "cache_read_input_tokens": c["cache_read"],
                "cache_creation_input_tokens": c["cache_creation"],
                "output_tokens": c["output"],
            }
            for model, c in sorted(acc.tokens_by_model.items())
        },
    }


def _latency_block(acc: _SessionAccumulator) -> dict:
    values = sorted(acc.latencies)
    if not values:
        return {"requests_measured": 0, "note": "No api-request/assistant pairs were linkable in this session."}
    return {
        "requests_measured": len(values),
        "total_seconds": round(sum(values), 2),
        "min_seconds": round(values[0], 2),
        "median_seconds": round(_percentile(values, 50), 2),
        "p90_seconds": round(_percentile(values, 90), 2),
        "max_seconds": round(values[-1], 2),
        "method": "assistant.timestamp - api-request.timestamp, matched on assistant.requestRef == api-request.id",
    }


def _tool_block(acc: _SessionAccumulator) -> dict:
    names = sorted(set(acc.tool_calls) | set(acc.tool_ok) | set(acc.tool_error))
    per_tool = {}
    for name in names:
        if name == "__unattributed__":
            continue
        per_tool[name] = {
            "calls": acc.tool_calls[name],
            "ok": acc.tool_ok[name],
            "errors": acc.tool_error[name],
        }
    return {
        "calls_total": sum(acc.tool_calls.values()),
        "distinct_tools": len(per_tool),
        "errors_total": sum(acc.tool_error.values()),
        "by_tool": per_tool,
        "components_invoked": dict(acc.components_invoked.most_common()),
        "components_loaded": dict(acc.components_loaded.most_common()),
        "instructions_loaded": dict(sorted(acc.instructions_loaded.items())),
        "shared_context_chars": dict(acc.shared_context_chars),
        "unattributed_results": acc.unmatched_results,
        "error_references": acc.error_refs,
    }


def _harness_block(acc: _SessionAccumulator) -> dict | None:
    """Harness-reported totals, kept separate from our own counts as an independent check."""
    cost = acc.harness_cost
    if not isinstance(cost, dict):
        return None
    model_usage = cost.get("modelUsage") if isinstance(cost.get("modelUsage"), dict) else {}
    return {
        "source": "cost-state record written by Claude Code",
        "total_cost_usd": cost.get("totalCostUSD"),
        "total_api_duration_ms": cost.get("totalAPIDuration"),
        "total_api_duration_without_retries_ms": cost.get("totalAPIDurationWithoutRetries"),
        "total_tool_duration_ms": cost.get("totalToolDuration"),
        "total_duration_ms": cost.get("totalDuration"),
        "lines_added": cost.get("totalLinesAdded"),
        "lines_removed": cost.get("totalLinesRemoved"),
        "has_unknown_model_cost": cost.get("hasUnknownModelCost"),
        "model_usage": model_usage,
        "tools_used": sorted(acc.harness_tools_used),
    }


def _resume_gap_seconds(acc: _SessionAccumulator) -> float | None:
    """Seconds of transcript that predate the run the harness billed, if any.

    A resumed session keeps the earlier history in the same transcript, while cost-state
    only accounts for the current run. That makes the transcript a superset of what the
    harness billed, so collected totals legitimately exceed the reported ones.
    """
    if acc.harness_start is None or acc.first_ts is None:
        return None
    gap = (acc.harness_start - acc.first_ts).total_seconds()
    return round(gap, 1) if gap > 0 else None


def _cross_check(acc: _SessionAccumulator, harness: dict | None) -> dict:
    """Compare our derived token totals against the harness's own, and show any gap.

    Agreement is evidence that the de-duplication rules are right. A gap is reported
    rather than hidden, because an unexplained gap means one of the two is wrong.
    """
    if not harness:
        return {"available": False, "reason": "session has no cost-state record"}

    resume_gap = _resume_gap_seconds(acc)

    reported = Counter()
    for stats in harness.get("model_usage", {}).values():
        if not isinstance(stats, dict):
            continue
        for field, key in (
            ("inputTokens", "input"),
            ("cacheReadInputTokens", "cache_read"),
            ("cacheCreationInputTokens", "cache_creation"),
            ("outputTokens", "output"),
        ):
            value = stats.get(field)
            if isinstance(value, int):
                reported[key] += value

    fields = {}
    over_counted = []
    for key in ("input", "cache_read", "cache_creation", "output"):
        ours = acc.tokens[key] - acc.incomplete_tokens[key]
        theirs = reported[key]
        delta = ours - theirs
        pct = (delta / theirs * 100) if theirs else None
        fields[key] = {
            "collected": ours,
            "harness_reported": theirs,
            "delta": delta,
            "relative_delta_pct": round(pct, 2) if pct is not None else None,
            "exact_match": delta == 0,
        }
        # Only a materially larger count indicates a de-duplication defect. Small positive
        # deltas are observed on real sessions where the two accountings genuinely differ
        # (see docs/dogfood/trace-schema.md); calling those "double counting" would be a
        # false alarm, and widening the band until everything passes would be worse.
        if pct is not None and pct > OVER_COUNT_TOLERANCE_PCT and resume_gap is None:
            over_counted.append(key)

    tools_match = sorted(acc.harness_tools_used) == sorted(
        n for n in acc.tool_calls if n != "__unknown__"
    )
    worst = max((f["relative_delta_pct"] or 0) for f in fields.values())
    if over_counted:
        verdict = "double_counting_suspected"
    elif resume_gap is not None and worst > 0:
        verdict = "resumed_session_superset"
    elif all(f["exact_match"] for f in fields.values()):
        verdict = "exact_match"
    elif worst > 0:
        verdict = "unexplained_positive_variance"
    else:
        verdict = "under_collected_as_expected"

    return {
        "available": True,
        "tokens": fields,
        "verdict": verdict,
        "comparison_basis": "completed responses only; interrupted responses are excluded "
        "because the harness does not bill them",
        "interrupted_messages_excluded": acc.incomplete_messages,
        "interrupted_tokens_excluded": dict(sorted(acc.incomplete_tokens.items())),
        "resumed_pre_billing_seconds": resume_gap,
        "over_counted_fields": over_counted,
        "double_counting_suspected": bool(over_counted),
        "tool_names_agree": tools_match,
        "interpretation": {
            "exact_match": "Per-message sums reproduce the harness totals exactly.",
            "under_collected_as_expected": "Collected no more than the harness on every field, "
            "which is what compacted or resumed history produces.",
            "unexplained_positive_variance": f"Collected up to {worst:.2f}% more than the harness on "
            "some field, below the de-duplication alarm threshold and not attributed to a known cause.",
            "double_counting_suspected": "Collected materially more than the harness: a repeated "
            "record is being counted twice.",
            "resumed_session_superset": f"Transcript begins {resume_gap or 0:.0f}s before the run "
            "cost-state billed, so it legitimately holds more activity than the harness reported.",
        }[verdict],
        "note": "The harness total is session-cumulative and survives compaction, so it is a "
        "reference point rather than an exact expectation. Over-counting is the defect this guards.",
    }


def collect_session(path: str) -> dict:
    """Parse one Claude Code transcript into deterministic observations."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"trace file does not exist: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"trace path is not a file: {p}")

    started = time.perf_counter()
    acc = _SessionAccumulator()
    session_ids: Counter = Counter()

    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            acc.lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                acc.parse_error_count += 1
                if len(acc.parse_errors) < MAX_PARSE_ERRORS:
                    acc.parse_errors.append({"line": line_no, "error": exc.msg})
                continue
            if not isinstance(obj, dict):
                acc.parse_error_count += 1
                if len(acc.parse_errors) < MAX_PARSE_ERRORS:
                    acc.parse_errors.append({"line": line_no, "error": "record is not an object"})
                continue
            sid = obj.get("sessionId")
            if isinstance(sid, str):
                session_ids[sid] += 1
            acc.add(obj, line_no)

    harness = _harness_block(acc)
    duration = None
    if acc.first_ts and acc.last_ts:
        duration = round((acc.last_ts - acc.first_ts).total_seconds(), 2)

    observed = {
        "records": acc.lines,
        "record_types": dict(sorted(acc.record_types.items())),
        "user_turns": acc.user_turns,
        "assistant_messages": acc.assistant_messages,
        "thinking_blocks": acc.thinking_blocks,
        "synthetic_messages": acc.synthetic_messages,
        "interrupted_messages": acc.incomplete_messages,
        "sidechain_records": acc.sidechain_records,
        "compact_boundaries": acc.compact_boundaries,
        "wall_clock_seconds": duration,
        "tokens": _usage_block(acc),
        "tools": _tool_block(acc),
        "latency": _latency_block(acc),
        "models": dict(sorted(acc.models.items())),
    }
    origin = _self_generated(acc)
    if origin:
        observed["origin"] = origin

    return {
        "schema_version": "0.3",
        "mode": "collect",
        "harness": "claude_code",
        "source": str(p),
        "read_only": True,
        "session": {
            "session_ids": sorted(session_ids),
            "bytes": p.stat().st_size,
            "cwd": sorted(acc.cwds),
            "git_branches": sorted(acc.git_branches),
            "harness_versions": sorted(acc.versions),
        },
        "observed": observed,
        "harness_reported": harness,
        "cross_check": _cross_check(acc, harness),
        "integrity": {
            "parse_errors": acc.parse_error_count,
            "parse_error_references": acc.parse_errors,
            "unknown_record_types": dict(sorted(acc.unknown_types.items())),
            "duplicate_usage_records_skipped": acc.duplicate_usage_records,
            "duplicate_tool_use_records_skipped": acc.duplicate_tool_use_records,
        },
        "profiler_cost": {
            "stage": "trace_collect",
            "bytes_read": p.stat().st_size,
            "lines_parsed": acc.lines,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": [
            "Token usage is counted once per message id: one API response is written as "
            "one record per content block, each repeating the same usage.",
            "Tool calls are counted from assistant records only and de-duplicated by "
            "tool_use id; api-request-blob records replay the request and are not activity.",
            "Cached and fresh input tokens are reported separately; effective_input_tokens is their sum.",
            "A tool call is Observed. Whether it was useful is not decided here.",
        ],
    }


def _relocation_candidates(base: Path, claude_home: Path) -> list[dict]:
    """History recorded under a different path for what looks like the same project.

    Transcripts are filed under a key derived from the working directory, so moving a
    project orphans everything recorded before the move. Asked about the new location, a
    collector that just reports zero gives a clean-looking answer that is wrong.

    Each candidate is evidenced by the `cwd` the transcript itself recorded, and candidates
    are surfaced rather than adopted: a same-named directory elsewhere is a reason to ask,
    not proof that two paths are one project.
    """
    projects = claude_home / "projects"
    if not projects.is_dir():
        return []
    found: list[dict] = []
    for entry in sorted(projects.iterdir()):
        if not entry.is_dir() or entry.name == _encode_project_key(base):
            continue
        transcripts = sorted(entry.glob("*.jsonl"))
        if not transcripts:
            continue
        recorded = _recorded_cwd(transcripts[-1])
        if recorded is None or Path(recorded).name != base.name:
            continue
        found.append({
            "project_dir": str(entry),
            "recorded_cwd": recorded,
            "sessions": len(transcripts),
            "basis": "the transcript records this cwd, whose final path segment matches the "
            "requested root",
        })
    return found


def _recorded_cwd(path: Path) -> str | None:
    """The cwd a transcript says it ran in, read from its first record that carries one."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(50):
                line = fh.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cwd = obj.get("cwd") if isinstance(obj, dict) else None
                if isinstance(cwd, str) and cwd:
                    return cwd
    except OSError:
        return None
    return None


def _session_entry(entry: Path, *, project_key: str | None = None) -> dict | None:
    try:
        st = entry.stat()
    except OSError:
        return None
    row = {
        "session_id": entry.stem,
        "path": str(entry),
        "bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    if project_key is not None:
        row["project_key"] = project_key
        # The transcript's own recorded cwd is the only reliable link back to a directory:
        # the project key is derived from where the work happened, and projects move.
        row["recorded_cwd"] = _recorded_cwd(entry)
    return row


def discover_sessions(root: str, *, whole_environment: bool = False) -> dict:
    """List Claude Code transcripts.

    Rooted at a project by default. With `whole_environment` the question stops being
    "what happened in this directory" and becomes "what happened across the whole stack":
    every project key under `~/.claude/projects` is enumerated, each session carrying the
    key it is filed under and the cwd its transcript records. A project that moved then
    needs no special handling, because nothing is being matched by path.
    """
    if not root or not root.strip():
        raise ValueError("sessions root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"sessions root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"sessions root is not a directory: {base}")

    started = time.perf_counter()
    claude_home = _user_tree("AGENT_PROFILER_CLAUDE_HOME", ".claude")
    project_dir = claude_home / "projects" / _encode_project_key(base)

    sessions = []
    projects_seen: list[str] = []
    if whole_environment:
        projects_dir = claude_home / "projects"
        for directory in sorted(p for p in projects_dir.glob("*") if p.is_dir()):
            found = [row for row in
                     (_session_entry(e, project_key=directory.name)
                      for e in sorted(directory.glob("*.jsonl")))
                     if row is not None]
            if found:
                projects_seen.append(directory.name)
                sessions.extend(found)
    elif project_dir.is_dir():
        sessions = [row for row in (_session_entry(e) for e in sorted(project_dir.glob("*.jsonl")))
                    if row is not None]

    sessions.sort(key=lambda s: s["modified"], reverse=True)
    # Relocation only matters when a root-scoped look found nothing; a whole-environment
    # pass already has every project in hand.
    relocated = ([] if whole_environment or sessions
                 else _relocation_candidates(base, claude_home))
    observed = {
        "sessions_found": len(sessions),
        "bytes_total": sum(s["bytes"] for s in sessions),
    }
    if whole_environment:
        observed["projects_found"] = len(projects_seen)
        observed["sessions_by_project"] = {
            key: sum(1 for s in sessions if s["project_key"] == key) for key in projects_seen
        }
    return {
        "schema_version": "0.3",
        "mode": "sessions",
        "harness": "claude_code",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "project_dir": str(project_dir),
        "project_dir_exists": project_dir.is_dir(),
        "relocation_candidates": relocated,
        "observed": observed,
        "sessions": sessions,
        "profiler_cost": {
            "stage": "session_discovery",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": [
            "Transcript presence proves a session was recorded for this root, not that the "
            "work in it succeeded.",
            "A resumed or compacted session can omit earlier history; totals are per transcript.",
        ],
    }


def collect_root(root: str, limit: int | None = None, *, whole_environment: bool = False,
                 use_cache: bool = True) -> dict:
    """Collect every session for a root, newest first, with an aggregate roll-up.

    With `whole_environment` the corpus is every project's transcripts rather than one
    directory's, which is what makes a statement about the environment possible.

    A finished transcript never changes, so each session's contribution is reused when the
    file's bytes are — see `core.FileCache` for why the key trusts nothing else.
    """
    index = discover_sessions(root, whole_environment=whole_environment)
    started = time.perf_counter()
    selected = index["sessions"][: limit if limit and limit > 0 else None]

    collected = []
    failures = []
    totals: Counter = Counter()
    tools: Counter = Counter()
    tool_errors: Counter = Counter()
    components: Counter = Counter()
    loaded: Counter = Counter()
    instructions: dict[str, dict] = {}
    shared_chars: Counter = Counter()
    per_project: dict[str, dict] = defaultdict(
        lambda: {"sessions": 0, "tool_calls": 0, "tokens": 0}
    )
    self_made = {"sessions": 0, "tokens": 0}
    self_made_by: Counter = Counter()
    bytes_read = 0
    lines = 0
    cache = FileCache("collect", enabled=use_cache)

    def summarize(path):
        full = collect_session(path)
        # Only what this function reads is kept.
        return {
            "observed": full["observed"],
            "harness_reported": full["harness_reported"],
            "cross_check": full["cross_check"],
            "integrity": full["integrity"],
            "bytes_read": full["profiler_cost"]["bytes_read"],
            "lines_parsed": full["profiler_cost"]["lines_parsed"],
        }

    for entry in selected:
        try:
            result = cache.get_or_compute(Path(entry["path"]),
                                          lambda p=entry["path"]: summarize(p))
        except (OSError, ValueError) as exc:
            failures.append({"path": entry["path"], "error": exc.__class__.__name__})
            continue
        obs = result["observed"]
        bytes_read += result["bytes_read"]
        lines += result["lines_parsed"]
        if obs.get("origin", {}).get("self_generated"):
            # Counted and reported, but kept out of the user's totals: this session is the
            # profiler's own exhaust, and attributing it to the person would be a lie
            # about both the corpus size and who spent the money.
            self_made["sessions"] += 1
            self_made["tokens"] += obs["tokens"]["total_tokens"]
            self_made_by[obs["origin"]["produced_by"]] += 1
            continue
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
            totals[key] += obs["tokens"][key]
        totals["user_turns"] += obs["user_turns"]
        totals["assistant_messages"] += obs["assistant_messages"]
        totals["tool_calls"] += obs["tools"]["calls_total"]
        totals["tool_errors"] += obs["tools"]["errors_total"]
        for name, stats in obs["tools"]["by_tool"].items():
            tools[name] += stats["calls"]
            tool_errors[name] += stats["errors"]
        for component, count in (obs["tools"].get("components_invoked") or {}).items():
            components[component] += count
        # Counted once per session, not once per listing. A session that resumes or
        # compacts receives the skill listing again, so summing occurrences produced
        # a count that exceeded its own denominator, which disproves itself.
        # The report says "in N of M sessions", so N has to be sessions.
        for component in (obs["tools"].get("components_loaded") or {}):
            loaded[component] += 1
        for instr_path, instr in (obs["tools"].get("instructions_loaded") or {}).items():
            agg = instructions.setdefault(instr_path, {"type": instr["type"], "loads": 0, "chars": 0})
            agg["loads"] += instr["loads"]
            agg["chars"] = max(agg["chars"], instr["chars"])
        for kind, chars in (obs["tools"].get("shared_context_chars") or {}).items():
            shared_chars[kind] += chars
        row = {
            "session_id": entry["session_id"],
            "path": entry["path"],
            "observed": obs,
            "harness_reported": result["harness_reported"],
            "cross_check": result["cross_check"],
            "integrity": result["integrity"],
        }
        if "project_key" in entry:
            row["project_key"] = entry["project_key"]
            row["recorded_cwd"] = entry.get("recorded_cwd")
            per_project[entry["project_key"]]["sessions"] += 1
            per_project[entry["project_key"]]["tool_calls"] += obs["tools"]["calls_total"]
            per_project[entry["project_key"]]["tokens"] += obs["tokens"]["input_tokens"] \
                + obs["tokens"]["cache_read_input_tokens"] \
                + obs["tokens"]["cache_creation_input_tokens"] + obs["tokens"]["output_tokens"]
            per_project[entry["project_key"]].setdefault("recorded_cwd", entry.get("recorded_cwd"))
        collected.append(row)

    cache.flush()
    effective_input = (
        totals["input_tokens"] + totals["cache_read_input_tokens"] + totals["cache_creation_input_tokens"]
    )
    return {
        "schema_version": "0.3",
        "mode": "collect_root",
        "harness": "claude_code",
        "root": index["root"],
        "scope": index["scope"],
        "read_only": True,
        "observed": {
            "sessions_available": index["observed"]["sessions_found"],
            "relocation_candidates": index.get("relocation_candidates") or [],
            "sessions_collected": len(collected),
            "self_generated": {
                **self_made,
                "by": dict(self_made_by.most_common()),
                "note": "Sessions this tool created itself (Stage 6 probes run `claude -p`, "
                        "which the harness records like any other session). Excluded from "
                        "every total above so the corpus describes the user's work, not the "
                        "profiler's.",
            },
            **({"projects_collected": len(per_project),
                "by_project": {k: per_project[k] for k in sorted(
                    per_project, key=lambda k: -per_project[k]["tokens"])}}
               if whole_environment else {}),
            "user_turns": totals["user_turns"],
            "assistant_messages": totals["assistant_messages"],
            "tokens": {
                "input_tokens": totals["input_tokens"],
                "cache_read_input_tokens": totals["cache_read_input_tokens"],
                "cache_creation_input_tokens": totals["cache_creation_input_tokens"],
                "effective_input_tokens": effective_input,
                "output_tokens": totals["output_tokens"],
                "total_tokens": effective_input + totals["output_tokens"],
            },
            "tool_calls": totals["tool_calls"],
            "tool_errors": totals["tool_errors"],
            "calls_by_tool": dict(tools.most_common()),
            "components_invoked": dict(components.most_common()),
            "components_loaded": dict(loaded.most_common()),
            "instructions_loaded": dict(sorted(instructions.items())),
            "shared_context_chars": dict(shared_chars),
            "errors_by_tool": {k: v for k, v in tool_errors.most_common() if v},
        },
        "sessions": collected,
        "failures": failures,
        # Never silent: a reused contribution must not be indistinguishable from a fresh
        # read, and `--no-cache` re-reads everything.
        "reuse": {"enabled": cache.enabled,
                  "sessions_reused": cache.reused,
                  "sessions_read": cache.computed,
                  "keyed_on": "transcript bytes, package source and path"},
        "profiler_cost": {
            "stage": "trace_collect_root",
            "sessions": len(collected),
            "bytes_read": bytes_read,
            "lines_parsed": lines,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": [
            "Aggregates cover the collected transcripts only; use sessions_available to see coverage.",
            "Tool call counts are Observed. Value and avoidability are not claimed here.",
        ],
    }

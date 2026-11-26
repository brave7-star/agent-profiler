"""The `useful` link: does a component change what the agent actually produces?

Every previous route to `useful` failed. Re-running a real corpus costs thousands. The
Stage 5 LLM judge was expensive and weak, and a model grading whether another model's output
was "better" is an opinion wearing a number. This module takes a third shape: tasks whose
answers are **machine-verifiable**, so success is a fact.

    subject_ends_with_reference  with 3/3, 3/3    without 0/3, 0/3
    body_line_length_limit       with 3/3, 3/3    without 0/3, 0/3
    arithmetic (control)         with 3/3         without 3/3

That is the project's first `useful` evidence: without this file the agent does not follow
the user's stated conventions, and with it, it does.

## Four controls, because a one-armed result proves nothing

1. **A negative control.** An arithmetic task runs in both arms. If the `without` arm fails
   it too, that arm is simply broken and the headline result is an artifact. A failed control
   invalidates the whole measurement rather than being noted beside it.
2. **An anti-recitation guard.** An instruction file containing a worked example can be
   *copied* rather than applied. The first draft of this experiment asked for a commit
   message describing the very bug the file uses as its example, so a perfect score needed
   no generalization at all. Scenarios now avoid the example, and reciting it voids the run.
3. **Checks must be content-specific.** Injecting a *different* instruction file, which
   states a trailing-reference rule but says nothing about body format, moved the reference
   check 3/3 and moved the body checks 0/3. So these checks respond to specific rules, not
   to the mere presence of a conventions document — the live alternative explanation.
4. **Only rules the file actually states are scored.** See below; this one is structural.

## Why a check must quote the rule it enforces

A fifth check, "the title is written in Chinese", separated just as cleanly as the others in
the first run. It is not a stated rule anywhere — it was inferred from an example being
Chinese, and it duly came back 2/3 when predicted 3/3. Scoring it would have meant measuring
compliance with a convention the user never wrote.

So a check is scorable only when its parameter can be read out of the file's own text. A
file that does not state a rule cannot be credited or faulted for it, and no check enters
the scored set without a stated rule behind it.

The parameters are extracted rather than listed, which is the second half of the same
principle. An early version held one author's conventions as literal strings — their
reference spelling, their limits, their wording. Anyone whose conventions read differently
got nothing scorable, and the components that did pass were the ones restating those same
conventions, which kept every measurement inside one setup. The rule kinds are general; the
numbers and spellings come from whatever the tool is pointed at.

## Arms

Each file is measured in the harness that loads it, because a result from the other one
would be attributed to the wrong component. The checks, the controls and the verdict are
shared; only the arm construction differs.

**Claude Code** (`CLAUDE.md`):

    without = --bare
    with    = --bare --append-system-prompt-file <verbatim copy of the real file>

`--bare` skips CLAUDE.md, memory, skills and hooks, so the baseline is minimal and exactly
one file is added back. This arm carries a caveat: the content is byte-identical to the real
file, but it arrives as an appended system prompt rather than through the harness's own
CLAUDE.md loading.

**Codex** (`AGENTS.md`):

    without = codex exec in a directory with no AGENTS.md above it
    with    = the same, with a verbatim copy placed as the project AGENTS.md

This arm needs *no* channel caveat, which makes it the stronger of the two. Codex finds the
real `~/AGENTS.md` by walking up from the working directory, and finds the injected copy the
same way, so the measurement uses the harness's own loading rather than a substitute for it.
The walk-up is also the hazard: a temp directory inside HOME would reach the real file and
contaminate the baseline, so the baseline is *checked* for an ancestor `AGENTS.md` and the
measurement refuses rather than reporting the resulting near-zero difference as "no effect".

Routes that do not work, recorded so nobody re-spends on them: a project `CLAUDE.md` in cwd
is ignored under `--bare` (auto-discovery is off), and `--add-dir` does not carry the file
either, despite bare mode's help listing it. Removal-based arms are hopeless for the usual
reason — hiding user settings hides the auth token with them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PROBE_REPEATS = 3
RETRY_BUDGET = 3
# Named so a test can zero it. The flake clusters in time, so an immediate retry is
# likely to land in the same cluster and waiting is the point of retrying at all.
RETRY_BACKOFF_SECONDS = 5
PROBE_TIMEOUT = 240

# A run that stops answering and never recovers, told apart from the known intermittent
# failure by its shape rather than by a message — the harness exits 1 with an empty stderr
# either way. Measured on a real 90-probe run: 36 consecutive successes, then 54
# consecutive failures with no recovery, while the intermittent failure's longest run of
# consecutive failures across earlier artifacts is 0. Six in a row is well beyond what the
# intermittent one produces and well short of the 54 that were burned for nothing.
#
# The retry budget cannot cover this: it is sized for an independent ~44% failure rate, so
# three attempts against a condition that persists all fail, and the batch continues
# spending into a wall. Every probe after the wall is money for a result that will read as
# "no valid runs".
CONSECUTIVE_FAILURE_LIMIT = 6

# Recalibrated from the <1000 guard used for default-mode probes, which misclassifies
# `--bare`'s legitimate 864-token prefix as an authentication failure. The real failure
# signature is 2-4 tokens, so a floor far below any genuine prefix separates them cleanly.
MIN_VALID_CTX = 100

# Measured over a real run rather than guessed, and used for the budget gate.
PROBE_COST_USD = 1.15

# A compliance task is one short answer. Far above the harness's fixed prefix means the run
# went off and did agentic work instead, which is how a stdin leak once cost many times a
# controlled probe — a
# runaway run must be a rejected sample, not an expensive data point.
PROBE_CONTEXT_CEILING = 60_000

CJK = re.compile(r"[一-鿿]")

# A commit title inside a fenced block in the instruction file is a worked example, and an
# arm that reproduces it has recited rather than applied the rule.
EXAMPLE_TITLE = re.compile(r"^(?:feat|fix|refactor|docs|chore|test|perf)[(:]\S*.*#\d+\s*$")


def _title_and_body(text: str) -> tuple[str, list[str]]:
    """Split a commit message, ignoring any fencing the model wrapped around it."""
    lines = list(text.strip().splitlines())
    while lines and (lines[0].startswith("```") or not lines[0].strip()):
        lines.pop(0)
    while lines and lines[-1].startswith("```"):
        lines.pop()
    if not lines:
        return "", []
    return lines[0].strip(), [l.strip() for l in lines[1:] if l.strip()]


def _example_titles(content: str) -> list[str]:
    """Commit titles the instruction file demonstrates, for the anti-recitation guard."""
    return [line.strip() for line in content.splitlines()
            if EXAMPLE_TITLE.match(line.strip())]


# Rule *kinds*, with the parameters read out of the file under test.
#
# An earlier version hard-coded one person's conventions as literal markers — their issue-id
# spelling, their character limits, their wording for "one point per line". That made the
# library useless to anyone whose conventions are worded differently, and worse: the
# components it then selected for measurement were the ones restating those same
# conventions, so the whole chain stayed inside one person's setup. A tool holds the
# mechanism; the specifics belong to whatever it is pointed at.
#
# Being stated is still required, and that requirement is what `extract` enforces: a rule
# counts as stated only when its parameter can be read out of the text. That is the same
# guard the literal markers provided — a check inferred from an example rather than from a
# rule separated cleanly once and was measuring a convention nobody had written down — and
# it now works for any wording instead of one.
#
# A limit is checked with a small tolerance because a bullet prefix and trailing punctuation
# are not part of the text a length rule is about, and two characters should not decide a
# verdict.
LENGTH_TOLERANCE = 4

# A trailing reference, in the shapes conventions actually specify: `#123`, `ABC-123`,
# `(#123)`, `GH-123`. The *rule* is "the subject ends with a reference"; which spelling a
# reader uses is read from their own text rather than assumed.
_REFERENCE = r"[(\[]?(?:#|[A-Z][A-Z0-9]+-|GH-)\d+[)\]]?[.,;:]?"

RULE_KINDS = (
    {
        "name": "subject_ends_with_reference",
        "rule": "the subject line ends with an issue or task reference",
        # Any statement that a reference belongs at the end, in either language, without
        # depending on how the reference itself is spelled.
        # "at the end of the subject", "标题末尾", and the equally common form that names
        # the message rather than its first line ("commit message 末尾带 ..."). Narrowing
        # this to title-or-subject missed a real file that states the rule that second way.
        "extract": lambda text: bool(re.search(
            r"(?:(?:at the |at )?end of the (?:title|subject|first line|commit message)"
            # A modal or adverb commonly sits between the noun and the verb ("subject must
            # end with", "subject should always end with"), so a few words are allowed.
            r"|(?:title|subject|first line|commit message|message)"
            r"(?:\s+\w+){0,3}\s+ends?\s+with"
            # The imperative, which is how instruction files usually address the agent:
            # "always end the subject line with ...". The declarative forms above all put
            # the noun first, so this whole mood was invisible to them — and it is the more
            # natural one to write a rule in.
            r"|end(?:ing)?\s+(?:the|your|each|every)\s+"
            r"(?:title|subject|first line|commit message|message)(?:\s+line)?\s+with"
            r"|(?:title|subject|标题|commit message|提交信息)[^\n]{0,24}(?:末尾|结尾))",
            text, re.I)),
        "predicate": lambda _: lambda subject, body: bool(
            re.search(_REFERENCE + r"$", subject.strip())),
    },
    {
        "name": "subject_length_limit",
        "rule": "the subject line stays within a stated length",
        "extract": lambda text: _stated_limit(
            text, r"(?:title|subject|first line|标题)[^\n]{0,40}?(\d{2,3})"),
        "predicate": lambda limit: lambda subject, body: (
            0 < len(subject) <= limit + LENGTH_TOLERANCE),
    },
    {
        "name": "body_is_itemised",
        "rule": "the body is written as one item per line",
        "extract": lambda text: bool(re.search(
            r"分点|一点一件事|每行一个|one (?:point|item|thing) per|bullet", text, re.I)),
        "predicate": lambda _: lambda subject, body: bool(body) and all(
            line.startswith(("-", "*", "•", "+")) for line in body),
    },
    {
        "name": "body_line_length_limit",
        "rule": "each body line stays within a stated length",
        "extract": lambda text: _stated_limit(
            text, r"(?:body|each line|per line|正文|每行)[^\n]{0,40}?(\d{2,3})"),
        "predicate": lambda limit: lambda subject, body: bool(body) and all(
            len(line) <= limit + LENGTH_TOLERANCE for line in body),
    },
)


def _stated_limit(text: str, pattern: str) -> int | None:
    """A numeric limit the text states, or None when it states none.

    Only counted where a unit follows the number, so a step number or a year cannot be read
    as a length. The smallest match wins: a document may state a limit twice, and the
    stricter one is the one a writer is held to.
    """
    found = []
    for match in re.finditer(pattern, text, re.I):
        tail = text[match.end():match.end() + 14]
        if re.match(r"\s*(?:characters?|chars?|columns?|字|个字)", tail, re.I):
            found.append(int(match.group(1)))
    return min(found) if found else None


# Two unrelated scenarios, because one could be a fluke, and neither resembles the worked
# example in the file under test — compliance has to be generalized, not recalled.
PROBE_SCENARIOS = (
    {
        "id": "commit_timesheet",
        "prompt": ("Write a git commit message for this change: the timesheet export "
                   "dropped its final row because of an off-by-one in the page loop. "
                   "The tracker item is 2033. Output only the commit message."),
    },
    {
        "id": "commit_cache",
        "prompt": ("Write a git commit message for this change: the asset thumbnail cache "
                   "never expired, so renamed files kept serving stale images. The tracker "
                   "item is 2104. Output only the commit message."),
    },
)

NEGATIVE_CONTROL = {
    "id": "negative_control",
    "prompt": "What is 17 times 23? Reply with the number only.",
    "predicate": lambda t, b: "391" in t or any("391" in l for l in b),
}


def _probe(prompt: str, workdir: Path, instruction: Path | None, timeout: int,
           *, bare: bool = True, extra_args: tuple[str, ...] = ()) -> dict:
    """One Claude Code run, returning its reply and the tokens its prefix carried.

    `bare` and `extra_args` exist for the skill arm, which cannot use `--bare`: bare mode
    skips plugin sync, so `--plugin-dir` silently loads nothing under it. That was found by
    nonce discrimination, not assumed — see `codex_probe.measure_skill_cost`.
    """
    argv = ["claude", "-p", prompt, "--output-format", "json"]
    if bare:
        argv.append("--bare")
    argv += list(extra_args)
    if instruction is not None:
        argv += ["--append-system-prompt-file", str(instruction)]
    try:
        # stdin closed for the reason a runaway probe cost many times a controlled one: `claude -p`
        # appends piped stdin to its prompt, and `capture_output` redirects only the
        # output streams, so an inherited stdin silently becomes part of the task.
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=str(workdir), env=dict(os.environ),
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "cost_usd": 0.0}
    if done.returncode != 0:
        # The harness's own complaint, not just the exit code. 72 of 90 probes in one real
        # run reported nothing but `exit 1`, which was indistinguishable from the known
        # auth/init flake and left no way to tell a transient failure from a wrong argument
        # — the run had to be diagnosed by reproducing it by hand afterwards.
        detail = " ".join((done.stderr or "").split())[:300]
        return {"ok": False, "cost_usd": 0.0,
                "error": f"exit {done.returncode}" + (f": {detail}" if detail else
                                                      " (no stderr)")}
    try:
        envelope = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non-JSON envelope", "cost_usd": 0.0}
    usage = envelope.get("usage") or {}
    ctx = sum(int(usage.get(key) or 0) for key in
              ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    cost = float(envelope.get("total_cost_usd") or 0.0)
    if ctx < MIN_VALID_CTX:
        # The identified authentication/initialization failure class. Counting it as a task
        # failure would manufacture evidence that the component was needed.
        return {"ok": False, "error": f"auth/init failure (ctx={ctx})", "cost_usd": cost}
    return {"ok": True, "text": envelope.get("result") or "", "ctx": ctx, "cost_usd": cost}


def _codex_probe(prompt: str, home: Path, workdir: Path, timeout: int) -> dict:
    """One Codex run, with the reply read from the rollout rather than from stdout."""
    binary = shutil.which("codex")
    if binary is None:
        return {"ok": False, "error": "the `codex` CLI was not found", "cost_usd": 0.0}
    started = time.time()
    env = dict(os.environ, CODEX_HOME=str(home))
    argv = [binary, "exec", "--skip-git-repo-check", "-s", "read-only", prompt]
    try:
        # Same hazard as `claude -p`: `codex exec` appends piped stdin to its prompt.
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=str(workdir), env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "cost_usd": 0.0}
    if done.returncode != 0:
        return {"ok": False, "error": f"exit {done.returncode}: {done.stderr.strip()[:120]}",
                "cost_usd": 0.0}

    newest, newest_mtime = None, started - 1
    for path in home.rglob("rollout-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= newest_mtime:
            newest, newest_mtime = path, mtime
    if newest is None:
        return {"ok": False, "error": "no rollout was written", "cost_usd": 0.0}

    text, tokens = None, None
    for line in newest.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = record.get("payload") or {}
        if (record.get("type") == "response_item" and payload.get("type") == "message"
                and payload.get("role") == "assistant"):
            chunks = [part["text"] for part in payload.get("content") or []
                      if isinstance(part, dict) and part.get("text")]
            if chunks:
                text = "\n".join(chunks)
        info = payload.get("info") or {}
        if isinstance(info, dict):
            usage = info.get("total_token_usage") or {}
            if isinstance(usage, dict) and usage.get("input_tokens"):
                tokens = int(usage["input_tokens"])
    if text is None:
        return {"ok": False, "error": "no assistant message in the rollout", "cost_usd": 0.0}
    if tokens is not None and tokens > PROBE_CONTEXT_CEILING:
        return {"ok": False, "error": f"input tokens ({tokens}) exceed the ceiling; this run "
                                      "did work instead of answering the task",
                "cost_usd": 0.0}
    # Codex reports tokens and no dollar figure, so none is invented here.
    return {"ok": True, "text": text, "ctx": tokens or 0, "cost_usd": None}


def _controlled_codex_home(tmp: Path) -> Path:
    """A CODEX_HOME carrying only credentials and config, with no instruction file."""
    real = Path.home() / ".codex"
    home = tmp / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml"):
        source = real / name
        if source.is_file():
            shutil.copy2(source, home / name)
    return home


def _ancestor_instruction(workdir: Path) -> Path | None:
    """Any `AGENTS.md` above the baseline arm, which would contaminate it.

    Codex finds instruction files by walking *up* from the working directory, and this
    user's real file sits at `~/AGENTS.md` — picked up that way rather than as a special
    global. A temp directory outside HOME therefore never reaches it, but that is an
    assumption about the walk-up, and assuming it is how a contaminated baseline would go
    unnoticed. So it is checked.
    """
    for parent in [workdir, *workdir.parents]:
        candidate = parent / "AGENTS.md"
        try:
            if candidate.is_file() and candidate.read_text(errors="replace").strip():
                return candidate
        except OSError:
            continue
    return None


def _claude_arms(tmp: Path, content: str, filename: str):
    """Claude Code arms: a minimal prefix, with one file appended back."""
    workdir = tmp / "work"
    workdir.mkdir()
    injected = tmp / filename
    injected.write_text(content, encoding="utf-8")

    def run(prompt: str, label: str, timeout: int) -> dict:
        return _probe(prompt, workdir, injected if label == "with" else None, timeout)

    return {
        "run": run,
        "arms": {"without": "--bare",
                 "with": "--bare --append-system-prompt-file <verbatim copy>"},
        "reports_cost": True,
        "channel_caveat":
            "The content is byte-identical to the real file, but it arrives as an appended "
            "system prompt rather than through the harness's own instruction loading, so "
            "this measures the content's behavioural effect and not that plumbing.",
    }


def _codex_arms(tmp: Path, content: str, filename: str):
    """Codex arms: two working directories, one carrying a project `AGENTS.md`."""
    home = _controlled_codex_home(tmp)
    without = tmp / "without"
    without.mkdir()
    with_dir = tmp / "with"
    with_dir.mkdir()
    (with_dir / "AGENTS.md").write_text(content, encoding="utf-8")

    contaminant = _ancestor_instruction(without)
    if contaminant is not None:
        return {"error": f"the baseline arm has an instruction file above it "
                         f"({contaminant}), so both arms would carry instructions"}

    def run(prompt: str, label: str, timeout: int) -> dict:
        return _codex_probe(prompt, home, with_dir if label == "with" else without, timeout)

    return {
        "run": run,
        "arms": {"without": "codex exec in a directory with no AGENTS.md above it",
                 "with": "codex exec in a directory carrying a verbatim copy as AGENTS.md"},
        "reports_cost": False,
        # Notably absent here: a delivery-channel caveat. Codex finds the real `~/AGENTS.md`
        # by walking up from the working directory, and finds the injected copy the same
        # way, so this arm uses the harness's own loading rather than a substitute for it.
        "channel_caveat": None,
    }


def _skill_arms_for(skill_dir: Path):
    """Arms that differ by exactly one skill, built the way the cost arm already is.

    The route is not a substitute for the harness's own loading: `--plugin-dir` is how the
    harness loads a plugin, and a plugin carrying one skill is one skill added. It also
    removes a confound the instruction route has to argue about — `--setting-sources ""`
    clears the user's own instruction files from *both* arms, so a rule those state as well
    cannot account for a difference between them.
    """
    def build(tmp: Path, content: str, filename: str) -> dict:
        from .codex_probe import _plugin_with_one_skill

        workdir = tmp / "work"
        workdir.mkdir()
        # The real directory is copied, never moved or modified: it is the object under
        # measurement, and a probe that edited it would be measuring its own edit.
        plugin = _plugin_with_one_skill(tmp, skill_dir)
        base = ("--setting-sources", "")
        arm_args = {"without": base, "with": base + ("--plugin-dir", str(plugin))}

        def run(prompt: str, label: str, timeout: int) -> dict:
            # `bare` is off deliberately: bare mode skips plugin sync, so `--plugin-dir`
            # would load nothing and the arms would be identical while appearing not to be.
            return _probe(prompt, workdir, None, timeout,
                          bare=False, extra_args=arm_args[label])

        return {
            "run": run,
            "arms": {"without": '--setting-sources ""',
                     "with": '--setting-sources "" --plugin-dir <copy of this skill>'},
            "reports_cost": True,
            "channel_caveat":
                "The skill arrives as a plugin, so its listing entry carries a plugin "
                "prefix the installed copy does not have. The body is byte-identical.",
        }

    return build


HARNESS_ARMS = {"claude_code": _claude_arms, "codex": _codex_arms}

# Which harness loads which instruction file, so each is measured where it actually applies.
INSTRUCTION_HARNESS = {"claude_md": "claude_code", "agents_md": "codex"}


def _probe_with_retry(run, prompt: str, label: str, timeout: int,
                      spend: list[float]) -> dict:
    """The auth/init flake runs near 40% and clusters in time, so a dead cell is retried.

    The wait between attempts goes through `RETRY_BACKOFF_SECONDS` so a test can set it to
    zero. It was a literal, and the tests that exercise failure paths had to sit through it:
    four of them spent 135 seconds of the suite asleep, which is slow enough that a suite
    nobody wants to run stops being a safety net.
    """
    last: dict = {"ok": False, "error": "no attempt"}
    for attempt in range(RETRY_BUDGET):
        result = run(prompt, label, timeout)
        if result.get("cost_usd") is not None:
            spend.append(result["cost_usd"])
        if result["ok"]:
            return result
        last = result
        if attempt + 1 < RETRY_BUDGET and RETRY_BACKOFF_SECONDS:
            time.sleep(RETRY_BACKOFF_SECONDS)
    return last


def scorable_checks(content: str) -> tuple[list[dict], list[dict]]:
    """The rules this text states, each bound to the parameter the text gives it.

    A rule kind becomes a scorable check only when its parameter can be read out of the
    content, so "stated" is decided by the document rather than by a list of someone's
    wordings. The returned checks carry a built predicate, which is why a caller never sees
    a parameter it would have to supply itself.
    """
    stated, not_stated = [], []
    for kind in RULE_KINDS:
        parameter = kind["extract"](content)
        if parameter:
            stated.append({"name": kind["name"], "rule": kind["rule"],
                           "parameter": None if parameter is True else parameter,
                           "predicate": kind["predicate"](parameter)})
        else:
            not_stated.append({"name": kind["name"], "rule": kind["rule"]})
    return stated, not_stated


def measure_instruction_usefulness(instruction_file: Path, *, repeats: int = PROBE_REPEATS,
                                   timeout: int = PROBE_TIMEOUT,
                                   harness: str = "claude_code") -> dict:
    """Does this instruction file change verifiable properties of the agent's output?

    `harness` selects the arms. The checks, the controls and the verdict are identical
    either way — a rule a machine can verify is harness-independent, and only the way an
    arm is built differs. Measuring a file in the harness that does not load it would
    attribute the result to the wrong component, so the caller picks deliberately.

    Returns a `useful` verdict with per-check counts, the controls' outcomes, and the
    caveats that must accompany any use of the result.
    """
    if harness not in HARNESS_ARMS:
        raise ValueError(f"unknown harness: {harness} "
                         f"(expected one of {', '.join(sorted(HARNESS_ARMS))})")
    if not instruction_file.is_file():
        return {"ok": False, "reason": f"not a file: {instruction_file}"}
    content = instruction_file.read_text(errors="replace")
    if not content.strip():
        return {"ok": False, "reason": f"{instruction_file} is empty, so there is nothing "
                                       "for either arm to comply with"}

    return _run_compliance(instruction_file, content, HARNESS_ARMS[harness],
                           repeats=repeats, timeout=timeout, harness=harness)


def measure_skill_usefulness(skill_dir: Path, *, repeats: int = PROBE_REPEATS,
                             timeout: int = PROBE_TIMEOUT,
                             require_loaded: bool = True) -> dict:
    """Does this skill change verifiable properties of the agent's output?

    Same checks, same controls, same verdict as for an instruction file, because a rule a
    machine can verify does not care which kind of component stated it. Only the arm
    differs, and the arm is the one already verified for cost.

    Claude Code only: `--plugin-dir` is a Claude Code flag, so a Codex-side skill has no
    arm here and is not measured rather than measured through a substitute.
    """
    entry = skill_dir / "SKILL.md"
    if not entry.is_file():
        return {"ok": False, "reason": f"no SKILL.md under {skill_dir}"}
    content = entry.read_text(errors="replace")
    if not content.strip():
        return {"ok": False, "reason": f"{entry} is empty, so there is nothing for either "
                                       "arm to comply with"}
    # A skill loads on a description match, so a task unrelated to its description never
    # reaches it and the two arms then differ by nothing. That result reads as "this skill
    # does not change the output" when it means "this task did not reach it" — three
    # indistinguishable causes collapsed into one number.
    #
    # So the arms are checked for the skill's *presence* before anything is scored, by
    # asking the harness to list what it has. Word overlap between the description and the
    # tasks was tried first and abandoned: a description written in another language yields
    # almost no tokens, and the three incidental English fragments in a real one shared
    # `loop` with a task about an off-by-one in a page loop, so the check passed by
    # coincidence. Overlap also cannot be computed where the languages differ at all, and a
    # semantic judgement would need a model — the thing this module exists to avoid.
    loaded = _skill_presence(skill_dir, timeout=timeout)
    if require_loaded and loaded.get("checked") and not loaded.get("present_in_with_arm"):
        return {
            "ok": False,
            "blocker": "COMPONENT_NOT_LOADED_BY_ARM",
            "reason": f"the harness did not list {skill_dir.name} in the arm meant to carry "
                      f"it, so a probe would compare two arms that are the same; "
                      f"pass require_loaded=False to run it anyway",
            "skill_presence": loaded,
        }
    result = _run_compliance(entry, content, _skill_arms_for(skill_dir),
                             repeats=repeats, timeout=timeout, harness="claude_code")
    # Kept on a result that did run, so a reader weighing a `no separation` can see that
    # the arm really did carry the skill.
    result["skill_presence"] = loaded
    return result


# `--plugin-dir` is a Claude Code flag, so these are the scopes with an arm. A Codex-side
# skill is left unmeasured rather than measured through a substitute route.
SKILL_ARM_HARNESSES = {"claude_code", "shared"}


def _skill_presence(skill_dir: Path, *, timeout: int) -> dict:
    """Ask the harness whether the arm meant to carry this skill actually lists it.

    One cheap run against the `with` arm, reusing the listing prompt the cost measurement
    already validated with a nonce skill. This is a fact the harness reports about itself,
    which is why it replaced guessing at whether a task "should" match a description.

    A failure to check is not a failure to load: if the run dies, `checked` is false and the
    caller proceeds rather than refusing on the strength of a probe that never answered.
    """
    from .codex_probe import SKILL_LIST_PROMPT, _listed_skills, _plugin_with_one_skill

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        workdir = tmp / "work"
        workdir.mkdir()
        plugin = _plugin_with_one_skill(tmp, skill_dir)
        run = _probe(SKILL_LIST_PROMPT, workdir, None, timeout, bare=False,
                     extra_args=("--setting-sources", "", "--plugin-dir", str(plugin)))
    if not run.get("ok"):
        return {"checked": False, "error": run.get("error"),
                "cost_usd": run.get("cost_usd") or 0.0}
    listed = _listed_skills(run.get("text") or "")
    return {"checked": True, "present_in_with_arm": skill_dir.name in listed,
             "listed": sorted(listed)[:12], "cost_usd": run.get("cost_usd") or 0.0}


def _harness_health(measurements: list[dict]) -> dict:
    """Per-harness probe success, so a dead arm is a finding rather than a pile of unprovens.

    `usable` is false when a harness answered nothing. Every measurement under it is then
    about this tool's inability to run, not about the components, and a reader who misses
    that distinction concludes those components do nothing.
    """
    out: dict[str, dict] = {}
    for measurement in measurements:
        records = measurement.get("records") or []
        entry = out.setdefault(measurement.get("harness") or "unknown",
                               {"probes": 0, "valid": 0, "components": 0, "errors": {}})
        entry["components"] += 1
        entry["probes"] += len(records)
        entry["valid"] += sum(1 for r in records if r.get("valid"))
        for record in records:
            if not record.get("valid"):
                key = str(record.get("error"))[:80]
                entry["errors"][key] = entry["errors"].get(key, 0) + 1
    for entry in out.values():
        entry["usable"] = entry["valid"] > 0
        entry["errors"] = dict(sorted(entry["errors"].items(), key=lambda kv: -kv[1])[:3])
    return out


def _declared_skill_dirs(scan: dict) -> set[Path]:
    """Installed skill directories that the skill arm can load.

    A skill root is a direct child of a `skills/` directory and is not itself inside another
    skill. The second test is not redundant: a skill that ships fixture skills puts them
    under `evals/fixtures/skills/<name>/SKILL.md`, which satisfies the first test exactly.
    Those are data belonging to the outer skill, and probing them would invent installs that
    nothing loads. Nesting is detected by looking for an enclosing `SKILL.md` rather than by
    matching directory names, so a differently-arranged fixture tree is caught too.
    """
    out: set[Path] = set()
    for item in scan.get("files") or []:
        if item.get("kind") != "skills" or item.get("origin") != "declared":
            continue
        if item.get("harness") not in SKILL_ARM_HARNESSES:
            continue
        path = Path(item["path"])
        if path.name != "SKILL.md":
            continue
        if path.parent.parent.name != "skills":
            continue
        if any((ancestor / "SKILL.md").is_file() for ancestor in path.parent.parents):
            continue
        out.add(path.parent)
    return out


def _run_compliance(subject: Path, content: str, build, *, repeats: int, timeout: int,
                    harness: str) -> dict:
    """Interleave the two arms over the scenarios and decide the verdict.

    One code path for every kind of component, so the part that is easy to get wrong — the
    alternation, the negative control, the anti-recitation guard — cannot drift between
    them. Three separate failures in this project came from comparing two separately-taken
    measurements of a drifting quantity, which measures the drift.
    """
    stated, not_stated = scorable_checks(content)
    if not stated:
        # No quotable rule means no scorable claim. Probing anyway would spend money to
        # produce a difference that could not be attributed to anything the subject says.
        return {
            "ok": False,
            "blocker": "NO_STATED_RULES",
            "reason": f"{subject.name} states none of the rules this module can "
                      "verify, so its usefulness is not measurable by this method",
            "checks_available": [k["name"] for k in RULE_KINDS],
        }

    examples = _example_titles(content)
    spend: list[float] = []
    records: list[dict] = []

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        built = build(tmp, content, subject.name)
        if built.get("error"):
            # A baseline that already carries instructions cannot answer the question, and
            # running anyway would produce a difference of nearly zero that reads as
            # "this component does nothing".
            return {"ok": False, "useful": "unproven", "blocker": "CONTAMINATED_BASELINE",
                    "reason": built["error"], "harness": harness,
                    "file": str(subject)}

        tasks = [dict(s, checks=stated) for s in PROBE_SCENARIOS]
        tasks.append(dict(NEGATIVE_CONTROL, checks=None))

        consecutive, exhausted = 0, None
        for task in tasks:
            if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                break
            for index in range(max(1, repeats)):
                if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                    break
                # Arms alternate inside one window. Three separate failures in this project
                # came from comparing two separately-taken measurements of a drifting
                # quantity, which measures the drift.
                order = ("with", "without") if index % 2 else ("without", "with")
                for label in order:
                    result = _probe_with_retry(built["run"], task["prompt"], label,
                                               timeout, spend)
                    if not result["ok"]:
                        consecutive += 1
                        records.append({"task": task["id"], "arm": label, "valid": False,
                                        "error": result.get("error")})
                        if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                            # Stop rather than spend the rest of the batch into a condition
                            # that is not recovering. Abandoning it is reported, because a
                            # short record set must not be mistaken for a complete one.
                            exhausted = {
                                "after_probes": len(records),
                                "consecutive_failures": consecutive,
                                "last_error": result.get("error"),
                            }
                            break
                        continue
                    title, body = _title_and_body(result["text"])
                    if task["checks"] is None:
                        passed = {"arithmetic": bool(NEGATIVE_CONTROL["predicate"](title, body))}
                    else:
                        passed = {c["name"]: bool(c["predicate"](title, body))
                                  for c in task["checks"]}
                    consecutive = 0
                    records.append({
                        "task": task["id"], "arm": label, "valid": True, "checks": passed,
                        "title": title, "body": body,
                        "recited_example": any(ex in title for ex in examples),
                    })

    return _verdict(subject, content, stated, not_stated, examples, records, spend,
                    harness=harness, built=built, exhausted=exhausted)


def _tally(records: list[dict], task_id: str, check: str) -> dict:
    row = {}
    for label in ("with", "without"):
        valid = [r for r in records
                 if r["task"] == task_id and r["arm"] == label and r["valid"]]
        row[label] = {"passes": sum(1 for r in valid if r["checks"].get(check)),
                      "valid": len(valid)}
    return row


def _verdict(instruction_file: Path, content: str, stated: list[dict],
             not_stated: list[dict], examples: list[str], records: list[dict],
             spend: list[float], *, harness: str, built: dict,
             exhausted: dict | None = None) -> dict:
    failed = [r for r in records if not r["valid"]]
    caveats = [
        "The tasks were authored for this measurement. The result is effect on these "
        "tasks, not effect across the user's real sessions; external validity is weak.",
        "Only rules the file states are scored. Its other content is unmeasured, so "
        "this is a floor on the file's usefulness and not a valuation of it.",
    ]
    if built["channel_caveat"]:
        caveats.insert(0, built["channel_caveat"])
    common = {
        "file": str(instruction_file),
        "harness": harness,
        "bytes": instruction_file.stat().st_size,
        "characters": len(content),
        "arms": built["arms"],
        "rules_scored": [{"name": c["name"], "rule": c["rule"]} for c in stated],
        "rules_not_stated": [c["name"] for c in not_stated],
        "records": records,
        "probes_attempted": len(records),
        "probes_failed": len(failed),
        # A batch stopped early because the harness stopped answering. Without this a short
        # record set looks like a complete measurement that happened to be small.
        "abandoned": exhausted,
        # Codex reports tokens and no dollar figure. `0.0` would read as "this was free",
        # so the absence is reported as an absence.
        "cost_usd": round(sum(spend), 4) if built["reports_cost"] else None,
        "cost_note": None if built["reports_cost"] else
                     f"{harness} reports no dollar figure for a run, so none is given; "
                     f"{len(records)} runs were made",
        "caveats": caveats,
    }

    # No run survived, so there is nothing to judge — a different fact from "the control
    # did not hold", which says a comparison was made and was unsound. A real run reported
    # CONTROL_FAILED for 4 components whose every probe had died, and that reads as evidence
    # about those components when it is only evidence that the harness never answered.
    if not any(r["valid"] for r in records):
        reasons = sorted({str(r.get("error")) for r in records if not r["valid"]})
        return {
            "ok": False,
            "useful": "unproven",
            "blocker": "NO_VALID_RUNS",
            "reason": (f"every one of the {len(records)} probes failed, so neither arm "
                       f"produced an answer to compare"
                       + (f"; the batch was abandoned after "
                          f"{exhausted['consecutive_failures']} consecutive failures rather "
                          f"than spending the rest of it" if exhausted else "")
                       + f": {'; '.join(reasons)[:200]}"),
            **common,
        }

    control = _tally(records, NEGATIVE_CONTROL["id"], "arithmetic")
    control_held = (control["without"]["valid"] > 0
                    and control["without"]["passes"] == control["without"]["valid"]
                    and control["with"]["passes"] == control["with"]["valid"])
    common["negative_control"] = dict(control, held=control_held)
    if not control_held:
        # Without this gate a broken `without` arm reads as a component that everything
        # depends on, which is the most flattering possible error and the easiest to miss.
        return {
            "ok": False,
            "useful": "unproven",
            "blocker": "CONTROL_FAILED",
            "reason": "the negative control did not pass in both arms, so the arms are not "
                      "comparable and any difference between them is an artifact",
            **common,
        }

    recited = [r for r in records if r["valid"] and r.get("recited_example")]
    common["recited_example_runs"] = len(recited)
    if recited:
        return {
            "ok": False,
            "useful": "unproven",
            "blocker": "RECITED_EXAMPLE",
            "reason": f"{len(recited)} run(s) reproduced a worked example from the file "
                      "verbatim, so compliance was recalled rather than applied",
            **common,
        }

    separations = {}
    for check in stated:
        per_task = {}
        for scenario in PROBE_SCENARIOS:
            row = _tally(records, scenario["id"], check["name"])
            clean = (row["with"]["valid"] > 0 and row["without"]["valid"] > 0
                     and row["with"]["passes"] == row["with"]["valid"]
                     and row["without"]["passes"] == 0)
            per_task[scenario["id"]] = dict(row, clean_separation=clean)
        separations[check["name"]] = per_task
    common["separations"] = separations

    # A clean separation on both independent scenarios is the bar: the `with` arm complies
    # every time and the `without` arm never does. One scenario could be a quirk of its
    # wording, so a single-scenario result is reported as partial rather than proven.
    proven = [name for name, per_task in separations.items()
              if all(t["clean_separation"] for t in per_task.values())]
    directional = [
        name for name, per_task in separations.items()
        if name not in proven
        and any(t["with"]["passes"] > t["without"]["passes"] for t in per_task.values())
    ]
    common["rules_proven"] = proven
    common["rules_directional"] = directional

    if proven:
        return {
            "ok": True,
            "useful": "proven",
            "useful_evidence": (
                f"on {len(PROBE_SCENARIOS)} independent tasks, the arm carrying this file "
                f"satisfied {len(proven)} stated rule(s) in every valid run while the arm "
                f"without it satisfied them in none: {', '.join(proven)}. The negative "
                "control passed in both arms, so the arms are comparable."),
            "claim": ("Without this file the agent does not follow the conventions the file "
                      "states; with it, it does. This is not a claim that the conventions "
                      "are worth following — the file defines the standard it is measured "
                      "against, and the user wrote that standard."),
            **common,
        }
    if directional:
        return {
            "ok": True,
            "useful": "partial",
            "useful_evidence": (
                f"the arm carrying this file complied more often on {len(directional)} "
                f"stated rule(s) ({', '.join(directional)}), but not in every valid run on "
                "every task, so the effect is real in direction and not clean"),
            **common,
        }
    return {
        "ok": False,
        "useful": "unproven",
        "blocker": "NO_SEPARATION",
        "reason": "neither arm complied differently with the rules this file states, so no "
                  "behavioural effect was detected",
        **common,
    }


def measure_useful(root: str, *, repeats: int = PROBE_REPEATS, timeout: int = PROBE_TIMEOUT,
                   whole_environment: bool = False, exclusions: dict | None = None) -> dict:
    """Measure `useful` for every instruction file discovered for a root.

    Only instruction files are covered, and the reason is no longer the arms. Both of the
    other arms now exist and are verified:

        mcp:   --strict-mcp-config --mcp-config <one server>
        skill: --setting-sources "" (drops the user's skills) + --plugin-dir <one skill>

    The skill arm was recorded as impossible — "no CLI flag disables one skill" — which was
    true and beside the point, because the instruction route never worked by disabling
    anything either. It works by *addition*. Adding one skill back measures as a clean
    positive difference with spread 0, and a nonce skill confirms that exactly the injected
    one appears.

    What is missing for both is the other half: a task whose success a machine can check.
    An instruction file states its own rules, so compliance is checkable. A browser
    automation server or a "review my week" skill has no equivalent, and inventing a proxy
    for it is how the first two designs of this stage failed.

    For MCP servers an arm now *does* exist and is not implemented here. It was recorded as
    blocked on the belief that isolating one server also hides its credentials; that was
    tested and is false. `--strict-mcp-config --mcp-config <one server>` loads exactly the
    named server — verified by asking each arm to enumerate its own `mcp__` tools, which
    returned `NONE` for an empty config and 25 `mcp__search-server__*` tools and nothing else
    for a one-server config. Credentials are a per-server question rather than a property of
    the route: the server tested declares no `env` at all, so nothing was hidden.

    What is still missing for MCP is the other half of the measurement — a task whose
    success a machine can check, the way a stated rule does for instruction files.
    A browser-automation server has no equivalent of "the commit title ends with #id".
    """
    from .core import inspect as inspect_stack, is_excluded

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")
    started = time.perf_counter()
    scan = inspect_stack(str(base), whole_environment=whole_environment,
                         exclusions=exclusions)

    measurements, skipped = [], []
    for item in scan.get("files") or []:
        if item.get("origin") != "declared":
            continue
        # Each file is measured in the harness that loads it. Injecting a Codex instruction
        # file into a Claude probe would show that the *content* influences an agent, which
        # is not the claim — the claim has to be about the component in its own harness.
        harness = INSTRUCTION_HARNESS.get(item.get("kind"))
        if harness is None:
            continue
        path = Path(item["path"])
        if exclusions and is_excluded(exclusions, "instruction", path.name):
            skipped.append({"file": str(path), "reason": "excluded_by_request"})
            continue
        result = measure_instruction_usefulness(path, repeats=repeats, timeout=timeout,
                                                harness=harness)
        result["file"] = str(path)
        if result.get("useful"):
            measurements.append(result)
        else:
            skipped.append({"file": str(path), "reason": result.get("reason"),
                            "blocker": result.get("blocker")})

    # Skills, through the arm that adds exactly one back. Every install is measured, not one
    # of them: the same skill can be installed more than once with copies that differ, the
    # harness lists the name once, and which copy it loaded is not recorded. Measuring one
    # would produce a verdict about an unidentified copy; measuring all of them makes the
    # verdict copy-independent when they agree, and names the disagreement when they do not.
    for directory in sorted(_declared_skill_dirs(scan)):
        if exclusions and is_excluded(exclusions, "skill", directory.name):
            skipped.append({"file": str(directory), "reason": "excluded_by_request"})
            continue
        result = measure_skill_usefulness(directory, repeats=repeats, timeout=timeout)
        # The SKILL.md, not the directory: the cost artifact records the entry file and the
        # index that joins it to a component derives the directory from it. Recording the
        # directory here would key the two artifacts differently and this measurement would
        # join to nothing while reading as never attempted.
        result.setdefault("file", str(directory / "SKILL.md"))
        result["skill_dir"] = str(directory)
        if result.get("useful"):
            measurements.append(result)
        else:
            skipped.append({"file": str(directory), "reason": result.get("reason"),
                            "blocker": result.get("blocker")})

    return {
        "schema_version": "0.1",
        "mode": "measure_useful",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "useful_measurements": measurements,
        "skipped": skipped,
        # Whether a whole harness failed, stated once at the top. In a real run every
        # Claude-side probe died while every Codex-side one succeeded, and nothing said so:
        # the finding was four components reading `unproven`, which looks like a result
        # about them. A reader has to see that one arm never ran at all.
        "harness_health": _harness_health(measurements),
        "profiler_cost": {
            "stage": "measure_useful",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "cost_usd": round(sum(m.get("cost_usd") or 0.0 for m in measurements), 4),
            "probes": sum(m.get("probes_attempted", 0) for m in measurements),
        },
        "notes": [
            "A separation is Observed: the arms differ only by this component, and the "
            "checks are machine-verifiable, so no model grades the outcome.",
            "Checks are scored only when the component states the rule, detected from its "
            "own text. A check that cannot quote a rule was dropped after it separated "
            "cleanly on a convention the user never wrote.",
            "Covers instruction files, each measured in the harness that loads it, and "
            "skills, through the arm that adds exactly one back: `--setting-sources \"\"` "
            "clears the user's skills and `--plugin-dir` adds one copy, which also clears "
            "the user's instruction files from both arms. Every install of a skill is "
            "measured rather than one of them, because copies can differ and which one the "
            "harness loaded is not recorded.",
            "What is still outside this method is a component that states no rule a "
            "machine can check — an MCP server ships no prose to quote at all. The arm for "
            "one server exists and is verified (`--strict-mcp-config`); the missing half is "
            "a task whose success a machine can judge, which a stated rule supplies and a "
            "server's job does not.",
        ],
    }

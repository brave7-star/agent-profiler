from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

# .jsonl is deliberately absent: the only .jsonl files in these trees are session
# transcripts and history, which are runtime output, not declared configuration.
CONFIG_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".toml"}
INSTRUCTION_REVIEW_BYTES = 16 * 1024

# Dependency/build trees are never part of a declared agent stack, wherever they appear.
DEPENDENCY_DIRS = {
    "node_modules",
    "site-packages",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Subtrees of ~/.claude produced by running the agent rather than by configuring it.
CLAUDE_RUNTIME_DIRS = {
    "backups",
    "cache",
    "debug",
    "file-history",
    "ide",
    "paste-cache",
    "plans",
    "session-env",
    "sessions",
    "shell-snapshots",
    "statsig",
    "todos",
}
CLAUDE_RUNTIME_FILES = {".last-cleanup", ".last-update-result.json", ".ds_store", "history.jsonl"}

# Subtrees of ~/.codex with the same property.
CODEX_RUNTIME_DIRS = {
    "ambient-suggestions",
    "attachments",
    "backups",
    "browser",
    "cache",
    "computer-use",
    "dictation-history",
    "generated_images",
    "log",
    "logs",
    "sessions",
    "tmp",
    ".tmp",
    "visualizations",
    "volatile",
}
# Both harnesses ship bundled catalogs and binaries inside the user config tree. They are
# present, but nothing here proves the user declared or loaded them.
BUNDLED_DIRS = {"vendor_imports", ".plugin-appserver"}
# Path components marking harness-shipped entities rather than user-authored ones.
BUNDLED_MARKERS = {".system", ".curated", ".builtin"}
# Credential material: recorded as excluded, never enumerated or read.
CREDENTIAL_FILES = {"auth.json", "credentials.json", ".credentials.json", "token.json", "tokens.json"}

CODEX_RUNTIME_FILES = {
    ".codex-global-state.json",
    ".codex-global-state.json.bak",
    "chrome-native-hosts-v2.json",
    "external_agent_session_imports.json",
    "history.jsonl",
    ".ds_store",
}

# Claude Code keeps its MCP registry and per-project overrides in ~/.claude.json, outside
# the ~/.claude tree, so MCP servers are invisible to a directory walk alone.
CLAUDE_REGISTRY_FILE = ".claude.json"
# Key names whose value is credential-shaped. Matching keys are reported by name only.
CREDENTIAL_KEY_HINT = re.compile(r"(authorization|token|api[_-]?key|secret|password|credential)", re.I)
CREDENTIAL_VALUE_HINT = re.compile(r"^\s*(bearer|basic)\s+\S+", re.I)

INSTRUCTION_NAMES = {"claude.md": "claude_md", "claude.local.md": "claude_md", "agents.md": "agents_md"}
SETTINGS_NAMES = {"settings.json", "settings.local.json"}
ENTITY_DIRS = ("skills", "agents", "rules", "commands", "hooks")

# Per-harness config directories that can appear anywhere in a project tree.
AGENT_CONFIG_DIRS = {".claude": ("project_claude_dir", "claude_code"), ".codex": ("project_codex_dir", "codex")}
PROJECT_MAX_DEPTH = 6


def _meta(path: Path, *, base: Path, kind: str, scope: str, source: str, harness: str, origin: str) -> dict | None:
    """Return one file's observation, or None when the file cannot be stat'ed."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    try:
        relative = str(path.relative_to(base))
    except ValueError:
        relative = str(path)
    return {
        "path": str(path),
        "relative_path": relative,
        "bytes": size,
        "kind": kind,
        "scope": scope,
        "source": source,
        "harness": harness,
        "origin": origin,
    }


def _origin_for(relative: Path) -> str:
    """Harness-shipped entities are observable but are not the user's declared stack."""
    return "bundled" if BUNDLED_MARKERS & {p.lower() for p in relative.parts} else "declared"


def _kind_for(relative: Path, *, harness: str) -> str:
    """Classify by declaring directory first; file names are only a fallback.

    Directory wins because `commands/create-docker-mcp-tunnel.md` is a command, not an
    MCP declaration, and a substring match on the file name gets that backwards.
    """
    parts = [p.lower() for p in relative.parts]
    dirs, name = parts[:-1], parts[-1]

    for entity in ENTITY_DIRS:
        if entity in dirs:
            return entity
    if "memory" in dirs or (dirs and dirs[0] == "projects" and name == "memory.md"):
        return "memory"

    if name in INSTRUCTION_NAMES:
        return INSTRUCTION_NAMES[name]
    if name in {".mcp.json", "mcp.json"}:
        return "mcp"
    if name in SETTINGS_NAMES:
        return "settings"
    if name == "config.toml" and harness == "codex":
        return "settings"
    return "other_config"


def _prune_reason(parent: tuple[str, ...], name: str, *, tree: str) -> str | None:
    """Why a directory is not part of the declared stack, or None to walk into it.

    `parent` is the lowered path parts of the containing directory, relative to the tree root.
    """
    lowered = name.lower()
    if lowered in DEPENDENCY_DIRS:
        return "dependency_or_build_tree"
    if tree == "project":
        return None

    # Caches nest (~/.codex/plugins/cache), so match them at any depth, not only the root.
    if lowered in {"cache", "caches", ".cache"} or lowered.endswith(("-cache", "_cache")):
        return "agent_runtime_state"
    if lowered in BUNDLED_DIRS:
        return "bundled_catalog"
    if tree == "claude_home":
        if parent == ("plugins",) and lowered == "marketplaces":
            return "plugin_marketplace_catalog"
        if not parent and lowered in CLAUDE_RUNTIME_DIRS:
            return "agent_runtime_state"
    if tree == "codex_home" and not parent and lowered in CODEX_RUNTIME_DIRS:
        return "agent_runtime_state"
    return None


def _skip_file_reason(relative: Path, *, tree: str) -> str | None:
    """Why a suffix-matching file is still not a declaration."""
    parts = [p.lower() for p in relative.parts]
    name = parts[-1]

    if tree == "project":
        return None
    if name in CREDENTIAL_FILES:
        return "credential_material"
    # A *_cache.json beside real config is harness state; entity trees are left alone so a
    # skill may legitimately document caching.
    if name.endswith(".json") and "cache" in name and not any(e in parts[:-1] for e in ENTITY_DIRS):
        return "agent_runtime_state"

    if tree == "claude_home":
        if name in CLAUDE_RUNTIME_FILES:
            return "agent_runtime_state"
        # ~/.claude/projects holds per-project session transcripts. The memory/ subtree is
        # the exception: those files are loaded into context, so they are a real surface.
        if parts[0] == "projects" and "memory" not in parts[:-1] and name != "memory.md":
            return "session_transcript_or_project_state"
    if tree == "codex_home" and name in CODEX_RUNTIME_FILES:
        return "agent_runtime_state"
    return None


def _walk(base: Path, *, tree: str, excluded: list[dict], errors: list[dict], stats: Counter) -> Iterator[Path]:
    """Yield candidate declaration files under base, pruning and recording what it drops."""
    if not base.is_dir():
        return

    def on_error(exc: OSError) -> None:
        errors.append({"path": str(getattr(exc, "filename", base)), "error": exc.__class__.__name__})

    for dirpath, dirnames, filenames in os.walk(base, topdown=True, onerror=on_error, followlinks=False):
        here = Path(dirpath)
        parent = tuple(p.lower() for p in here.relative_to(base).parts)
        stats["dirs_visited"] += 1
        kept = []
        for d in sorted(dirnames):
            reason = _prune_reason(parent, d, tree=tree)
            if reason:
                excluded.append({"path": str(here / d), "entry": "directory", "reason": reason, "tree": tree})
            else:
                kept.append(d)
        dirnames[:] = kept

        for fname in sorted(filenames):
            stats["files_considered"] += 1
            p = here / fname
            if p.suffix.lower() not in CONFIG_SUFFIXES and fname != ".mcp.json":
                continue
            relative = p.relative_to(base)
            reason = _skip_file_reason(relative, tree=tree)
            if reason:
                excluded.append({"path": str(p), "entry": "file", "reason": reason, "tree": tree})
                continue
            yield p


def _load_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


# Minimal `[table.sub]` header scanner, used only when tomllib is unavailable (Python < 3.11).
TOML_TABLE_HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")
TOML_SIMPLE_KEY = re.compile(r"""^\s*([A-Za-z0-9_-]+)\s*=\s*["']?([^"'#]*)["']?""")


def _load_toml(path: Path) -> tuple[dict | None, str]:
    """Parse a TOML file, returning (data, parser) where parser names how it was read.

    Without tomllib the fallback recovers table names and simple string keys only, which is
    enough to name declared MCP servers. The parser used is reported so that a degraded
    read is never presented as a full one.
    """
    if tomllib is not None:
        try:
            with path.open("rb") as fh:
                return tomllib.load(fh), "tomllib"
        except (OSError, ValueError):
            return None, "tomllib_error"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "unreadable"

    data: dict = {}
    section: dict | None = None
    for line in text.splitlines():
        header = TOML_TABLE_HEADER.match(line)
        if header:
            # Dotted names create nested tables; quotes around a segment are stripped.
            node = data
            for part in [seg.strip().strip('"').strip("'") for seg in header.group(1).split(".")]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            section = node
            continue
        pair = TOML_SIMPLE_KEY.match(line)
        if pair and section is not None:
            section.setdefault(pair.group(1), pair.group(2).strip())
    return data, "header_scan_fallback"


def _mcp_names(block: object) -> dict[str, str]:
    """Extract declared MCP server names and transport only. Never URLs or credentials."""
    if not isinstance(block, dict):
        return {}
    names: dict[str, str] = {}
    for name, cfg in block.items():
        transport = "unknown"
        if isinstance(cfg, dict):
            if isinstance(cfg.get("type"), str):
                transport = cfg["type"]
            elif cfg.get("command"):
                transport = "stdio"
            elif cfg.get("url"):
                transport = "http"
        names[str(name)] = transport
    return names


def _credential_keys(obj: object, prefix: str = "") -> list[str]:
    """Return dotted key paths holding credential-shaped values. Values are never returned."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            where = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, str) and v.strip():
                if CREDENTIAL_KEY_HINT.search(str(k)) or CREDENTIAL_VALUE_HINT.match(v):
                    hits.append(where)
            else:
                hits.extend(_credential_keys(v, where))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_credential_keys(v, f"{prefix}[{i}]"))
    return hits


def _git_tracked(path: Path) -> bool | None:
    """Whether git tracks this file. None when the question cannot be answered."""
    try:
        done = subprocess.run(
            ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode == 0


def _encode_project_key(path: Path) -> str:
    """Reproduce Claude Code's ~/.claude/projects directory naming for a project path."""
    return str(path).replace("/", "-").replace(".", "-")


def _settings_features(path: Path) -> set[str]:
    """Return only structural feature names; never return configuration values."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(obj, dict):
        return set()

    features = set()
    lowered = {str(k).lower() for k in obj}
    if "hooks" in lowered:
        features.add("hooks")
    if "mcpservers" in lowered or "mcp_servers" in lowered or "mcp" in lowered:
        features.add("mcp")
    return features


def _discover_project(base: Path, excluded: list[dict], errors: list[dict], stats: Counter) -> list[dict]:
    """Walk the project tree for instruction files and nested agent config directories.

    Claude Code and Codex both read instruction files from subdirectories, so a root-only
    scan misses them in a monorepo. The walk is bounded by dependency pruning and a depth
    cap so that scanning a large repository stays cheap.
    """
    found: list[dict] = []

    def on_error(exc: OSError) -> None:
        errors.append({"path": str(getattr(exc, "filename", base)), "error": exc.__class__.__name__})

    for dirpath, dirnames, filenames in os.walk(base, topdown=True, onerror=on_error, followlinks=False):
        here = Path(dirpath)
        depth = len(here.relative_to(base).parts)
        stats["dirs_visited"] += 1

        kept = []
        for d in sorted(dirnames):
            lowered = d.lower()
            if lowered in DEPENDENCY_DIRS:
                excluded.append(
                    {"path": str(here / d), "entry": "directory", "reason": "dependency_or_build_tree", "tree": "project"}
                )
                continue
            if lowered in AGENT_CONFIG_DIRS:
                source, harness = AGENT_CONFIG_DIRS[lowered]
                sub = here / d
                if _is_user_tree(sub):
                    # The user's own config tree, reached by walking a root that contains
                    # it — scanning `--all` from the home directory always does.
                    #
                    # It must not be scanned here. `_prune_reason` returns None for every
                    # rule as soon as the tree is `project`, which is right for a real
                    # repository (a project's `.claude/skills` is not a cache) and wrong
                    # for this one: the user trees have their own curated scan under
                    # `user` scope, and re-entering them from here applies none of it.
                    # Measured on a real environment, it added thousands of entries from
                    # `~/.codex/.tmp`, plugin caches, `vendor_imports`, browser sessions
                    # and an app bundle, every one marked `origin: declared` — and
                    # `codex_skill_paths` went to 550 components on a machine with about
                    # ten. The inventory said 593 skill entities.
                    #
                    # Skipped by identity rather than by name, so a directory the harness
                    # adds inside its own tree tomorrow cannot reopen this. `~/.codex/.tmp`
                    # is dated 2026-11-19: that is how this arrived.
                    excluded.append(
                        {"path": str(sub), "entry": "directory",
                         "reason": "user_config_tree_scanned_as_user_scope", "tree": "project"}
                    )
                    continue
                for p in _walk(sub, tree="project", excluded=excluded, errors=errors, stats=stats):
                    item = _meta(
                        p,
                        base=base,
                        kind=_kind_for(p.relative_to(sub), harness=harness),
                        scope="project",
                        source=source,
                        harness=harness,
                        origin=_origin_for(p.relative_to(sub)),
                    )
                    if item:
                        found.append(item)
                continue
            if depth >= PROJECT_MAX_DEPTH:
                excluded.append(
                    {"path": str(here / d), "entry": "directory", "reason": "beyond_project_depth_limit", "tree": "project"}
                )
                continue
            kept.append(d)
        dirnames[:] = kept

        source = "project_root" if depth == 0 else "project_nested"
        for fname in sorted(filenames):
            stats["files_considered"] += 1
            kind = INSTRUCTION_NAMES.get(fname.lower())
            if kind is None and fname == ".mcp.json":
                kind = "mcp"
            if kind is None:
                continue
            harness = "codex" if kind == "agents_md" else "claude_code"
            item = _meta(
                here / fname, base=base, kind=kind, scope="project", source=source, harness=harness, origin="declared"
            )
            if item:
                found.append(item)
    return found


def _user_home() -> Path:
    override = os.environ.get("AGENT_PROFILER_HOME")
    return Path(override).expanduser().resolve() if override else Path.home()


def _user_tree(var: str, default_name: str) -> Path:
    override = os.environ.get(var)
    if override:
        return Path(override).expanduser().resolve()
    return (_user_home() / default_name).resolve()


# The roots `_discover_user` scans, and the ones `_discover_project` must leave to it.
USER_TREES = (
    ("AGENT_PROFILER_CLAUDE_HOME", ".claude"),
    ("AGENT_PROFILER_CODEX_HOME", ".codex"),
    ("AGENT_PROFILER_AGENTS_HOME", ".agents"),
)


def _is_user_tree(path: Path) -> bool:
    """Whether this directory is one of the user config trees.

    Resolved on both sides, because the two walks reach the same directory by different
    routes and a symlinked home would otherwise compare unequal.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == _user_tree(var, name) for var, name in USER_TREES)


def _discover_user(excluded: list[dict], errors: list[dict], stats: Counter) -> list[dict]:
    found: list[dict] = []

    # Do not scan arbitrary home-directory content; only the known agent config trees.
    claude_home = _user_tree("AGENT_PROFILER_CLAUDE_HOME", ".claude")
    codex_home = _user_tree("AGENT_PROFILER_CODEX_HOME", ".codex")
    # `~/.agents` is a shared, harness-independent skills root. It was missing here, and
    # the omission was a real false negative: four skills live in it on this machine and
    # Codex demonstrably loads them — a session's `state.host_skills` lists each
    # one's path with `shown: true`, and names this directory among its skill `roots`.
    # The scan reported "installed: 15/15 proven" while never looking at it, which is the
    # worst kind of miss: a confident claim about completeness that was not complete.
    agents_home = _user_tree("AGENT_PROFILER_AGENTS_HOME", ".agents")

    agents_md = _user_home() / "AGENTS.md"
    if agents_md.is_file():
        item = _meta(
            agents_md,
            base=agents_md.parent,
            kind="agents_md",
            scope="user",
            source="user_home",
            harness="codex",
            origin="declared",
        )
        if item:
            found.append(item)

    for root, tree, source, harness in (
        (claude_home, "claude_home", "user_claude_dir", "claude_code"),
        (codex_home, "codex_home", "user_codex_dir", "codex"),
        # Harness-independent, so the declaring harness is not knowable from the path.
        # `shared` says that plainly rather than guessing one and attributing use to it.
        (agents_home, "agents_home", "user_agents_dir", "shared"),
    ):
        for p in _walk(root, tree=tree, excluded=excluded, errors=errors, stats=stats):
            item = _meta(
                p,
                base=root,
                kind=_kind_for(p.relative_to(root), harness=harness),
                scope="user",
                source=source,
                harness=harness,
                origin=_origin_for(p.relative_to(root)),
            )
            if item:
                found.append(item)
    return found


def _discover_mcp_servers(
    base: Path, candidates: list[dict], stats: Counter, degraded: list[dict],
    *, whole_environment: bool = False
) -> tuple[list[dict], list[dict]]:
    """Resolve declared MCP servers to entities, and note the registry file if present.

    Three sources declare servers: project .mcp.json, Claude Code's ~/.claude.json
    (user-level plus per-project overrides) and Codex config.toml.

    Per-project overrides are normally filtered to the scanned root. With
    `whole_environment` they are all reported, each carrying the project it belongs to,
    because the question being answered is about the environment rather than one project.
    """
    servers: list[dict] = []
    extra_files: list[dict] = []

    def add(names: dict[str, str], *, scope: str, source: str, path: Path,
            declared_for: str | None = None) -> None:
        for name, transport in sorted(names.items()):
            entry = {"name": name, "scope": scope, "source": source, "transport": transport,
                     "declared_in": str(path)}
            if declared_for is not None:
                entry["declared_for_project"] = declared_for
            servers.append(entry)

    for item in candidates:
        path = Path(item["path"])
        if item["kind"] == "mcp":
            stats["files_read"] += 1
            obj = _load_json(path) or {}
            add(_mcp_names(obj.get("mcpServers")), scope=item["scope"], source=item["source"], path=path)
        elif item["kind"] == "settings" and path.suffix == ".toml":
            stats["files_read"] += 1
            obj, parser = _load_toml(path)
            names = _mcp_names((obj or {}).get("mcp_servers"))
            if parser != "tomllib" and names:
                degraded.append({"path": str(path), "parser": parser, "impact": "mcp_server_options_not_parsed"})
            add(names, scope=item["scope"], source=item["source"], path=path)
        elif item["kind"] == "settings" and path.suffix == ".json":
            stats["files_read"] += 1
            obj = _load_json(path) or {}
            add(_mcp_names(obj.get("mcpServers")), scope=item["scope"], source=item["source"], path=path)

    registry = _user_home() / CLAUDE_REGISTRY_FILE
    if registry.is_file():
        item = _meta(
            registry,
            base=registry.parent,
            kind="registry",
            scope="user",
            source="user_claude_json",
            harness="claude_code",
            origin="declared",
        )
        if item:
            extra_files.append(item)
        stats["files_read"] += 1
        obj = _load_json(registry) or {}
        add(_mcp_names(obj.get("mcpServers")), scope="user", source="user_claude_json", path=registry)
        projects = obj.get("projects")
        if isinstance(projects, dict):
            for proj, cfg in sorted(projects.items()):
                if not isinstance(cfg, dict):
                    continue
                mine = Path(str(proj)).resolve() == base
                if mine:
                    add(_mcp_names(cfg.get("mcpServers")), scope="project", source="user_claude_json", path=registry)
                elif whole_environment:
                    add(_mcp_names(cfg.get("mcpServers")), scope="other_project",
                        source="user_claude_json", path=registry, declared_for=str(proj))
    return servers, extra_files


def _credential_findings(candidates: list[dict], stats: Counter) -> list[dict]:
    """Flag agent config holding credential-shaped values, and whether git tracks it."""
    findings: list[dict] = []
    for item in candidates:
        if item["kind"] not in {"mcp", "settings", "registry"}:
            continue
        path = Path(item["path"])
        stats["files_read"] += 1
        obj = _load_json(path) if path.suffix == ".json" else _load_toml(path)[0]
        keys = _credential_keys(obj) if obj else []
        if not keys:
            continue
        tracked = _git_tracked(path)
        findings.append(
            {
                "type": "credential_in_agent_config",
                "path": item["path"],
                "scope": item["scope"],
                "keys": sorted(keys),
                "git_tracked": tracked,
                "evidence": f"{len(keys)} credential-shaped value(s) at {', '.join(sorted(keys))}; git_tracked={tracked}",
                "reason": "Agent config holding a literal credential is a disclosure risk, especially when tracked by git.",
                "judgment": "review_only",
            }
        )
    return findings


def _duplicate_mcp_findings(servers: list[dict]) -> list[dict]:
    """Flag servers declared in more than one place; the effective config is then ambiguous."""
    by_name: dict[str, list[dict]] = {}
    for server in servers:
        by_name.setdefault(server["name"], []).append(server)

    findings = []
    for name, entries in sorted(by_name.items()):
        sources = sorted({e["declared_in"] for e in entries})
        if len(sources) < 2:
            continue
        findings.append(
            {
                "type": "mcp_declared_in_multiple_places",
                "name": name,
                "scopes": sorted({e["scope"] for e in entries}),
                "declared_in": sources,
                "evidence": f"MCP server '{name}' is declared in {len(sources)} files",
                "reason": "Duplicate declarations make the effective configuration ambiguous and can load the server more than once.",
                "judgment": "review_only",
            }
        )
    return findings


def _catalog_summary(excluded: list[dict]) -> dict:
    """Summarise the marketplace catalog instead of counting it as the user's stack.

    A cloned marketplace lists plugins that are *available* to install. Nothing here
    proves any of them is enabled, so they are reported apart from declared config.
    """
    marketplaces: dict[str, dict] = {}
    for entry in (e for e in excluded if e["reason"] == "plugin_marketplace_catalog"):
        root = Path(entry["path"])
        try:
            clones = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for clone in clones:
            available = 0
            for bucket in ("plugins", "external_plugins"):
                try:
                    available += sum(1 for p in (clone / bucket).iterdir() if p.is_dir())
                except OSError:
                    pass
            marketplaces[clone.name] = {"path": str(clone), "plugins_available": available}
    return {
        "marketplaces": dict(sorted(marketplaces.items())),
        "judgment": "available_not_proven_active",
        "note": "Catalog contents are not enumerated as declared config; nothing here proves a plugin is enabled.",
    }


def _components(items: list[dict]) -> dict:
    """Roll declared files up to entities: a skill is a directory, not each of its files.

    Only origin=="declared" is counted, so harness-bundled entities never inflate the user's stack.
    """
    entities: dict[str, set[str]] = {}
    files: Counter = Counter()
    byte_total: Counter = Counter()

    for item in items:
        kind = item["kind"]
        if kind not in ENTITY_DIRS or item["origin"] != "declared":
            continue
        rel = Path(item["relative_path"])
        parts = list(rel.parts)
        lowered = [p.lower() for p in parts]
        idx = lowered.index(kind)
        tail = parts[idx + 1 :]
        if not tail:
            continue
        # skills/<name>/SKILL.md -> one entity per directory; commands/<name>.md -> per file.
        entity = tail[0] if len(tail) > 1 else Path(tail[0]).stem
        # Harness is part of the key: the same skill name can exist for Claude Code and Codex.
        entities.setdefault(kind, set()).add(f"{item['scope']}/{item['harness']}/{entity}")
        files[kind] += 1
        byte_total[kind] += item["bytes"]

    return {
        kind: {
            "entities": len(names),
            "files": files[kind],
            "bytes": byte_total[kind],
            "names": sorted(names),
        }
        for kind, names in sorted(entities.items())
    }


def _dedupe(items: list[dict]) -> list[dict]:
    by_path: dict[str, dict] = {}
    for item in items:
        by_path[item["path"]] = item
    return list(by_path.values())


EXCLUSIONS_FILENAME = "agent-profiler-exclude.json"
# `instruction` joined this list when instruction files stopped being read-only subjects:
# the `useful` probe sends a verbatim copy of a CLAUDE.md or AGENTS.md to the API, so a user
# who wants a particular instruction file left alone needs a way to say so.
EXCLUDABLE_KINDS = ("mcp", "skill", "agent", "command", "instruction")


def load_exclusions(path: str | None, root: Path | None = None) -> dict:
    """What the user has asked this tool to leave alone.

    An exclusion is *not* a filter. An excluded entity is still inventoried and still
    counted, carrying `excluded_by_request`, because a silently shorter list is
    indistinguishable from a smaller environment — and a profiler whose omissions are
    invisible is the failure this project exists to avoid.

    What an exclusion does guarantee is that no stage reaches out to the thing: in
    particular `evaluate` never launches a probe configured with an excluded server, so
    nothing is sent to its endpoint.
    """
    chosen = Path(path).expanduser() if path else None
    if chosen is None and root is not None:
        default = root / EXCLUSIONS_FILENAME
        chosen = default if default.is_file() else None
    if chosen is None:
        return {"names": {}, "source": None, "reason": None}
    # A path the caller named explicitly must exist: falling back to "nothing excluded"
    # would start sending traffic the user asked this tool to withhold.
    if not chosen.is_file():
        raise FileNotFoundError(f"exclusions file not found: {chosen}")

    obj = _load_json(chosen)
    if not isinstance(obj, dict):
        raise ValueError(f"{chosen}: expected a JSON object")
    names: dict[str, set] = {}
    for kind, value in obj.items():
        if kind == "reason":
            continue
        if kind not in EXCLUDABLE_KINDS:
            raise ValueError(
                f"{chosen}: unknown exclusion kind {kind!r}; expected one of "
                f"{', '.join(EXCLUDABLE_KINDS)}"
            )
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise ValueError(f"{chosen}: {kind} must be a list of non-empty names")
        names[kind] = set(value)
    reason = obj.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{chosen}: reason must be a string")
    return {"names": names, "source": str(chosen), "reason": reason}


def is_excluded(exclusions: dict | None, kind: str, name: str) -> bool:
    return bool(exclusions) and name in (exclusions.get("names") or {}).get(kind, ())


def inspect(root: str, *, whole_environment: bool = False, exclusions: dict | None = None) -> dict:
    """Inventory a stack.

    With `whole_environment`, the scan stops being about one project: every per-project
    MCP declaration in `~/.claude.json` is reported rather than only the scanned root's.
    That closes a real false negative — a server declared for the home directory is live
    in sessions and was invisible to a root-scoped scan.
    """
    # An empty root would silently resolve to the working directory; make the caller say so.
    if not root or not root.strip():
        raise ValueError("inspect root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"inspect root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"inspect root is not a directory: {base}")

    excluded: list[dict] = []
    errors: list[dict] = []
    stats: Counter = Counter()
    started = time.perf_counter()
    candidates = _dedupe(_discover_project(base, excluded, errors, stats) + _discover_user(excluded, errors, stats))
    degraded: list[dict] = []
    mcp_servers, registry_files = _discover_mcp_servers(
        base, candidates, stats, degraded, whole_environment=whole_environment
    )
    candidates = _dedupe(candidates + registry_files)

    # Marked, never dropped: the inventory stays complete and the stages that would reach
    # out to a server consult the flag instead of receiving a shorter list.
    for server in mcp_servers:
        if is_excluded(exclusions, "mcp", server["name"]):
            server["excluded_by_request"] = True
            server["exclusion_reason"] = (exclusions or {}).get("reason") or "excluded by request"

    # Memory is stored per project, so only the scanned project's memory is loaded for it.
    project_key = _encode_project_key(base)
    for item in candidates:
        if item["kind"] == "memory":
            parts = Path(item["relative_path"]).parts
            memory_project = parts[1] if len(parts) > 1 and parts[0] == "projects" else ""
            item["memory_project"] = memory_project
            item["applies_to_root"] = memory_project == project_key

    candidates.sort(key=lambda x: (x["scope"], x["relative_path"], x["path"]))

    by_kind = Counter(item["kind"] for item in candidates)
    by_scope = Counter(item["scope"] for item in candidates)
    by_harness = Counter(item["harness"] for item in candidates)
    by_origin = Counter(item["origin"] for item in candidates)
    bytes_by_scope: Counter = Counter()
    for item in candidates:
        bytes_by_scope[item["scope"]] += item["bytes"]

    # A settings file can declare hooks/MCP without a dedicated file path. Count the declaration
    # as an observed structural signal, but do not expose its contents.
    declared_features = Counter()
    for item in candidates:
        if item["kind"] == "settings":
            stats["files_read"] += 1
            for feature in _settings_features(Path(item["path"])):
                declared_features[feature] += 1

    review_candidates = _credential_findings(candidates, stats) + _duplicate_mcp_findings(mcp_servers)
    for item in candidates:
        if item["kind"] in {"claude_md", "agents_md"} and item["bytes"] >= INSTRUCTION_REVIEW_BYTES:
            review_candidates.append(
                {
                    "type": "large_instruction_file",
                    "path": item["path"],
                    "scope": item["scope"],
                    "bytes": item["bytes"],
                    "evidence": f"{Path(item['path']).name} is {item['bytes']} bytes",
                    "reason": "Large instruction files can materially affect context size and should be reviewed.",
                    "judgment": "review_only",
                }
            )

    observed = {
        "files_found": len(candidates),
        "bytes_found": sum(x["bytes"] for x in candidates),
        "bytes_by_scope": dict(sorted(bytes_by_scope.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_harness": dict(sorted(by_harness.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "components": _components(candidates),
        "mcp_servers_declared": len(mcp_servers),
        "mcp_servers_excluded_by_request": sum(
            1 for s in mcp_servers if s.get("excluded_by_request")
        ),
        "memory_files_for_root": sum(1 for x in candidates if x.get("applies_to_root")),
        "files_excluded": len(excluded),
    }
    if declared_features:
        observed["settings_declared_features"] = dict(sorted(declared_features.items()))

    exclusion_reasons = Counter(e["reason"] for e in excluded)

    # The profiler is part of the stack it audits, so it reports its own cost (0 LLM tokens here).
    profiler_cost = {
        "stage": "static_scan",
        "dirs_visited": stats["dirs_visited"],
        "files_considered": stats["files_considered"],
        "files_read_for_content": stats["files_read"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
    }

    return {
        "schema_version": "0.2",
        "mode": "inspect",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "exclusions": {
            "source": (exclusions or {}).get("source"),
            "reason": (exclusions or {}).get("reason"),
            "names": {k: sorted(v) for k, v in sorted(((exclusions or {}).get("names") or {}).items())},
        },
        "observed": observed,
        "files": candidates,
        "mcp_servers": mcp_servers,
        "excluded_paths": sorted(excluded, key=lambda e: e["path"]),
        "excluded_reasons": dict(sorted(exclusion_reasons.items())),
        "catalog": _catalog_summary(excluded),
        "profiler_cost": profiler_cost,
        "degraded_reads": degraded,
        "errors": errors,
        "review_candidates": review_candidates,
        "notes": [
            "File size and presence are observations, not proof of waste.",
            "Discovery covers Claude Code (project root/.claude, user ~/.claude) and Codex "
            "(AGENTS.md, project .codex, user ~/.codex); dependency and build trees are not scanned.",
            "Runtime state (session transcripts, caches, history) is excluded from the declared "
            "stack and listed in excluded_paths; ~/.claude/projects/*/memory is kept because it is "
            "loaded into context.",
            "Marketplace plugins are reported in 'catalog' as available, not as the active stack.",
            "MCP servers are resolved to names and transports from project .mcp.json, ~/.claude.json "
            "and Codex config.toml; URLs, headers and credential values are never read out.",
            "~/.claude.json mixes declarations with runtime history, so its byte size is not an "
            "instruction-size signal.",
            "Memory is per project: 'applies_to_root' marks the memory that loads for this root.",
            "Review candidates flag evidence for human review and are not automatic waste judgments.",
            "An excluded entity is still inventoried and marked `excluded_by_request`. "
            "Exclusion stops stages from reaching out to it; it does not hide it, because an "
            "invisible omission is indistinguishable from a smaller environment.",
        ],
    }


def trace(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    rows = []
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    else:
        obj = json.loads(text)
        rows = obj if isinstance(obj, list) else [obj]
    token_keys = Counter()
    tool_names = Counter()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                kl = k.lower()
                if "token" in kl and isinstance(x, (int, float)):
                    token_keys[k] += x
                if kl in {"tool", "tool_name", "name"} and isinstance(x, str) and "tool" in str(v).lower():
                    tool_names[x] += 1
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for r in rows:
        walk(r)
    return {
        "schema_version": "0.1",
        "mode": "trace",
        "source": str(p),
        "read_only": True,
        "observed": {"records": len(rows), "token_fields": dict(token_keys), "tool_mentions": dict(tool_names)},
        "review_candidates": [],
        "notes": ["Trace parsing is schema-agnostic in v0.1-dev; field-specific adapters will be added from real traces."],
    }


def compare(a: dict, b: dict) -> dict:
    def nums(prefix, v, out):
        if isinstance(v, dict):
            for k, x in v.items():
                nums(prefix + [k], x, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[".".join(prefix)] = v

    na = {}
    nb = {}
    nums([], a.get("observed", {}), na)
    nums([], b.get("observed", {}), nb)
    keys = sorted(set(na) | set(nb))
    diffs = []
    for k in keys:
        av = na.get(k)
        bv = nb.get(k)
        if av is not None and bv is not None:
            diffs.append({"metric": k, "baseline": av, "candidate": bv, "delta": bv - av})
    return {
        "schema_version": "0.1",
        "mode": "compare",
        "read_only": True,
        "observed": {"differences": diffs},
        "review_candidates": [],
        "notes": ["A delta is not automatically an improvement or regression."],
    }


# --------------------------------------------------------------------------------------
# Reusing per-file work across runs
# --------------------------------------------------------------------------------------


CACHE_SCHEMA = "1"


def _cache_path() -> Path:
    override = os.environ.get("AGENT_PROFILER_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "agent-profiler" / "corpus-cache.json"


def file_digest(path: Path) -> str | None:
    """sha256 of a file's bytes, or None if it cannot be read.

    sha256 rather than blake2b because this machine has hardware acceleration for it and
    is measurably faster on a multi-gigabyte corpus, by better than a factor of two.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


_CODE_FINGERPRINT: str | None = None


def code_fingerprint() -> str:
    """A digest of every module in this package.

    Hashed rather than hand-versioned: a constant someone has to remember to bump is a
    constant that eventually is not bumped, and the failure mode is a stale number
    presented as a measurement. Hashing the whole package rather than a per-stage list of
    dependencies removes a second judgment that would go stale the same way — a cache
    invalidated too often costs time, one invalidated too rarely reports fiction.
    """
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is None:
        digest = hashlib.sha256()
        try:
            for source in sorted(Path(__file__).resolve().parent.glob("*.py")):
                digest.update(source.read_bytes())
        except OSError:
            # Unreadable source means the fingerprint cannot be trusted, so make it unique
            # to this process: nothing matches, and nothing stale is served.
            return f"unverifiable-{os.getpid()}-{time.time()}"
        _CODE_FINGERPRINT = digest.hexdigest()[:16]
    return _CODE_FINGERPRINT


class FileCache:
    """Per-file results, reused only when the file and the code are both unchanged.

    Every stage here walks a corpus of transcripts that only grows at the tail, so the
    same files are re-read on every run. Measurement says the reading itself is near a
    floor — decompressing the Codex side is 6.1 of its 14 seconds, parallel reads were
    *slower*, and skipping a line costs more than parsing it — so the work is not made
    faster, it is not repeated.

    The key is deliberately paranoid, because a cache that serves a stale answer is worse
    than a slow one, and this project lost real time to exactly that: Python's own
    bytecode cache validates a source file on its **size and whole-second mtime**, so a
    same-size edit made within that second is invisible to it. Nothing here is trusted
    except content — the file's bytes, hashed, plus a digest of every module in the
    package, plus the path, since two paths can hold identical bytes and a result can name
    its source.

    What a stage caches must also be independent of anything outside the file. The Codex
    usage rollup caches the paths a session read, raw, and applies the machine's inventory
    when they are merged: caching the attributed answer would mean installing a skill did
    not change the result.
    """

    def __init__(self, kind: str, *, enabled: bool = True) -> None:
        self.kind = kind
        self.enabled = enabled
        self.reused = 0
        self.computed = 0
        self._dirty = False
        self._entries = _load_cache() if enabled else {}
        self._fingerprint = code_fingerprint() if enabled else ""

    def _key(self, path: Path) -> str | None:
        if not self.enabled:
            return None
        digest = file_digest(path)
        return None if digest is None else f"{self.kind}:{self._fingerprint}:{digest}:{path}"

    def get_or_compute(self, path: Path, compute):
        """`compute()` runs only on a miss, and its exceptions reach the caller unchanged."""
        key = self._key(path)
        if key is not None:
            cached = self._entries.get(key)
            if isinstance(cached, dict):
                self.reused += 1
                return cached
        value = compute()
        self.computed += 1
        if key is not None and isinstance(value, dict):
            self._entries[key] = value
            self._dirty = True
        return value

    def flush(self) -> None:
        if self._dirty:
            _save_cache(self._entries)
            self._dirty = False

    def report(self) -> dict:
        """Stated in every stage's output: a reused figure must never be
        indistinguishable from a freshly measured one."""
        return {"enabled": self.enabled, "reused": self.reused, "computed": self.computed}


def _load_cache() -> dict:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("cache_schema") != CACHE_SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_cache(entries: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"cache_schema": CACHE_SCHEMA, "entries": entries},
                       ensure_ascii=False, indent=0),
            encoding="utf-8")
    except OSError:
        pass

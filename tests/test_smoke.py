"""Regression suite for the static scanner.

Every case below corresponds to a false positive, false negative or misclassification
observed in a real dogfood run; for the evidence trail.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler.cli import main as cli_main
from agent_profiler.core import (
    EXCLUSIONS_FILENAME,
    compare,
    inspect,
    load_exclusions,
    trace,
)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_project(root: Path) -> None:
    _write(root / "CLAUDE.md", "x" * (17 * 1024))
    _write(root / "AGENTS.md", "x" * (18 * 1024))
    _write(
        root / ".mcp.json",
        json.dumps(
            {"mcpServers": {"demo-server": {"type": "http", "url": "https://example.invalid/mcp",
                                         "headers": {"Authorization": "Bearer FAKE-NOT-A-REAL-TOKEN"}}}}
        ),
    )
    _write(root / ".claude/settings.json", '{"hooks": {}, "mcpServers": {}}')
    _write(root / ".claude/skills/demo/SKILL.md", "demo")
    _write(root / ".claude/skills/demo/references/extra.md", "reference")
    _write(root / ".claude/agents/reviewer.md", "agent")
    _write(root / ".claude/rules/style.md", "rule")
    _write(root / ".claude/commands/create-docker-mcp-tunnel.md", "command")
    _write(root / ".claude/hooks/pre-commit.json", "{}")
    _write(root / ".codex/config.toml", "model = 'gpt'\n[mcp_servers.local]\ncommand = 'npx'\n")
    # Instruction files and agent dirs also live in subdirectories of a monorepo.
    _write(root / "packages/api/CLAUDE.md", "nested instructions")
    _write(root / "packages/api/.claude/skills/nested-skill/SKILL.md", "nested skill")
    # Dependency trees, at the project root and nested inside a skill.
    _write(root / "node_modules/pkg/skills/fake/SKILL.md", "must not be scanned")
    _write(root / ".claude/skills/demo/node_modules/dep/SKILL.md", "must not be scanned")


def _build_home(home: Path, root: Path) -> None:
    claude = home / ".claude"
    codex = home / ".codex"

    # Claude Code's MCP registry lives here, outside the ~/.claude tree.
    _write(
        home / ".claude.json",
        json.dumps(
            {
                "mcpServers": {"registry-server": {"command": "x"}},
                "projects": {str(root): {"mcpServers": {"search-server": {"command": "npx"}}}},
            }
        ),
    )

    _write(home / "AGENTS.md", "user codex instructions")
    _write(claude / "CLAUDE.md", "user claude instructions")
    _write(claude / "settings.json", '{"hooks": {}}')
    _write(claude / "skills/user-skill/SKILL.md", "user skill")
    # The same skill name under two harnesses must count as two entities.
    _write(claude / "skills/shared-skill/SKILL.md", "shared, claude copy")
    _write(codex / "skills/shared-skill/SKILL.md", "shared, codex copy")
    _write(claude / "plugins/known_marketplaces.json", "{}")
    # demo-server is declared at both project and user level in the real environment.
    _write(claude / "settings.local.json", json.dumps({"mcpServers": {"demo-server": {"type": "http"}}}))

    # Runtime state: produced by running the agent, not declared by the user.
    _write(claude / "projects/-a-b/7f3c.jsonl", '{"type":"user"}\n')
    _write(claude / "projects/-a-b/state.json", "{}")
    _write(claude / "sessions/current.json", "{}")
    _write(claude / "shell-snapshots/snap.json", "{}")
    _write(claude / "file-history/edit.json", "{}")
    _write(claude / "history.jsonl", "{}\n")
    # ...except memory, which is loaded into context. Memory is per project, and the index
    # can sit either in memory/ or directly in the project directory.
    _write(claude / "projects/-a-b/memory/MEMORY.md", "- [note](note.md)")
    _write(claude / "projects/-a-b/MEMORY.md", "- [bare index](note.md)")
    own = str(root.resolve()).replace("/", "-").replace(".", "-")
    _write(claude / f"projects/{own}/memory/MEMORY.md", "- [this project](note.md)")

    # A cloned marketplace: plugins that are available to install, not proven active.
    mk = claude / "plugins/marketplaces/claude-plugins-official"
    _write(mk / "plugins/mcp-tunnels/commands/create-docker-mcp-tunnel.md", "command")
    _write(mk / "plugins/mcp-tunnels/skills/tunnel/SKILL.md", "skill")
    _write(mk / "plugins/other-plugin/agents/reviewer.md", "agent")
    _write(mk / "external_plugins/github/.mcp.json", "{}")
    _write(mk / ".github/workflows/check-mcp-urls.yml", "on: push")

    # Harness-shipped catalogs, caches, binaries and credentials inside the config trees.
    _write(claude / "plugins/plugin-directory-cache-v2.json", "{}")
    _write(codex / "vendor_imports/skills/skills/.curated/sec/SKILL.md", "curated")
    _write(codex / "plugins/cache/openai-bundled/browser/skills/b/SKILL.md", "bundled")
    _write(codex / "plugins/.plugin-appserver/codex-cli/codex-package.json", "{}")
    _write(codex / "models_cache.json", "{}")
    _write(codex / "auth.json", '{"token": "SECRET"}')
    _write(codex / "skills/.system/builtin/SKILL.md", "system skill")

    _write(codex / "config.toml", "model = 'gpt'\n[mcp_servers.codex-user]\ncommand = 'y'\n")
    _write(codex / "AGENTS.md", "codex user instructions")
    _write(codex / "sessions/rollout.json", "{}")
    _write(codex / ".codex-global-state.json", "{}")


@contextmanager
def _env():
    """Isolate the scan from the real user environment in both trees."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        _build_project(root)
        _build_home(home, root)
        saved = {k: os.environ.get(k) for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            yield root, home, inspect(str(root))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _paths(result, **match):
    return [f for f in result["files"] if all(f[k] == v for k, v in match.items())]


def test_contract():
    with _env() as (root, _home, a):
        assert a["schema_version"] == "0.2"
        assert a["read_only"] is True
        assert a["root"] == str(root.resolve())
        assert all({"kind", "scope", "source", "harness", "origin"}.issubset(f) for f in a["files"])
        assert a["errors"] == []


def test_dependency_trees_are_pruned():
    """Round-1 defect: node_modules artifacts counted as the user's stack."""
    with _env() as (_root, _home, a):
        assert all("node_modules" not in f["path"] for f in a["files"])
        pruned = {e["path"] for e in a["excluded_paths"] if e["reason"] == "dependency_or_build_tree"}
        assert any(p.endswith("/node_modules") for p in pruned)
        # The nested case matters: a skill may vendor its own dependencies.
        assert any("skills/demo/node_modules" in p for p in pruned)
        assert a["excluded_reasons"]["dependency_or_build_tree"] >= 2


def test_session_transcripts_are_not_config():
    """Round-2 defect: ~/.claude/projects transcripts were 99.3% of reported bytes."""
    with _env() as (_root, _home, a):
        assert all(not f["path"].endswith(".jsonl") for f in a["files"])
        assert all("/sessions/" not in f["path"] for f in a["files"])
        assert all("/shell-snapshots/" not in f["path"] for f in a["files"])
        assert all("/file-history/" not in f["path"] for f in a["files"])
        assert a["excluded_reasons"]["session_transcript_or_project_state"] >= 1
        assert a["excluded_reasons"]["agent_runtime_state"] >= 3


def test_memory_is_kept_as_an_instruction_surface():
    """Pruning projects/ must not take memory/ with it: memory is loaded into context."""
    with _env() as (_root, _home, a):
        memory = _paths(a, kind="memory")
        assert len(memory) == 3
        assert all(m["scope"] == "user" for m in memory)
        # A bare MEMORY.md in the project directory is an index, not session state.
        assert any(m["relative_path"] == "projects/-a-b/MEMORY.md" for m in memory)


def test_marketplace_catalog_is_not_the_active_stack():
    """Real defect: a marketplace catalog holds far more component files than anyone has
    installed, and counting them inflated skills, agents, commands and hooks alike."""
    with _env() as (_root, _home, a):
        assert all("/marketplaces/" not in f["path"] for f in a["files"])
        assert "claude-plugins-official" in a["catalog"]["marketplaces"]
        assert a["catalog"]["marketplaces"]["claude-plugins-official"]["plugins_available"] == 3
        assert a["catalog"]["judgment"] == "available_not_proven_active"
        # known_marketplaces.json is the user's own plugin config and stays.
        assert any(f["relative_path"] == "plugins/known_marketplaces.json" for f in a["files"])


def test_codex_stack_is_discovered():
    """Round-2 defect: AGENTS.md and the whole Codex surface were invisible."""
    with _env() as (_root, _home, a):
        agents_md = _paths(a, kind="agents_md")
        assert {f["scope"] for f in agents_md} == {"project", "user"}
        assert len(agents_md) == 3  # project AGENTS.md, ~/AGENTS.md, ~/.codex/AGENTS.md
        codex_settings = [f for f in _paths(a, kind="settings") if f["harness"] == "codex"]
        assert {f["source"] for f in codex_settings} == {"project_codex_dir", "user_codex_dir"}
        assert a["observed"]["by_harness"]["codex"] == 7


def test_directory_beats_filename_in_classification():
    """Round-2 defect: 'mcp' as a substring overrode the declaring directory."""
    with _env() as (_root, _home, a):
        command = [f for f in a["files"] if f["relative_path"].endswith("create-docker-mcp-tunnel.md")]
        assert len(command) == 1
        assert command[0]["kind"] == "commands"
        assert [f["relative_path"] for f in _paths(a, kind="mcp")] == [".mcp.json"]


def test_entities_are_rolled_up_from_artifacts():
    """Round-2 defect: 219 skill *files* were reported as skills."""
    with _env() as (_root, _home, a):
        skills = a["observed"]["components"]["skills"]
        assert skills["files"] == 6  # the bundled .system skill is excluded
        assert skills["entities"] == 5
        # Same name, two harnesses: both must survive the rollup.
        assert "user/claude_code/shared-skill" in skills["names"]
        assert "user/codex/shared-skill" in skills["names"]
        assert a["observed"]["components"]["commands"]["entities"] == 1
        assert a["observed"]["components"]["agents"]["entities"] == 1


def test_nested_instruction_files_are_found():
    """Round-3 defect: only the project root was read, so monorepo subdirs were invisible."""
    with _env() as (_root, _home, a):
        nested = [f for f in a["files"] if f["source"] == "project_nested"]
        assert [f["relative_path"] for f in nested] == ["packages/api/CLAUDE.md"]
        assert any(f["relative_path"].startswith("packages/api/.claude/skills/") for f in a["files"])


def test_bundled_catalogs_and_caches_are_excluded():
    """Real defect: `~/.codex` vendor_imports and plugins/cache were counted as installed
    components, and both hold far more files than the user ever installed."""
    with _env() as (_root, _home, a):
        assert all("vendor_imports" not in f["path"] for f in a["files"])
        assert all("/plugins/cache/" not in f["path"] for f in a["files"])
        assert all(".plugin-appserver" not in f["path"] for f in a["files"])
        assert all("cache" not in Path(f["path"]).name for f in a["files"])
        assert a["excluded_reasons"]["bundled_catalog"] >= 2
        # A nested cache must be pruned, not just one sitting at the tree root.
        assert any(e["path"].endswith("plugins/cache") for e in a["excluded_paths"])


def test_credentials_are_never_enumerated():
    """Round-3 defect: ~/.codex/auth.json was reported as discovered configuration."""
    with _env() as (_root, _home, a):
        assert all(Path(f["path"]).name not in {"auth.json", "credentials.json"} for f in a["files"])
        assert a["excluded_reasons"]["credential_material"] == 1


def test_bundled_entities_do_not_inflate_the_declared_stack():
    """A harness-shipped skill is observable, but it is not the user's own stack."""
    with _env() as (_root, _home, a):
        bundled = [f for f in a["files"] if f["origin"] == "bundled"]
        assert [f["relative_path"] for f in bundled] == ["skills/.system/builtin/SKILL.md"]
        assert a["observed"]["by_origin"] == {"bundled": 1, "declared": 27}
        # Bundled entities stay visible in files[] but must not reach the entity rollup.
        assert "builtin" not in str(a["observed"]["components"])


def test_scope_and_byte_accounting():
    with _env() as (_root, _home, a):
        assert a["observed"]["by_scope"]["project"] == 13
        assert a["observed"]["by_scope"]["user"] == 15
        assert a["observed"]["files_found"] == 28
        by_scope = a["observed"]["bytes_by_scope"]
        assert by_scope["project"] + by_scope["user"] == a["observed"]["bytes_found"]
        assert a["observed"]["settings_declared_features"]["hooks"] == 2
        assert a["observed"]["settings_declared_features"]["mcp"] == 2


def test_large_instruction_files_flagged_for_both_harnesses():
    with _env() as (_root, _home, a):
        large = [r for r in a["review_candidates"] if r["type"] == "large_instruction_file"]
        assert {Path(r["path"]).name for r in large} == {"CLAUDE.md", "AGENTS.md"}
        assert all(r["judgment"] == "review_only" for r in a["review_candidates"])


def test_unreadable_entries_do_not_crash():
    """A broken symlink and an unreadable directory must degrade, not raise."""
    with _env() as (root, _home, _a):
        (root / ".claude/skills/demo/dangling.md").symlink_to(root / "nope/missing.md")
        locked = root / ".claude/locked"
        locked.mkdir()
        _write(locked / "settings.json", "{}")
        locked.chmod(0o000)
        try:
            b = inspect(str(root))
            assert all(not f["relative_path"].endswith("dangling.md") for f in b["files"])
        finally:
            locked.chmod(0o755)


def test_mcp_servers_resolved_from_all_three_sources():
    """Round-4 defect: only .mcp.json was read, so most declared servers were missed."""
    with _env() as (_root, _home, a):
        found = {(m["scope"], m["name"], m["source"]) for m in a["mcp_servers"]}
        assert ("project", "demo-server", "project_root") in found
        assert ("user", "registry-server", "user_claude_json") in found
        # A per-project override in ~/.claude.json applies to this root only.
        assert ("project", "search-server", "user_claude_json") in found
        assert ("user", "codex-user", "user_codex_dir") in found
        assert ("project", "local", "project_codex_dir") in found
        assert a["observed"]["mcp_servers_declared"] == 6
        assert {m["transport"] for m in a["mcp_servers"]} == {"http", "stdio"}


def test_duplicate_mcp_declarations_are_flagged():
    """Round-5 finding: demo-server was declared in four separate files."""
    with _env() as (_root, _home, a):
        dupes = [r for r in a["review_candidates"] if r["type"] == "mcp_declared_in_multiple_places"]
        assert [d["name"] for d in dupes] == ["demo-server"]
        assert len(dupes[0]["declared_in"]) == 2
        assert dupes[0]["scopes"] == ["project", "user"]


def test_mcp_values_are_never_emitted():
    """Server names and transports are evidence; URLs and credentials are not ours to print."""
    with _env() as (_root, _home, a):
        blob = json.dumps(a)
        assert "FAKE-NOT-A-REAL-TOKEN" not in blob
        assert "example.invalid" not in blob


def test_credentials_in_config_are_flagged_by_key_not_value():
    """Round-4 finding: a real project tracked an MCP bearer token in git."""
    with _env() as (_root, _home, a):
        cred = [r for r in a["review_candidates"] if r["type"] == "credential_in_agent_config"]
        assert len(cred) == 1
        assert cred[0]["keys"] == ["mcpServers.demo-server.headers.Authorization"]
        assert cred[0]["scope"] == "project"
        assert cred[0]["git_tracked"] is False  # the fixture is not a git repo
        assert cred[0]["judgment"] == "review_only"


def test_memory_is_scoped_to_the_scanned_project():
    """Memory is per project, so only this root's memory is reported as applying."""
    with _env() as (_root, _home, a):
        memory = _paths(a, kind="memory")
        assert sum(1 for m in memory if m["applies_to_root"]) == 1
        assert a["observed"]["memory_files_for_root"] == 1
        assert any(m["memory_project"] == "-a-b" and not m["applies_to_root"] for m in memory)


def test_degraded_reads_are_declared():
    """A partial parse must be labelled, never presented as a complete one."""
    with _env() as (_root, _home, a):
        for entry in a["degraded_reads"]:
            assert entry["parser"] in {"header_scan_fallback", "tomllib_error", "unreadable"}
            assert entry["impact"]
        # With tomllib present nothing degrades; without it, config.toml must be listed.
        from agent_profiler.core import tomllib as _toml
        assert (a["degraded_reads"] == []) if _toml else len(a["degraded_reads"]) >= 1


def test_project_depth_is_bounded():
    """The project walk must stay cheap on a deep tree."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write(root / "a/b/c/d/e/f/CLAUDE.md", "at the limit")
        _write(root / "a/b/c/d/e/f/g/CLAUDE.md", "beyond the limit")
        os.environ["AGENT_PROFILER_HOME"] = str(root / "no-home")
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(root / "no-home/.claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(root / "no-home/.codex")
        try:
            a = inspect(str(root))
            found = {f["relative_path"] for f in a["files"]}
            assert "a/b/c/d/e/f/CLAUDE.md" in found
            assert "a/b/c/d/e/f/g/CLAUDE.md" not in found
            assert a["excluded_reasons"]["beyond_project_depth_limit"] == 1
        finally:
            for k in ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME"):
                os.environ.pop(k, None)


def test_profiler_reports_its_own_cost():
    with _env() as (_root, _home, a):
        cost = a["profiler_cost"]
        assert cost["llm_input_tokens"] == 0 and cost["llm_output_tokens"] == 0
        assert cost["dirs_visited"] > 0 and cost["files_considered"] > 0
        assert cost["elapsed_ms"] >= 0


def test_trace_and_compare_unchanged():
    with _env() as (root, _home, a):
        tp = root / "trace.jsonl"
        tp.write_text('{"input_tokens":100,"tool_name":"Read"}\n{"output_tokens":20,"tool_name":"Read"}\n')
        b = trace(str(tp))
        assert b["observed"]["token_fields"]["input_tokens"] == 100
        c = compare(a, a)
        assert all(x["delta"] == 0 for x in c["observed"]["differences"])


def test_another_projects_mcp_declaration_is_found_only_in_whole_environment_mode():
    """The real false negative that motivated `--all`: `search-server` is declared under the
    home directory's project key in ~/.claude.json, is live in sessions, and a scan rooted
    at a subdirectory could not see it. Per-project scoping is right for a project question
    and wrong for an environment one."""
    with _env() as (root, home, scoped):
        _write(home / ".claude.json", json.dumps({
            "mcpServers": {"registry-server": {"command": "x"}},
            "projects": {
                str(root): {"mcpServers": {"search-server": {"command": "npx"}}},
                str(home): {"mcpServers": {"elsewhere": {"command": "npx"}}},
            },
        }))
        scoped = inspect(str(root))
        whole = inspect(str(root), whole_environment=True)

    assert scoped["scope"] == "project" and whole["scope"] == "whole_environment"
    assert "elsewhere" not in {s["name"] for s in scoped["mcp_servers"]}
    found = next(s for s in whole["mcp_servers"] if s["name"] == "elsewhere")
    # It must say whose declaration it is, or the inventory is misleading in the other direction.
    assert found["scope"] == "other_project"
    assert found["declared_for_project"]


def test_an_excluded_server_is_marked_and_never_dropped():
    """Exclusion must not shrink the inventory: a silently shorter list is
    indistinguishable from a smaller environment."""
    with _env() as (root, _home, _a):
        path = root / EXCLUSIONS_FILENAME
        path.write_text(json.dumps({"mcp": ["demo-server"], "reason": "private"}), encoding="utf-8")
        plain = inspect(str(root))
        marked = inspect(str(root), exclusions=load_exclusions(None, root))

    assert len(marked["mcp_servers"]) == len(plain["mcp_servers"])
    excluded = [s for s in marked["mcp_servers"] if s.get("excluded_by_request")]
    assert excluded and all(s["name"] == "demo-server" for s in excluded)
    assert all(s["exclusion_reason"] == "private" for s in excluded)
    assert marked["observed"]["mcp_servers_excluded_by_request"] == len(excluded)
    assert marked["exclusions"]["names"]["mcp"] == ["demo-server"]


def test_a_named_exclusions_file_that_is_missing_is_an_error():
    """Falling back to "nothing excluded" would start sending traffic the user asked this
    tool to withhold, which is the one failure here that must not be silent."""
    with _env() as (root, _home, _a):
        try:
            load_exclusions(str(root / "nope.json"), root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a named missing exclusions file must raise")


def test_exclusions_reject_what_they_cannot_honour():
    with _env() as (root, _home, _a):
        for bad in ({"servers": ["x"]}, {"mcp": "x"}, {"mcp": [""]}, {"mcp": ["x"], "reason": 5}):
            path = root / "bad-exclude.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            try:
                load_exclusions(str(path), root)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad!r}")


def test_invalid_root_exits_nonzero():
    with _env() as (root, _home, _a):
        assert cli_main(["inspect", "--root", str(root / "does-not-exist")]) == 2
        assert cli_main(["inspect", "--root", str(root / "CLAUDE.md")]) == 2
        # An empty root must not silently fall back to the working directory.
        assert cli_main(["inspect", "--root", ""]) == 2


def test_the_shared_agents_skills_root_is_discovered():
    """Real false negative, found by reading a record the project had dismissed.

    `~/.agents/skills` is a harness-independent skills root. It was not scanned, so four
    real skills were missing from an inventory that reported "installed: 15/15 proven" —
    a confident completeness claim that was not complete. A session's
    `world_state.host_skills` names this directory among its skill roots and lists each
    skill with `shown: true`, so these are loaded, not dormant.
    """
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as proj:
        agents = Path(home) / ".agents" / "skills" / "shared-skill"
        agents.mkdir(parents=True)
        (agents / "SKILL.md").write_text("---\nname: shared-skill\n---\nbody\n",
                                         encoding="utf-8")
        saved = os.environ.get("AGENT_PROFILER_AGENTS_HOME")
        os.environ["AGENT_PROFILER_AGENTS_HOME"] = str(Path(home) / ".agents")
        try:
            data = inspect(proj)
        finally:
            if saved is None:
                os.environ.pop("AGENT_PROFILER_AGENTS_HOME", None)
            else:
                os.environ["AGENT_PROFILER_AGENTS_HOME"] = saved
    paths = [f["path"] for f in data["files"] if f.get("kind") == "skills"]
    assert any("shared-skill" in p for p in paths), paths
    entry = next(f for f in data["files"] if "shared-skill" in f["path"])
    # The path does not say which harness loads it, so no harness may be claimed.
    assert entry["harness"] == "shared"
    assert entry["scope"] == "user"


def test_scanning_a_root_that_contains_the_user_tree_does_not_rescan_it():
    """Found by asking what `inspect` walks. Scanning `--all` from the home directory —
    the mode this tool exists for — made the project walk re-enter `~/.claude` and
    `~/.codex` as if they were a project's config directories. `_prune_reason` returns None
    for every rule the moment the tree is `project`, which is right for a repository and
    wrong for these: the user trees have their own curated scan, and re-entering them
    applied none of it.

    On a real environment that added thousands of entries from `.tmp`, plugin caches,
    `vendor_imports`, browser sessions and an app bundle, every one marked
    `origin: declared`. `files_found` was inflated more than twentyfold, the inventory claimed
    593 skill entities against 13, and the Codex attribution map went to 550 components on
    a machine with ten. `~/.codex/.tmp` is dated 2026-11-19, so the numbers inflated
    silently when the harness added a directory — which is why the skip is by identity and
    not by name.
    """
    with tempfile.TemporaryDirectory() as h:
        # Resolved: on macOS /var is a firmlink to /private/var, and the scan reports the
        # resolved path while the fixture would otherwise compare the unresolved one.
        home = Path(h).resolve()
        real = home / ".codex" / "skills" / "mine"
        real.mkdir(parents=True)
        (real / "SKILL.md").write_text("---\nname: mine\n---\nbody\n", encoding="utf-8")
        # The shapes the harness keeps inside its own tree, none of them the user's stack.
        for junk in (".tmp/plugins/vendored/SKILL.md",
                     ".tmp/bundled-marketplaces/catalog/SKILL.md",
                     "plugins/cache/some-plugin/SKILL.md",
                     "vendor_imports/skills/imported/SKILL.md"):
                p = home / ".codex" / junk
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("---\nname: junk\n---\nbody\n", encoding="utf-8")
        # A genuine project under the same root must still be scanned as a project.
        proj = home / "work" / ".claude" / "skills" / "theirs"
        proj.mkdir(parents=True)
        (proj / "SKILL.md").write_text("---\nname: theirs\n---\nbody\n", encoding="utf-8")

        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME",
                  "AGENT_PROFILER_CODEX_HOME", "AGENT_PROFILER_AGENTS_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        os.environ["AGENT_PROFILER_AGENTS_HOME"] = str(home / ".agents")
        try:
            result = inspect(str(home), whole_environment=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    paths = [f["path"] for f in result["files"]]
    # The real skill is found once, under user scope.
    mine = [f for f in result["files"] if f["path"] == str(real / "SKILL.md")]
    assert len(mine) == 1, mine
    assert mine[0]["scope"] == "user" and mine[0]["source"] == "user_codex_dir"
    # Nothing from the harness's own working directories is reported at all.
    for junk in (".tmp", "plugins/cache", "vendor_imports"):
        assert not [p for p in paths if f"/.codex/{junk}" in p], junk
    # A real project's config directory is untouched by the skip.
    theirs = [f for f in result["files"] if f["path"] == str(proj / "SKILL.md")]
    assert len(theirs) == 1 and theirs[0]["scope"] == "project", theirs
    # And the skip is recorded rather than silent.
    reasons = {e["reason"] for e in result["excluded_paths"]}
    assert "user_config_tree_scanned_as_user_scope" in reasons, sorted(reasons)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} scanner regression tests passed")


if __name__ == "__main__":
    main()

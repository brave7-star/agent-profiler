"""Checkpoint completed subjects and resume only an unchanged measurement."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from . import store


def _digest(path: Path, skill: bool) -> str:
    if skill:
        from .codex_probe import _skill_digest

        return _skill_digest(path.parent)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeasurementBatch:
    def __init__(self, root: Path, stage: str, *, resume: bool, checkpoint: bool,
                 parameters: dict):
        self.root, self.stage = root, stage
        self.resume, self.checkpoint = resume, checkpoint
        self.parameters = parameters
        self.previous = (store.stored(keys=store.MEASUREMENT_KEYS) or {}) if resume else {}
        self.paths: list[str] = []
        self.save_errors: list[str] = []
        self.failures: dict[str, int] = {}
        self.stopped: dict[str, str] = {}

    def prepare(self, path: Path, key: str, harness: str, *, skill: bool = False):
        if self.save_errors:
            return None, "checkpoint_failed; stop before spending on another component"
        if harness in self.stopped:
            return None, self.stopped[harness]
        digest = _digest(path, skill)
        if self.resume:
            matching = [r for r in self.previous.get(key, [])
                        if r.get("file") and Path(r["file"]).resolve() == path.resolve()
                        and r.get("harness") == harness and r.get("input_digest") == digest
                        and r.get("probe_parameters") == self.parameters]
            if matching:
                latest = matching[-1]
                if (latest.get("reproducible") is False
                        or (latest.get("measured_under") or {}).get("mixed")
                        or (latest.get("measured_under") or {}).get("arms_on_own_route") is False):
                    return digest, "previous_measurement_unstable; diagnose before repeating"
                return digest, "already_measured_same_content"
        return digest, None

    def record(self, result: dict, path: Path, key: str, harness: str,
               digest: str, *, skill: bool = False) -> None:
        result.setdefault("file", str(path))
        result["harness"] = harness
        result["probe_parameters"] = dict(self.parameters)
        # A file edited during the run cannot be certified as the bytes we measured.
        try:
            unchanged = _digest(path, skill) == digest
        except OSError:
            unchanged = False
        if unchanged:
            result["input_digest"] = digest
        else:
            result["input_changed_during_measurement"] = True

        if self.checkpoint and store._is_measurement(result):
            artifact = {"schema_version": "0.1", "mode": self.stage,
                        "root": str(self.root), "read_only": True,
                        "measured_at": datetime.now(timezone.utc).isoformat(), key: [result]}
            saved = store.remember(artifact, stage=self.stage)
            if saved:
                self.paths.append(str(saved))
            else:
                self.save_errors.append(str(path))

        if result.get("blocker") == "ARMS_NOT_SERVED_ALIKE" or result.get("abandoned"):
            self.stopped[harness] = "batch_stopped; harness measurement needs diagnosis"
            return
        if result.get("ok") or result.get("inconclusive") or result.get("blocker") == "NO_SEPARATION":
            self.failures[harness] = 0
            return
        attempted = bool(result.get("pairs") or result.get("probes_attempted"))
        if attempted:
            self.failures[harness] = self.failures.get(harness, 0) + 1
            if self.failures[harness] >= 2:
                self.stopped[harness] = "batch_stopped; two consecutive component measurements failed"

    def summary(self) -> dict:
        return {"checkpointed": self.checkpoint, "checkpoint_paths": self.paths,
                "checkpoint_errors": self.save_errors, "resume": self.resume,
                "stopped_harnesses": self.stopped}

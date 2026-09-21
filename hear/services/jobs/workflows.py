"""Declarative job routing shared by validation, scheduling, and execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowSpec:
    canonical_name: str
    execution_lane: str
    operation: str
    requires_audio: bool = False
    legacy_alias: bool = False


WORKFLOWS: dict[str, WorkflowSpec] = {
    "pipeline": WorkflowSpec("pipeline", "pipeline", "pipeline", requires_audio=True),
    "transcription": WorkflowSpec(
        "transcription", "transcription", "transcription", requires_audio=True
    ),
    "magic_clean": WorkflowSpec(
        "magic_clean", "magic_clean", "magic_clean", requires_audio=True
    ),
    "categorization": WorkflowSpec("categorization", "categorization", "categorization"),
    "audio_tag": WorkflowSpec("audio_tag", "audio_tag", "audio_tag", requires_audio=True),
    "rebuild": WorkflowSpec("rebuild", "pipeline", "pipeline", requires_audio=True),
    "reconstruct": WorkflowSpec(
        "reconstruct", "reconstruction", "reconstruct", requires_audio=True
    ),
    "edit_transcript": WorkflowSpec(
        "edit_transcript", "reconstruction", "edit_transcript", requires_audio=True
    ),
    "discovery": WorkflowSpec("discovery", "discovery", "discovery", requires_audio=True),
}

JOB_TYPE_ALIASES = {"tagging": "categorization"}


def normalize_job_type(value: str | None) -> str:
    """Normalize transport spelling without silently inventing a workflow."""

    raw = (value or "pipeline").strip().replace("-", "_")
    return JOB_TYPE_ALIASES.get(raw, raw)


def workflow_for(value: str | None) -> WorkflowSpec | None:
    return WORKFLOWS.get(normalize_job_type(value))


def supported_job_types() -> frozenset[str]:
    return frozenset(WORKFLOWS)

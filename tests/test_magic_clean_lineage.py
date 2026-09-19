from __future__ import annotations

import hashlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.services.magic_clean.lineage import (
    MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY,
    MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY,
    MAGIC_CLEAN_ROOT_URL_KEY,
    MagicCleanLineageError,
    extract_enhanced_audio_url,
    resolve_magic_clean_hash_alias,
    resolve_magic_clean_lineage,
    sha256_decoded_pcm,
    sha256_file,
)

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _job(
    job_id: str,
    *,
    input_url: str,
    output_url: str,
    backend_id: str = "backend-a",
    track_id: str = "track-a",
    status: str = "completed",
    job_type: str = "magic_clean",
    root_url: str = "",
    delivered_file_sha256: str = "",
    delivered_pcm_sha256: str = "",
):
    options = {}
    if root_url:
        options[MAGIC_CLEAN_ROOT_URL_KEY] = root_url
    if delivered_file_sha256:
        options[MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY] = delivered_file_sha256
    if delivered_pcm_sha256:
        options[MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY] = delivered_pcm_sha256
    return SimpleNamespace(
        id=job_id,
        backend_id=backend_id,
        track_id=track_id,
        status=status,
        job_type=job_type,
        input_url=input_url,
        job_options=options,
        result_json={"enhanced_audio": {"audio_url": output_url}},
    )


def test_extract_enhanced_audio_url_reads_only_exact_nested_field():
    assert extract_enhanced_audio_url(
        {"enhanced_audio": {"audio_url": "  https://audio/output.mp3  "}}
    ) == "https://audio/output.mp3"
    assert extract_enhanced_audio_url({"enhanced_url": "https://audio/wrong.mp3"}) == ""
    assert extract_enhanced_audio_url({"enhanced_audio": "https://audio/wrong.mp3"}) == ""
    assert extract_enhanced_audio_url(None) == ""


def test_five_exact_derivative_runs_resolve_to_one_canonical_source():
    root_url = "https://audio/root.wav"
    current_url = root_url
    jobs = []
    for index in range(1, 6):
        output_url = f"https://audio/clean-{index}.mp3"
        jobs.append(
            _job(
                f"job-{index}",
                input_url=current_url,
                output_url=output_url,
            )
        )
        current_url = output_url

    resolution = resolve_magic_clean_lineage(
        current_url,
        backend_id="backend-a",
        track_id="track-a",
        jobs=jobs,
    )

    assert resolution.root_url == root_url
    assert resolution.matched_by == "exact_url"
    assert resolution.is_known_derivative
    assert resolution.hops == 5
    assert resolution.lineage_job_ids == (
        "job-5",
        "job-4",
        "job-3",
        "job-2",
        "job-1",
    )


def test_persisted_root_hint_short_circuits_legacy_parent_chain():
    job = _job(
        "job-1",
        input_url="https://audio/intermediate.mp3",
        output_url="https://audio/output.mp3",
        root_url="https://audio/root.wav",
    )

    resolution = resolve_magic_clean_lineage(
        "https://audio/output.mp3",
        backend_id="backend-a",
        track_id="track-a",
        jobs=[job],
    )

    assert resolution.root_url == "https://audio/root.wav"
    assert resolution.lineage_job_ids == ("job-1",)


@pytest.mark.parametrize(
    "candidate",
    [
        _job(
            "cross-backend",
            input_url="https://audio/root.wav",
            output_url="https://audio/output.mp3",
            backend_id="backend-b",
        ),
        _job(
            "cross-track",
            input_url="https://audio/root.wav",
            output_url="https://audio/output.mp3",
            track_id="track-b",
        ),
        _job(
            "not-completed",
            input_url="https://audio/root.wav",
            output_url="https://audio/output.mp3",
            status="running",
        ),
        _job(
            "not-magic-clean",
            input_url="https://audio/root.wav",
            output_url="https://audio/output.mp3",
            job_type="reconstruct",
        ),
    ],
)
def test_unknown_or_cross_scope_exact_output_is_a_new_source(candidate):
    submitted_url = "https://audio/output.mp3"

    resolution = resolve_magic_clean_lineage(
        submitted_url,
        backend_id="backend-a",
        track_id="track-a",
        jobs=[candidate],
    )

    assert resolution.root_url == submitted_url
    assert resolution.matched_by == "new_source"
    assert not resolution.is_known_derivative


def test_exact_url_ambiguity_fails_closed():
    jobs = [
        _job(
            "job-1",
            input_url="https://audio/root-a.wav",
            output_url="https://audio/shared.mp3",
        ),
        _job(
            "job-2",
            input_url="https://audio/root-b.wav",
            output_url="https://audio/shared.mp3",
        ),
    ]

    with pytest.raises(MagicCleanLineageError, match="ambiguous"):
        resolve_magic_clean_lineage(
            "https://audio/shared.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=jobs,
        )


def test_cycle_and_excessive_depth_fail_closed():
    cyclic_jobs = [
        _job(
            "job-a",
            input_url="https://audio/output-b.mp3",
            output_url="https://audio/output-a.mp3",
        ),
        _job(
            "job-b",
            input_url="https://audio/output-a.mp3",
            output_url="https://audio/output-b.mp3",
        ),
    ]
    with pytest.raises(MagicCleanLineageError, match="cycle"):
        resolve_magic_clean_lineage(
            "https://audio/output-a.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=cyclic_jobs,
        )

    deep_jobs = [
        _job(
            "job-1",
            input_url="https://audio/root.wav",
            output_url="https://audio/output-1.mp3",
        ),
        _job(
            "job-2",
            input_url="https://audio/output-1.mp3",
            output_url="https://audio/output-2.mp3",
        ),
    ]
    with pytest.raises(MagicCleanLineageError, match="depth"):
        resolve_magic_clean_lineage(
            "https://audio/output-2.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=deep_jobs,
            max_depth=1,
        )


def test_known_output_without_parent_or_root_fails_closed():
    job = _job(
        "job-1",
        input_url="",
        output_url="https://audio/output.mp3",
    )

    with pytest.raises(MagicCleanLineageError, match="no persisted root or parent"):
        resolve_magic_clean_lineage(
            "https://audio/output.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=[job],
        )


@pytest.mark.parametrize(
    ("hash_field", "argument_name", "expected_method"),
    [
        (
            "delivered_file_sha256",
            "submitted_file_sha256",
            "delivered_file_sha256",
        ),
        (
            "delivered_pcm_sha256",
            "submitted_pcm_sha256",
            "delivered_pcm_sha256",
        ),
    ],
)
def test_hash_alias_resolves_prior_output_to_root(
    hash_field,
    argument_name,
    expected_method,
):
    digest = _digest(expected_method)
    job = _job(
        "job-1",
        input_url="https://audio/root.wav",
        output_url="https://audio/output.mp3",
        **{hash_field: digest},
    )

    resolution = resolve_magic_clean_hash_alias(
        "https://audio/query-alias.mp3?token=changed",
        backend_id="backend-a",
        track_id="track-a",
        jobs=[job],
        **{argument_name: digest.upper()},
    )

    assert resolution.root_url == "https://audio/root.wav"
    assert resolution.matched_by == expected_method
    assert resolution.matched_job_ids == ("job-1",)


def test_multiple_hash_matches_are_safe_only_when_their_roots_agree():
    shared_digest = _digest("copied-bitstream")
    safe_jobs = [
        _job(
            "job-1",
            input_url="https://audio/root.wav",
            output_url="https://audio/output-1.mp3",
            delivered_file_sha256=shared_digest,
        ),
        _job(
            "job-2",
            input_url="https://audio/root.wav",
            output_url="https://audio/output-2.mp3",
            delivered_file_sha256=shared_digest,
        ),
    ]

    resolution = resolve_magic_clean_hash_alias(
        "https://audio/reupload.mp3",
        backend_id="backend-a",
        track_id="track-a",
        jobs=safe_jobs,
        submitted_file_sha256=shared_digest,
    )

    assert resolution.root_url == "https://audio/root.wav"
    assert resolution.matched_job_ids == ("job-1", "job-2")

    conflicting_jobs = [
        safe_jobs[0],
        _job(
            "job-3",
            input_url="https://audio/other-root.wav",
            output_url="https://audio/output-3.mp3",
            delivered_file_sha256=shared_digest,
        ),
    ]
    with pytest.raises(MagicCleanLineageError, match="ambiguous"):
        resolve_magic_clean_hash_alias(
            "https://audio/reupload.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=conflicting_jobs,
            submitted_file_sha256=shared_digest,
        )


def test_cross_scope_hash_and_unrecognized_transcode_are_new_sources():
    known_digest = _digest("known")
    job = _job(
        "job-1",
        input_url="https://audio/root.wav",
        output_url="https://audio/output.mp3",
        backend_id="backend-b",
        delivered_file_sha256=known_digest,
    )

    cross_scope = resolve_magic_clean_hash_alias(
        "https://audio/reupload.mp3",
        backend_id="backend-a",
        track_id="track-a",
        jobs=[job],
        submitted_file_sha256=known_digest,
    )
    transcoded = resolve_magic_clean_hash_alias(
        "https://audio/transcoded.mp3",
        backend_id="backend-a",
        track_id="track-a",
        jobs=[job],
        submitted_file_sha256=_digest("different bytes"),
        submitted_pcm_sha256=_digest("different pcm"),
    )

    assert cross_scope.matched_by == "new_source"
    assert cross_scope.root_url == "https://audio/reupload.mp3"
    assert transcoded.matched_by == "new_source"
    assert transcoded.root_url == "https://audio/transcoded.mp3"


def test_invalid_submitted_hash_is_rejected():
    with pytest.raises(ValueError, match="submitted_file_sha256"):
        resolve_magic_clean_hash_alias(
            "https://audio/input.mp3",
            backend_id="backend-a",
            track_id="track-a",
            jobs=[],
            submitted_file_sha256="not-a-sha256",
        )


def test_file_sha256_streams_exact_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    content = bytes(range(256)) * 100
    path.write_bytes(content)

    assert sha256_file(path, chunk_size=17) == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=0)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_decoded_pcm_hash_is_container_independent_and_repeatable(tmp_path):
    sample_rate = 8_000
    samples = (np.arange(800, dtype=np.int16) % 127 - 63).astype(np.int16)
    wav_path = tmp_path / "source.wav"
    flac_path = tmp_path / "source.flac"
    sf.write(wav_path, samples, sample_rate, subtype="PCM_16")
    sf.write(flac_path, samples, sample_rate, subtype="PCM_16")

    wav_hash = sha256_decoded_pcm(wav_path, chunk_size=31)
    flac_hash = sha256_decoded_pcm(flac_path, chunk_size=47)

    assert sha256_file(wav_path) != sha256_file(flac_path)
    assert wav_hash == flac_hash
    assert wav_hash == sha256_decoded_pcm(wav_path)
    with pytest.raises(ValueError, match="timeout_seconds"):
        sha256_decoded_pcm(wav_path, timeout_seconds=0)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_decoded_pcm_hash_header_distinguishes_rate_and_channel_layout(tmp_path):
    sample_values = np.array([1000, -2000], dtype=np.int16)
    rate_a = tmp_path / "rate-a.wav"
    rate_b = tmp_path / "rate-b.wav"
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    sf.write(rate_a, sample_values, 8_000, subtype="PCM_16")
    sf.write(rate_b, sample_values, 16_000, subtype="PCM_16")
    sf.write(mono, sample_values, 8_000, subtype="PCM_16")
    sf.write(stereo, sample_values.reshape(1, 2), 8_000, subtype="PCM_16")

    assert sha256_decoded_pcm(rate_a) != sha256_decoded_pcm(rate_b)
    assert sha256_decoded_pcm(mono) != sha256_decoded_pcm(stereo)

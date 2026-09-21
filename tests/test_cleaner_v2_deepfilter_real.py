"""Opt-in offline real-checkpoint smoke; not a listening/GPU certification gate."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.runtime.cleaner.deepfilter_loader import PinnedDeepFilterAssets, PinnedDeepFilterFactory
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import AttemptTicket
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterSession
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.fixture(scope="module")
def real_model(tmp_path_factory):
    checkpoint = os.environ.get("HEAR_DF3_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set HEAR_DF3_TEST_CHECKPOINT to the pinned offline checkpoint")
    config = Path(__file__).resolve().parents[1] / "deploy/cleaner/deepfilter3.ini"
    assets = PinnedDeepFilterAssets(
        config,
        "0a926b0471793d7ba7446b07a8bdc10eafa5c9e3b93de4d65496e2cbcacc40d3",
        Path(checkpoint),
        "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003",
        (
            ("deepfilternet", "0.5.6"),
            ("deepfilterlib", "0.5.6"),
            ("torch", "2.8.0+cu128"),
            ("torchaudio", "2.8.0+cu128"),
            ("numpy", "1.26.4"),
        ),
        "cpu",
    )
    assert hashlib.sha256(config.read_bytes()).hexdigest() == assets.config_sha256
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 480000),
        tmp_path_factory.mktemp("df3-real"),
        time.monotonic() + 120,
        threading.Event(),
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = None
    try:
        model = PinnedDeepFilterFactory(assets).open(guard)
        yield model
    finally:
        if model is not None:
            model.close()
        torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "channels,frames", [(1, 1), (1, 479), (1, 481), (1, 48000), (2, 481), (2, 48001)]
)
def test_real_checkpoint_short_and_tail(real_model, channels, frames):
    samples = np.random.default_rng(123).normal(0, 0.01, (channels, frames)).astype(np.float32)
    output = real_model.enhance(samples, 18)
    assert output.shape == samples.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_real_checkpoint_resets_state_and_keeps_silence(real_model):
    samples = np.random.default_rng(456).normal(0, 0.01, (2, 48001)).astype(np.float32)
    first = real_model.enhance(samples, 12)
    silence = real_model.enhance(np.zeros_like(samples), 24)
    second = real_model.enhance(samples, 12)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(silence).all()
    assert np.max(np.abs(silence)) < 1e-7


def test_real_checkpoint_does_not_inherit_caller_autocast(real_model):
    samples = np.random.default_rng(987).normal(0, 0.01, (1, 48000)).astype(np.float32)
    baseline = real_model.enhance(samples, 18)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nested = real_model.enhance(samples, 18)
    np.testing.assert_array_equal(nested, baseline)
    assert nested.dtype == np.float32


@pytest.mark.parametrize("rate", [44100, 96000])
def test_real_checkpoint_file_session_resamples_and_preserves_tail(
    real_model, tmp_path, ticket, rate
):
    policy = ContextualPolicy(48000, 4800)
    ticket["plan"]["runtime"]["longform_policy_sha256"] = policy.digest
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    guard = ResourceGuard(
        ResourceBudget(20000000, 5000000, 600000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    source, destination = tmp_path / "in.wav", tmp_path / "out.wav"
    samples = np.random.default_rng(123).normal(0, 0.01, (rate + 17, 2)).astype(np.float32)
    sf.write(source, samples, rate, subtype="FLOAT")
    # The module fixture owns the real backend lifetime; this session borrows it.
    lease = threading.Lock()
    lease.acquire()
    session = DeepFilterSession(plan, real_model, policy, lease)
    try:
        session.process(source, destination, plan, guard)
    finally:
        lease.release()
    output, actual_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert actual_rate == rate
    assert output.shape == samples.shape
    assert np.isfinite(output).all()
    assert not list(tmp_path.glob("df3-resample-*"))

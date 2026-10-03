"""Shared Natural cleaning plan for DeepFilterNet engine, mastering, and quality tests."""

from hear.services.magic_clean.contracts import CleanPlan, RuntimeIdentity

DIGEST = "a" * 64


def natural_plan(longform_policy_sha256: str = DIGEST) -> CleanPlan:
    return CleanPlan(
        profile="natural",
        profile_version="v1",
        catalogue_sha256=DIGEST,
        runtime=RuntimeIdentity(
            engine="deepfilternet3",
            runtime_sha256=DIGEST,
            checkpoint_sha256=DIGEST,
            precision_policy_sha256=DIGEST,
            longform_policy_sha256=longform_policy_sha256,
        ),
        attenuation_limit_db=18,
        prompt_sha256=None,
        channel_policy="preserve",
        mono_acknowledged=False,
        adjust_loudness=True,
        match_comparison_loudness=True,
        shorten_pauses=False,
        seed=42,
    )

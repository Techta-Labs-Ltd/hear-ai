from dataclasses import dataclass

from hear.contracts.jobs import JobType, MagicCleanProfile, WorkerRole


@dataclass(frozen=True, slots=True)
class RoleCapabilities:
    role: WorkerRole
    job_types: frozenset[JobType]
    magic_clean_profiles: frozenset[MagicCleanProfile]

    def accepts(self, job_type: JobType, profile: MagicCleanProfile | None = None) -> bool:
        if job_type not in self.job_types:
            return False
        if job_type != JobType.MAGIC_CLEAN:
            return True
        return profile is not None and profile in self.magic_clean_profiles


class RoleRegistry:
    def __init__(self) -> None:
        self._roles = {
            WorkerRole.PIPELINE: RoleCapabilities(
                role=WorkerRole.PIPELINE,
                job_types=frozenset({JobType.PIPELINE, JobType.TRANSCRIPTION}),
                magic_clean_profiles=frozenset(),
            ),
            WorkerRole.TRANSCRIPTION: RoleCapabilities(
                role=WorkerRole.TRANSCRIPTION,
                job_types=frozenset({JobType.TRANSCRIPTION}),
                magic_clean_profiles=frozenset(),
            ),
            WorkerRole.RECONSTRUCTION: RoleCapabilities(
                role=WorkerRole.RECONSTRUCTION,
                job_types=frozenset({JobType.RECONSTRUCTION}),
                magic_clean_profiles=frozenset(),
            ),
            WorkerRole.MAGIC_CLEAN_NATURAL: RoleCapabilities(
                role=WorkerRole.MAGIC_CLEAN_NATURAL,
                job_types=frozenset({JobType.MAGIC_CLEAN}),
                magic_clean_profiles=frozenset({MagicCleanProfile.NATURAL}),
            ),
            WorkerRole.MAGIC_CLEAN_VOICE_FOCUS: RoleCapabilities(
                role=WorkerRole.MAGIC_CLEAN_VOICE_FOCUS,
                job_types=frozenset({JobType.MAGIC_CLEAN}),
                magic_clean_profiles=frozenset({MagicCleanProfile.VOICE_FOCUS}),
            ),
            WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE: RoleCapabilities(
                role=WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE,
                job_types=frozenset({JobType.MAGIC_CLEAN}),
                magic_clean_profiles=frozenset({MagicCleanProfile.MUSIC_ATMOSPHERE}),
            ),
            WorkerRole.MAGIC_CLEAN_STEM_MIX: RoleCapabilities(
                role=WorkerRole.MAGIC_CLEAN_STEM_MIX,
                job_types=frozenset({JobType.MAGIC_CLEAN}),
                magic_clean_profiles=frozenset({MagicCleanProfile.STEM_MIX}),
            ),
        }

    def get(self, role: WorkerRole) -> RoleCapabilities:
        return self._roles[role]

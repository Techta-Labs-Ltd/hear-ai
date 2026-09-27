from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from hear.contracts.jobs import AttemptEnvelope, JobType, MagicCleanProfile


class WorkerRole(StrEnum):
    PIPELINE = "pipeline"
    TRANSCRIPTION = "transcription"
    RECONSTRUCTION = "reconstruction"
    MAGIC_CLEAN_NATURAL = "magic_clean_natural"


class WorkerCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: WorkerRole
    job_types: tuple[JobType, ...]
    magic_clean_profile: MagicCleanProfile | None = None
    magic_clean_profiles: tuple[MagicCleanProfile, ...] = ()

    def accepts(self, envelope: AttemptEnvelope) -> bool:
        if envelope.job_type not in self.job_types:
            return False
        if envelope.job_type != JobType.MAGIC_CLEAN:
            return True
        profiles = self.magic_clean_profiles or (
            (self.magic_clean_profile,) if self.magic_clean_profile else ()
        )
        return envelope.options.get("profile") in {profile.value for profile in profiles}


class WorkerCapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities = {
            WorkerRole.PIPELINE: WorkerCapability(
                role=WorkerRole.PIPELINE,
                job_types=(JobType.PIPELINE, JobType.TRANSCRIPTION),
            ),
            WorkerRole.TRANSCRIPTION: WorkerCapability(
                role=WorkerRole.TRANSCRIPTION,
                job_types=(JobType.TRANSCRIPTION,),
            ),
            WorkerRole.RECONSTRUCTION: WorkerCapability(
                role=WorkerRole.RECONSTRUCTION,
                job_types=(JobType.RECONSTRUCTION,),
            ),
            WorkerRole.MAGIC_CLEAN_NATURAL: WorkerCapability(
                role=WorkerRole.MAGIC_CLEAN_NATURAL,
                job_types=(JobType.MAGIC_CLEAN,),
                magic_clean_profile=MagicCleanProfile.NATURAL,
                magic_clean_profiles=tuple(MagicCleanProfile),
            ),
        }

    def get(self, role: WorkerRole) -> WorkerCapability:
        return self._capabilities[role]

    def all(self) -> tuple[WorkerCapability, ...]:
        return tuple(self._capabilities.values())

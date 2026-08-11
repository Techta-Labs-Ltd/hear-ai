import hashlib
import json
import grpc
from unittest.mock import MagicMock

import pytest
from pytest import approx
from google.protobuf.json_format import ParseDict

from hear.config import settings
from hear.core.backend_registry import backend_registry
from hear.proto import pipeline_pb2
from hear.services.transport.grpc import (
    JOB_TYPE_PAYLOAD_MAP,
    PipelineGrpcService,
)
from hear.services.transport.operations import ServiceError


def configure_backend_auth(monkeypatch, service_key: str = "secret") -> None:
    digest = hashlib.sha256(service_key.encode()).hexdigest()
    monkeypatch.setattr(
        settings,
        "BACKEND_REGISTRY_JSON",
        json.dumps({
            "backend-a": {
                "service_key_sha256": digest,
                "allowed_endpoint_urls": ["https://s3.example.test"],
                "allowed_buckets": ["bucket-a"],
                "allowed_public_base_urls": ["https://cdn.example.test"],
            }
        }),
    )
    backend_registry.cache_clear()


class FakeContext:
    def __init__(self, metadata=()):
        self._metadata = metadata
        self.code = None
        self.details = None

    def invocation_metadata(self):
        return self._metadata

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


EVERY_RPC = {
    "SubmitJob": ("SubmitJobRequest", "SubmitJobResponse"),
    "Subscribe": ("SubscribeRequest", "PipelineEvent"),
    "GetResult": ("GetResultRequest", "JobResult"),
    "CancelJob": ("GetResultRequest", "JobResult"),
    "GetQueueStats": ("Empty", "QueueStatsReply"),
    "Moderate": ("TextRequest", "ModerationReply"),
    "Categorize": ("CategorizeRequest", "CategorizationReply"),
    "CreatePreview": ("ReconstructRequest", "CreatePreviewReply"),
    "ConfirmPreview": ("PreviewRequest", "ConfirmPreviewReply"),
    "RemoveSegment": ("RemoveSegmentRequest", "RemoveSegmentReply"),
    "RollbackPreview": ("PreviewRequest", "RollbackPreviewReply"),
    "GetPreview": ("PreviewRequest", "Preview"),
    "ListDiscovery": ("DiscoveryRequest", "ListDiscoveryReply"),
    "TrainCategorizer": ("TrainRequest", "TrainReply"),
    "IngestCategoryEvent": ("CategoryEvent", "IngestReply"),
    "UpdatePlatformSettings": ("PlatformSettingsRequest", "PlatformSettingsReply"),
    "Health": ("Empty", "HealthReply"),
}


class TestTypedResponseContract:
    def test_every_rpc_has_correct_response_type(self):
        svc = pipeline_pb2.DESCRIPTOR.services_by_name["Pipeline"]
        methods = {m.name: m for m in svc.methods}
        assert set(methods) == set(EVERY_RPC), "RPC set mismatch"
        for name, (req, res) in EVERY_RPC.items():
            m = methods[name]
            assert m.input_type.name == req, f"{name}: expected request {req}, got {m.input_type.name}"
            assert m.output_type.name == res, f"{name}: expected response {res}, got {m.output_type.name}"

    def test_no_rpc_returns_google_protobuf_struct(self):
        svc = pipeline_pb2.DESCRIPTOR.services_by_name["Pipeline"]
        struct_name = "google.protobuf.Struct"
        for m in svc.methods:
            full_out = m.output_type.full_name if m.output_type else ""
            assert struct_name not in full_out, (
                f"{m.name} still returns {struct_name}"
            )


class TestParseDictMessageConstruction:
    def test_ingest_reply_accepts_database_uuid(self):
        example_id = "93d58148-d62f-47b7-b7f4-0a11b5ac86fa"
        msg = ParseDict(
            {"status": "accepted", "example_id": example_id},
            pipeline_pb2.IngestReply(),
        )
        assert msg.example_id == example_id

    def test_moderation_reply(self):
        d = dict(
            flagged=True,
            severity="high",
            intent="harmful",
            reason="toxic content",
            flagged_categories=["hate"],
            blocked_words_found=["badword"],
        )
        msg = ParseDict(d, pipeline_pb2.ModerationReply(), ignore_unknown_fields=True)
        assert msg.flagged is True
        assert msg.severity == "high"
        assert msg.intent == "harmful"
        assert msg.reason == "toxic content"
        assert list(msg.flagged_categories) == ["hate"]
        assert list(msg.blocked_words_found) == ["badword"]

    def test_moderation_reply_empty(self):
        msg = ParseDict({}, pipeline_pb2.ModerationReply(), ignore_unknown_fields=True)
        assert msg.flagged is False
        assert msg.severity == ""
        assert list(msg.flagged_categories) == []

    def test_categorization_reply(self):
        d = dict(
            categories=["Nature"],
            tags=["#weather", "#nature"],
            confidence_scores={"#weather": 0.95},
            sentiment="neutral",
            new_tags_added=["#weather"],
            new_categories_added=["Nature"],
            settings_applied=False,
            llm_used=True,
            categorizer_mode="qwen_primary",
        )
        msg = ParseDict(d, pipeline_pb2.CategorizationReply(), ignore_unknown_fields=True)
        assert list(msg.categories) == ["Nature"]
        assert list(msg.tags) == ["#weather", "#nature"]
        assert list(msg.new_tags_added) == ["#weather"]
        assert msg.sentiment == "neutral"
        assert msg.llm_used is True
        assert msg.categorizer_mode == "qwen_primary"

    def test_queue_stats_reply(self):
        d = dict(active=2, queued=5, total=7, estimated_wait_s=30.0, avg_job_duration_s=45.0)
        msg = ParseDict(d, pipeline_pb2.QueueStatsReply(), ignore_unknown_fields=True)
        assert msg.active == 2
        assert msg.queued == 5
        assert msg.total == 7
        assert msg.estimated_wait_s == 30.0
        assert msg.avg_job_duration_s == 45.0

    def test_create_preview_reply(self):
        d = dict(
            preview_id="p-123",
            preview_audio_url="https://storage.example/preview.mp3",
            preview_duration=52.9,
            quality_metrics=dict(
                dnsmos_ovr=3.5,
                loudness_match_db=-1.2,
                duration_delta_ms=15.0,
                clipping_detected=False,
                passed=True,
            ),
            expires_at="2026-07-15T00:00:00Z",
            segments_applied=1,
            track_id="track-123",
        )
        msg = ParseDict(d, pipeline_pb2.CreatePreviewReply(), ignore_unknown_fields=True)
        assert msg.preview_id == "p-123"
        assert msg.preview_duration == approx(52.9)
        assert msg.segments_applied == 1
        assert msg.track_id == "track-123"
        assert msg.quality_metrics.dnsmos_ovr == approx(3.5)
        assert msg.quality_metrics.passed is True

    def test_confirm_preview_reply(self):
        d = dict(
            audio_url="https://storage.example/final.mp3",
            b2_key="reconstructed/track-123/final.mp3",
            duration=52.9,
            track_id="track-123",
            user_id="user-1",
            job_type="reconstruct",
            action="confirm",
            status="completed",
        )
        msg = ParseDict(d, pipeline_pb2.ConfirmPreviewReply(), ignore_unknown_fields=True)
        assert msg.audio_url == "https://storage.example/final.mp3"
        assert msg.duration == approx(52.9)
        assert msg.action == "confirm"
        assert msg.status == "completed"

    def test_confirm_preview_reply_without_user_id(self):
        d = dict(
            audio_url="https://storage.example/final.mp3",
            b2_key="key",
            duration=10.0,
            track_id="t-1",
            job_type="reconstruct",
            action="confirm",
            status="completed",
        )
        msg = ParseDict(d, pipeline_pb2.ConfirmPreviewReply(), ignore_unknown_fields=True)
        assert msg.audio_url == "https://storage.example/final.mp3"
        assert msg.HasField("user_id") is False

    def test_remove_segment_reply(self):
        d = dict(
            audio_url="https://storage.example/result.mp3",
            b2_key="reconstructed/track-123/result.mp3",
            duration=47.0,
            segments_removed=1,
            removed_duration=5.0,
            track_id="track-123",
            user_id="user-1",
            job_type="reconstruct",
            action="remove",
            status="completed",
        )
        msg = ParseDict(d, pipeline_pb2.RemoveSegmentReply(), ignore_unknown_fields=True)
        assert msg.segments_removed == 1
        assert msg.removed_duration == 5.0
        assert msg.action == "remove"

    def test_rollback_preview_reply(self):
        d = dict(preview_id="p-123", status="rolled_back")
        msg = ParseDict(d, pipeline_pb2.RollbackPreviewReply(), ignore_unknown_fields=True)
        assert msg.preview_id == "p-123"
        assert msg.status == "rolled_back"

    def test_preview_message(self):
        d = dict(
            preview_id="p-123",
            track_id="track-123",
            audio_url="https://storage.example/preview.mp3",
            b2_key="previews/track-123/p-123.mp3",
            status="pending",
            expires_at="2026-07-15T00:00:00Z",
            changes=[dict(segment_start=1.0, segment_end=3.0, new_text="hello")],
            same_speaker=True,
            created_at="2026-07-14T12:00:00Z",
            user_id="user-1",
            quality_metrics=dict(
                dnsmos_ovr=3.5,
                loudness_match_db=0.0,
                duration_delta_ms=0.0,
                clipping_detected=False,
                passed=True,
            ),
        )
        msg = ParseDict(d, pipeline_pb2.Preview(), ignore_unknown_fields=True)
        assert msg.preview_id == "p-123"
        assert msg.same_speaker is True
        assert len(msg.changes) == 1
        assert msg.changes[0].segment_start == 1.0
        assert msg.quality_metrics.passed is True

    def test_train_reply(self):
        d = dict(status="completed", detail="trained on 50 examples")
        msg = ParseDict(d, pipeline_pb2.TrainReply(), ignore_unknown_fields=True)
        assert msg.status == "completed"
        assert msg.detail == "trained on 50 examples"

    def test_ingest_reply(self):
        d = dict(status="accepted", example_id="93d58148-d62f-47b7-b7f4-0a11b5ac86fa")
        msg = ParseDict(d, pipeline_pb2.IngestReply(), ignore_unknown_fields=True)
        assert msg.status == "accepted"
        assert msg.example_id == "93d58148-d62f-47b7-b7f4-0a11b5ac86fa"

    def test_platform_settings_reply(self):
        d = dict(status="accepted", blocked_keywords_count=3, auto_tag_keywords_count=5)
        msg = ParseDict(d, pipeline_pb2.PlatformSettingsReply(), ignore_unknown_fields=True)
        assert msg.status == "accepted"
        assert msg.blocked_keywords_count == 3
        assert msg.auto_tag_keywords_count == 5

    def test_health_reply(self):
        d = dict(
            status="healthy",
            gpu_available=True,
            gpu_name="NVIDIA A40",
            gpu_memory=dict(free_mb=9708.3, used_mb=37989.4, total_mb=47697.7),
            active_jobs=0,
            queued_jobs=0,
        )
        msg = ParseDict(d, pipeline_pb2.HealthReply(), ignore_unknown_fields=True)
        assert msg.status == "healthy"
        assert msg.gpu_available is True
        assert msg.gpu_name == "NVIDIA A40"
        assert msg.gpu_memory.free_mb == approx(9708.3)
        assert msg.gpu_memory.used_mb == approx(37989.4)
        assert msg.active_jobs == 0

    def test_health_reply_no_gpu(self):
        d = dict(
            status="healthy",
            gpu_available=False,
            gpu_name="",
            active_jobs=1,
            queued_jobs=2,
        )
        msg = ParseDict(d, pipeline_pb2.HealthReply(), ignore_unknown_fields=True)
        assert msg.gpu_available is False
        assert msg.gpu_name == ""
        assert msg.active_jobs == 1
        assert msg.queued_jobs == 2

    def test_list_discovery_reply(self):
        d = dict(
            sort="latest",
            limit=2,
            offset=0,
            total=47,
            items=[
                dict(
                    track_id="t-1",
                    job_id="j-1",
                    discovery=dict(title="Test", id="c-1"),
                    latest_at="2026-07-14T00:00:00Z",
                    published_at="",
                    trending_score=0.5,
                    completed_at="2026-07-14T00:00:00Z",
                )
            ],
        )
        msg = ParseDict(d, pipeline_pb2.ListDiscoveryReply(), ignore_unknown_fields=True)
        assert msg.sort == "latest"
        assert msg.limit == 2
        assert msg.total == 47
        assert len(msg.items) == 1
        assert msg.items[0].track_id == "t-1"
        assert msg.items[0].trending_score == 0.5

    def test_pipeline_payload(self):
        d = dict(
            source_audio_url="https://storage.example/audio.mp3",
            transcription=dict(
                transcript="Hello world",
                segments=[
                    dict(
                        start=0.0,
                        end=2.0,
                        text="Hello world",
                        speaker="SPEAKER_00",
                        words=[
                            dict(word="Hello", start=0.0, end=0.5, score=0.98, speaker="SPEAKER_00")
                        ],
                    )
                ],
                language="en",
                confidence=0.95,
            ),
            moderation=dict(flagged=False, severity="none", intent="safe"),
            categorization=dict(
                categories=["Nature"],
                tags=["#weather"],
                sentiment="neutral",
                categorizer_mode="qwen_primary",
            ),
            edited_transcript=None,
            discovery=dict(id="c-1", title="Test"),
            content_description="A description",
        )
        msg = ParseDict(d, pipeline_pb2.PipelinePayload(), ignore_unknown_fields=True)
        assert msg.source_audio_url == "https://storage.example/audio.mp3"
        assert msg.transcription.transcript == "Hello world"
        assert len(msg.transcription.segments) == 1
        assert msg.transcription.segments[0].words[0].word == "Hello"
        assert msg.moderation.flagged is False
        assert msg.categorization.categories[0] == "Nature"
        assert msg.content_description == "A description"

    def test_transcription_payload(self):
        d = dict(
            source_audio_url="https://storage.example/audio.mp3",
            transcription=dict(
                transcript="Hello world",
                segments=[],
                language="en",
                confidence=0.95,
            ),
        )
        msg = ParseDict(d, pipeline_pb2.TranscriptionPayload(), ignore_unknown_fields=True)
        assert msg.source_audio_url == "https://storage.example/audio.mp3"
        assert msg.transcription.transcript == "Hello world"
        assert msg.transcription.language == "en"

    def test_audio_tag_payload(self):
        d = dict(
            source_audio_url="https://storage.example/audio.mp3",
            transcription="Tag this as news",
            suggestions=["#news", "#construction"],
        )
        msg = ParseDict(d, pipeline_pb2.AudioTagPayload(), ignore_unknown_fields=True)
        assert msg.transcription == "Tag this as news"
        assert list(msg.suggestions) == ["#news", "#construction"]

    def test_magic_clean_payload(self):
        d = dict(
            enhanced=True,
            enhanced_audio=dict(
                audio_url="https://storage.example/enhanced.mp3",
                b2_key="enhanced/track-123/job-123.mp3",
            ),
            quality=dict(
                quality_score=0.981,
                snr_db=84.12,
                peak_db=-0.09,
                lufs=-16.94,
                clipping_detected=False,
            ),
            stage_times=dict(enhancing=5.0, uploading=1.0),
            transcription=dict(transcript="", segments=[], language="en", confidence=0.0),
            moderation=dict(flagged=False, severity="none", intent="safe"),
        )
        msg = ParseDict(d, pipeline_pb2.MagicCleanPayload(), ignore_unknown_fields=True)
        assert msg.enhanced is True
        assert msg.enhanced_audio.audio_url == "https://storage.example/enhanced.mp3"
        assert msg.quality.quality_score == approx(0.981)
        assert msg.quality.clipping_detected is False

    def test_reconstruct_payload(self):
        d = dict(
            edited_transcript="Edited text",
            rebuilt_audio=dict(
                audio_url="https://storage.example/rebuilt.mp3",
                b2_key="reconstructed/track-123/result.mp3",
                duration=52.9,
            ),
            is_regenerated=True,
            transcription=dict(transcript="", segments=[], language="en", confidence=0.0),
            moderation=dict(flagged=False, severity="none", intent="safe"),
        )
        msg = ParseDict(d, pipeline_pb2.ReconstructPayload(), ignore_unknown_fields=True)
        assert msg.edited_transcript == "Edited text"
        assert msg.rebuilt_audio.duration == approx(52.9)
        assert msg.is_regenerated is True

    def test_reconstruct_payload_without_edited_transcript(self):
        d = dict(
            rebuilt_audio=dict(
                audio_url="https://storage.example/rebuilt.mp3",
                b2_key="reconstructed/track-123/result.mp3",
                duration=52.9,
            ),
            is_regenerated=False,
            transcription=dict(transcript="", segments=[], language="en", confidence=0.0),
            moderation=dict(flagged=False, severity="none", intent="safe"),
        )
        msg = ParseDict(d, pipeline_pb2.ReconstructPayload(), ignore_unknown_fields=True)
        assert msg.HasField("edited_transcript") is False
        assert msg.is_regenerated is False


class TestJobTypePayloadMap:
    def test_covers_all_supported_job_types(self):
        expected = {"pipeline", "categorization", "rebuild", "transcription", "audio_tag", "magic_clean", "reconstruct", "edit_transcript", "discovery"}
        assert set(JOB_TYPE_PAYLOAD_MAP) == expected

    def test_pipeline_group_maps_to_pipeline_field(self):
        for jt in ("pipeline", "categorization", "rebuild"):
            field, cls = JOB_TYPE_PAYLOAD_MAP[jt]
            assert field == "pipeline"
            assert cls is pipeline_pb2.PipelinePayload

    def test_transcription_maps_to_transcription_field(self):
        field, cls = JOB_TYPE_PAYLOAD_MAP["transcription"]
        assert field == "transcription"
        assert cls is pipeline_pb2.TranscriptionPayload

    def test_audio_tag_maps_to_audio_tag_field(self):
        field, cls = JOB_TYPE_PAYLOAD_MAP["audio_tag"]
        assert field == "audio_tag"
        assert cls is pipeline_pb2.AudioTagPayload

    def test_magic_clean_maps_to_magic_clean_field(self):
        field, cls = JOB_TYPE_PAYLOAD_MAP["magic_clean"]
        assert field == "magic_clean"
        assert cls is pipeline_pb2.MagicCleanPayload

    def test_reconstruct_group_maps_to_reconstruct_field(self):
        for jt in ("reconstruct", "edit_transcript"):
            field, cls = JOB_TYPE_PAYLOAD_MAP[jt]
            assert field == "reconstruct"
            assert cls is pipeline_pb2.ReconstructPayload

    def test_every_oneof_field_name_is_valid_on_job_result(self):
        job_result = pipeline_pb2.JobResult()
        for job_type, (field_name, _) in JOB_TYPE_PAYLOAD_MAP.items():
            assert hasattr(job_result, field_name), (
                f"JobResult has no oneof field '{field_name}' for job_type '{job_type}'"
            )


class TestCallHelper:
    @pytest.mark.anyio
    async def test_success_returns_typed_message(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            return {"preview_id": "p-1", "status": "rolled_back"}

        result = await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert isinstance(result, pipeline_pb2.RollbackPreviewReply)
        assert result.preview_id == "p-1"
        assert result.status == "rolled_back"

    @pytest.mark.anyio
    async def test_auth_rejection_returns_empty_message(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "wrong"),))

        async def fn():
            return {"preview_id": "p-1", "status": "rolled_back"}

        result = await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert isinstance(result, pipeline_pb2.RollbackPreviewReply)
        assert result.preview_id == ""
        assert ctx.code == grpc.StatusCode.UNAUTHENTICATED

    @pytest.mark.anyio
    async def test_service_error_sets_grpc_code(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            raise ServiceError(404, "not found")

        result = await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert isinstance(result, pipeline_pb2.RollbackPreviewReply)
        assert ctx.code == grpc.StatusCode.NOT_FOUND
        assert ctx.details == "not found"

    @pytest.mark.anyio
    async def test_service_error_422_sets_invalid_argument(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            raise ServiceError(422, "bad request")

        await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert ctx.details == "bad request"

    @pytest.mark.anyio
    async def test_service_error_503_sets_unavailable(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            raise ServiceError(503, "not ready")

        await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert ctx.code == grpc.StatusCode.UNAVAILABLE
        assert ctx.details == "not ready"

    @pytest.mark.anyio
    async def test_unknown_exception_sets_internal(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            raise RuntimeError("boom")

        await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert ctx.code == grpc.StatusCode.INTERNAL

    @pytest.mark.anyio
    async def test_service_error_409_sets_already_exists(self, monkeypatch):
        configure_backend_auth(monkeypatch, "secret")
        svc = PipelineGrpcService(orchestrator=MagicMock())
        ctx = FakeContext((("x-api-key", "secret"), ("application", "hear")))

        async def fn():
            raise ServiceError(409, "conflict")

        await svc._call(
            ctx,
            fn,
            pipeline_pb2.RollbackPreviewReply,
        )
        assert ctx.code == grpc.StatusCode.ALREADY_EXISTS
        assert ctx.details == "conflict"

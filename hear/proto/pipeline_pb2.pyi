from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SegmentChange(_message.Message):
    __slots__ = ("segment_start", "segment_end", "new_text", "original_text")
    SEGMENT_START_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_END_FIELD_NUMBER: _ClassVar[int]
    NEW_TEXT_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_TEXT_FIELD_NUMBER: _ClassVar[int]
    segment_start: float
    segment_end: float
    new_text: str
    original_text: str
    def __init__(self, segment_start: _Optional[float] = ..., segment_end: _Optional[float] = ..., new_text: _Optional[str] = ..., original_text: _Optional[str] = ...) -> None: ...

class SubmitJobRequest(_message.Message):
    __slots__ = ("job_id", "track_id", "job_type", "max_tags", "audio_url", "edited_transcript", "changes", "same_speaker", "grouped", "group_id", "kind", "source", "track_count", "speed_multipliers", "playback_instruction", "user_id", "speech", "music", "background", "type", "media_file_id", "cut_silence")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_TAGS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    EDITED_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    SAME_SPEAKER_FIELD_NUMBER: _ClassVar[int]
    GROUPED_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TRACK_COUNT_FIELD_NUMBER: _ClassVar[int]
    SPEED_MULTIPLIERS_FIELD_NUMBER: _ClassVar[int]
    PLAYBACK_INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SPEECH_FIELD_NUMBER: _ClassVar[int]
    MUSIC_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MEDIA_FILE_ID_FIELD_NUMBER: _ClassVar[int]
    CUT_SILENCE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    track_id: str
    job_type: str
    max_tags: int
    audio_url: str
    edited_transcript: str
    changes: _containers.RepeatedCompositeFieldContainer[SegmentChange]
    same_speaker: bool
    grouped: bool
    group_id: str
    kind: str
    source: str
    track_count: int
    speed_multipliers: _containers.RepeatedScalarFieldContainer[float]
    playback_instruction: str
    user_id: str
    speech: int
    music: int
    background: int
    type: str
    media_file_id: str
    cut_silence: bool
    def __init__(self, job_id: _Optional[str] = ..., track_id: _Optional[str] = ..., job_type: _Optional[str] = ..., max_tags: _Optional[int] = ..., audio_url: _Optional[str] = ..., edited_transcript: _Optional[str] = ..., changes: _Optional[_Iterable[_Union[SegmentChange, _Mapping]]] = ..., same_speaker: bool = ..., grouped: bool = ..., group_id: _Optional[str] = ..., kind: _Optional[str] = ..., source: _Optional[str] = ..., track_count: _Optional[int] = ..., speed_multipliers: _Optional[_Iterable[float]] = ..., playback_instruction: _Optional[str] = ..., user_id: _Optional[str] = ..., speech: _Optional[int] = ..., music: _Optional[int] = ..., background: _Optional[int] = ..., type: _Optional[str] = ..., media_file_id: _Optional[str] = ..., cut_silence: bool = ...) -> None: ...

class SubmitJobResponse(_message.Message):
    __slots__ = ("job_id", "run_id", "status", "error")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    run_id: str
    status: str
    error: str
    def __init__(self, job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class PipelineEvent(_message.Message):
    __slots__ = ("event", "job_id", "run_id", "track_id", "job_type", "status", "current_stage", "label", "description", "progress_pct", "elapsed_seconds", "estimated_remaining", "error", "result")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PCT_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_REMAINING_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    event: str
    job_id: str
    run_id: str
    track_id: str
    job_type: str
    status: str
    current_stage: str
    label: str
    description: str
    progress_pct: int
    elapsed_seconds: float
    estimated_remaining: float
    error: str
    result: _struct_pb2.Struct
    def __init__(self, event: _Optional[str] = ..., job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., track_id: _Optional[str] = ..., job_type: _Optional[str] = ..., status: _Optional[str] = ..., current_stage: _Optional[str] = ..., label: _Optional[str] = ..., description: _Optional[str] = ..., progress_pct: _Optional[int] = ..., elapsed_seconds: _Optional[float] = ..., estimated_remaining: _Optional[float] = ..., error: _Optional[str] = ..., result: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class GetResultRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class Word(_message.Message):
    __slots__ = ("word", "start", "end", "score", "speaker")
    WORD_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    word: str
    start: float
    end: float
    score: float
    speaker: str
    def __init__(self, word: _Optional[str] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., score: _Optional[float] = ..., speaker: _Optional[str] = ...) -> None: ...

class Segment(_message.Message):
    __slots__ = ("start", "end", "text", "speaker", "words")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    start: float
    end: float
    text: str
    speaker: str
    words: _containers.RepeatedCompositeFieldContainer[Word]
    def __init__(self, start: _Optional[float] = ..., end: _Optional[float] = ..., text: _Optional[str] = ..., speaker: _Optional[str] = ..., words: _Optional[_Iterable[_Union[Word, _Mapping]]] = ...) -> None: ...

class TranscriptionObject(_message.Message):
    __slots__ = ("transcript", "segments", "language", "confidence")
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    transcript: str
    segments: _containers.RepeatedCompositeFieldContainer[Segment]
    language: str
    confidence: float
    def __init__(self, transcript: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[Segment, _Mapping]]] = ..., language: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class QualityMetrics(_message.Message):
    __slots__ = ("dnsmos_ovr", "loudness_match_db", "duration_delta_ms", "clipping_detected", "passed")
    DNSMOS_OVR_FIELD_NUMBER: _ClassVar[int]
    LOUDNESS_MATCH_DB_FIELD_NUMBER: _ClassVar[int]
    DURATION_DELTA_MS_FIELD_NUMBER: _ClassVar[int]
    CLIPPING_DETECTED_FIELD_NUMBER: _ClassVar[int]
    PASSED_FIELD_NUMBER: _ClassVar[int]
    dnsmos_ovr: float
    loudness_match_db: float
    duration_delta_ms: float
    clipping_detected: bool
    passed: bool
    def __init__(self, dnsmos_ovr: _Optional[float] = ..., loudness_match_db: _Optional[float] = ..., duration_delta_ms: _Optional[float] = ..., clipping_detected: bool = ..., passed: bool = ...) -> None: ...

class EnhancedAudio(_message.Message):
    __slots__ = ("audio_url", "b2_key")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    b2_key: str
    def __init__(self, audio_url: _Optional[str] = ..., b2_key: _Optional[str] = ...) -> None: ...

class AudioQuality(_message.Message):
    __slots__ = ("quality_score", "snr_db", "peak_db", "lufs", "clipping_detected")
    QUALITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    SNR_DB_FIELD_NUMBER: _ClassVar[int]
    PEAK_DB_FIELD_NUMBER: _ClassVar[int]
    LUFS_FIELD_NUMBER: _ClassVar[int]
    CLIPPING_DETECTED_FIELD_NUMBER: _ClassVar[int]
    quality_score: float
    snr_db: float
    peak_db: float
    lufs: float
    clipping_detected: bool
    def __init__(self, quality_score: _Optional[float] = ..., snr_db: _Optional[float] = ..., peak_db: _Optional[float] = ..., lufs: _Optional[float] = ..., clipping_detected: bool = ...) -> None: ...

class RebuiltAudio(_message.Message):
    __slots__ = ("audio_url", "b2_key", "duration")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    b2_key: str
    duration: float
    def __init__(self, audio_url: _Optional[str] = ..., b2_key: _Optional[str] = ..., duration: _Optional[float] = ...) -> None: ...

class SegmentAudio(_message.Message):
    __slots__ = ("segment_start", "segment_end", "b2_key", "audio_url", "duration", "is_deletion")
    SEGMENT_START_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_END_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    IS_DELETION_FIELD_NUMBER: _ClassVar[int]
    segment_start: float
    segment_end: float
    b2_key: str
    audio_url: str
    duration: float
    is_deletion: bool
    def __init__(self, segment_start: _Optional[float] = ..., segment_end: _Optional[float] = ..., b2_key: _Optional[str] = ..., audio_url: _Optional[str] = ..., duration: _Optional[float] = ..., is_deletion: bool = ...) -> None: ...

class PipelinePayload(_message.Message):
    __slots__ = ("source_audio_url", "transcription", "moderation", "categorization", "edited_transcript", "discovery", "content_description", "compressed_audio", "report", "flagged")
    SOURCE_AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MODERATION_FIELD_NUMBER: _ClassVar[int]
    CATEGORIZATION_FIELD_NUMBER: _ClassVar[int]
    EDITED_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    DISCOVERY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    FLAGGED_FIELD_NUMBER: _ClassVar[int]
    source_audio_url: str
    transcription: TranscriptionObject
    moderation: ModerationReply
    categorization: CategorizationReply
    edited_transcript: str
    discovery: _struct_pb2.Struct
    content_description: str
    compressed_audio: _struct_pb2.Struct
    report: _struct_pb2.Struct
    flagged: bool
    def __init__(self, source_audio_url: _Optional[str] = ..., transcription: _Optional[_Union[TranscriptionObject, _Mapping]] = ..., moderation: _Optional[_Union[ModerationReply, _Mapping]] = ..., categorization: _Optional[_Union[CategorizationReply, _Mapping]] = ..., edited_transcript: _Optional[str] = ..., discovery: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., content_description: _Optional[str] = ..., compressed_audio: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., report: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., flagged: bool = ...) -> None: ...

class TranscriptionPayload(_message.Message):
    __slots__ = ("source_audio_url", "transcription")
    SOURCE_AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    source_audio_url: str
    transcription: TranscriptionObject
    def __init__(self, source_audio_url: _Optional[str] = ..., transcription: _Optional[_Union[TranscriptionObject, _Mapping]] = ...) -> None: ...

class AudioTagPayload(_message.Message):
    __slots__ = ("source_audio_url", "transcription", "suggestions")
    SOURCE_AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    source_audio_url: str
    transcription: str
    suggestions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, source_audio_url: _Optional[str] = ..., transcription: _Optional[str] = ..., suggestions: _Optional[_Iterable[str]] = ...) -> None: ...

class MagicCleanPayload(_message.Message):
    __slots__ = ("enhanced", "enhanced_audio", "quality", "stage_times", "transcription", "moderation")
    ENHANCED_FIELD_NUMBER: _ClassVar[int]
    ENHANCED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    STAGE_TIMES_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MODERATION_FIELD_NUMBER: _ClassVar[int]
    enhanced: bool
    enhanced_audio: EnhancedAudio
    quality: AudioQuality
    stage_times: _struct_pb2.Struct
    transcription: TranscriptionObject
    moderation: ModerationReply
    def __init__(self, enhanced: bool = ..., enhanced_audio: _Optional[_Union[EnhancedAudio, _Mapping]] = ..., quality: _Optional[_Union[AudioQuality, _Mapping]] = ..., stage_times: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., transcription: _Optional[_Union[TranscriptionObject, _Mapping]] = ..., moderation: _Optional[_Union[ModerationReply, _Mapping]] = ...) -> None: ...

class ReconstructPayload(_message.Message):
    __slots__ = ("edited_transcript", "rebuilt_audio", "is_regenerated", "transcription", "moderation", "segments")
    EDITED_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    REBUILT_AUDIO_FIELD_NUMBER: _ClassVar[int]
    IS_REGENERATED_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MODERATION_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    edited_transcript: str
    rebuilt_audio: RebuiltAudio
    is_regenerated: bool
    transcription: TranscriptionObject
    moderation: ModerationReply
    segments: _containers.RepeatedCompositeFieldContainer[SegmentAudio]
    def __init__(self, edited_transcript: _Optional[str] = ..., rebuilt_audio: _Optional[_Union[RebuiltAudio, _Mapping]] = ..., is_regenerated: bool = ..., transcription: _Optional[_Union[TranscriptionObject, _Mapping]] = ..., moderation: _Optional[_Union[ModerationReply, _Mapping]] = ..., segments: _Optional[_Iterable[_Union[SegmentAudio, _Mapping]]] = ...) -> None: ...

class JobResult(_message.Message):
    __slots__ = ("job_id", "run_id", "track_id", "job_type", "status", "current_stage", "error", "pipeline", "transcription", "audio_tag", "magic_clean", "reconstruct")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    AUDIO_TAG_FIELD_NUMBER: _ClassVar[int]
    MAGIC_CLEAN_FIELD_NUMBER: _ClassVar[int]
    RECONSTRUCT_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    run_id: str
    track_id: str
    job_type: str
    status: str
    current_stage: str
    error: str
    pipeline: PipelinePayload
    transcription: TranscriptionPayload
    audio_tag: AudioTagPayload
    magic_clean: MagicCleanPayload
    reconstruct: ReconstructPayload
    def __init__(self, job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., track_id: _Optional[str] = ..., job_type: _Optional[str] = ..., status: _Optional[str] = ..., current_stage: _Optional[str] = ..., error: _Optional[str] = ..., pipeline: _Optional[_Union[PipelinePayload, _Mapping]] = ..., transcription: _Optional[_Union[TranscriptionPayload, _Mapping]] = ..., audio_tag: _Optional[_Union[AudioTagPayload, _Mapping]] = ..., magic_clean: _Optional[_Union[MagicCleanPayload, _Mapping]] = ..., reconstruct: _Optional[_Union[ReconstructPayload, _Mapping]] = ...) -> None: ...

class TextRequest(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class CategorizeRequest(_message.Message):
    __slots__ = ("text", "custom_tags", "max_tags")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_TAGS_FIELD_NUMBER: _ClassVar[int]
    MAX_TAGS_FIELD_NUMBER: _ClassVar[int]
    text: str
    custom_tags: _containers.RepeatedScalarFieldContainer[str]
    max_tags: int
    def __init__(self, text: _Optional[str] = ..., custom_tags: _Optional[_Iterable[str]] = ..., max_tags: _Optional[int] = ...) -> None: ...

class ReconstructRequest(_message.Message):
    __slots__ = ("audio_url", "track_id", "changes", "segment_start", "segment_end", "new_text", "same_speaker")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_START_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_END_FIELD_NUMBER: _ClassVar[int]
    NEW_TEXT_FIELD_NUMBER: _ClassVar[int]
    SAME_SPEAKER_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    track_id: str
    changes: _containers.RepeatedCompositeFieldContainer[SegmentChange]
    segment_start: float
    segment_end: float
    new_text: str
    same_speaker: bool
    def __init__(self, audio_url: _Optional[str] = ..., track_id: _Optional[str] = ..., changes: _Optional[_Iterable[_Union[SegmentChange, _Mapping]]] = ..., segment_start: _Optional[float] = ..., segment_end: _Optional[float] = ..., new_text: _Optional[str] = ..., same_speaker: bool = ...) -> None: ...

class PreviewRequest(_message.Message):
    __slots__ = ("preview_id", "track_id", "user_id")
    PREVIEW_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    preview_id: str
    track_id: str
    user_id: str
    def __init__(self, preview_id: _Optional[str] = ..., track_id: _Optional[str] = ..., user_id: _Optional[str] = ...) -> None: ...

class RemoveSegmentRequest(_message.Message):
    __slots__ = ("track_id", "audio_url", "segment_start", "segment_end", "user_id")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_START_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_END_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    track_id: str
    audio_url: str
    segment_start: float
    segment_end: float
    user_id: str
    def __init__(self, track_id: _Optional[str] = ..., audio_url: _Optional[str] = ..., segment_start: _Optional[float] = ..., segment_end: _Optional[float] = ..., user_id: _Optional[str] = ...) -> None: ...

class DiscoveryRequest(_message.Message):
    __slots__ = ("sort", "limit", "offset")
    SORT_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    sort: str
    limit: int
    offset: int
    def __init__(self, sort: _Optional[str] = ..., limit: _Optional[int] = ..., offset: _Optional[int] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("target",)
    TARGET_FIELD_NUMBER: _ClassVar[int]
    target: str
    def __init__(self, target: _Optional[str] = ...) -> None: ...

class CategoryEvent(_message.Message):
    __slots__ = ("event_type", "text", "category", "tags", "label", "source_id")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    text: str
    category: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    label: str
    source_id: str
    def __init__(self, event_type: _Optional[str] = ..., text: _Optional[str] = ..., category: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., label: _Optional[str] = ..., source_id: _Optional[str] = ...) -> None: ...

class PlatformSettingsRequest(_message.Message):
    __slots__ = ("blocked_keywords", "auto_tag_keywords")
    BLOCKED_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    AUTO_TAG_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    blocked_keywords: str
    auto_tag_keywords: str
    def __init__(self, blocked_keywords: _Optional[str] = ..., auto_tag_keywords: _Optional[str] = ...) -> None: ...

class QueueStatsReply(_message.Message):
    __slots__ = ("active", "queued", "total", "estimated_wait_s", "avg_job_duration_s")
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    QUEUED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_WAIT_S_FIELD_NUMBER: _ClassVar[int]
    AVG_JOB_DURATION_S_FIELD_NUMBER: _ClassVar[int]
    active: int
    queued: int
    total: int
    estimated_wait_s: float
    avg_job_duration_s: float
    def __init__(self, active: _Optional[int] = ..., queued: _Optional[int] = ..., total: _Optional[int] = ..., estimated_wait_s: _Optional[float] = ..., avg_job_duration_s: _Optional[float] = ...) -> None: ...

class ModerationReply(_message.Message):
    __slots__ = ("flagged", "severity", "intent", "reason", "flagged_categories", "blocked_words_found")
    FLAGGED_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    FLAGGED_CATEGORIES_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_WORDS_FOUND_FIELD_NUMBER: _ClassVar[int]
    flagged: bool
    severity: str
    intent: str
    reason: str
    flagged_categories: _containers.RepeatedScalarFieldContainer[str]
    blocked_words_found: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, flagged: bool = ..., severity: _Optional[str] = ..., intent: _Optional[str] = ..., reason: _Optional[str] = ..., flagged_categories: _Optional[_Iterable[str]] = ..., blocked_words_found: _Optional[_Iterable[str]] = ...) -> None: ...

class CategorizationReply(_message.Message):
    __slots__ = ("categories", "tags", "confidence_scores", "sentiment", "new_tags_added", "new_categories_added", "settings_applied", "llm_used", "categorizer_mode")
    CATEGORIES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_SCORES_FIELD_NUMBER: _ClassVar[int]
    SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    NEW_TAGS_ADDED_FIELD_NUMBER: _ClassVar[int]
    NEW_CATEGORIES_ADDED_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_APPLIED_FIELD_NUMBER: _ClassVar[int]
    LLM_USED_FIELD_NUMBER: _ClassVar[int]
    CATEGORIZER_MODE_FIELD_NUMBER: _ClassVar[int]
    categories: _containers.RepeatedScalarFieldContainer[str]
    tags: _containers.RepeatedScalarFieldContainer[str]
    confidence_scores: _struct_pb2.Struct
    sentiment: str
    new_tags_added: _containers.RepeatedScalarFieldContainer[str]
    new_categories_added: _containers.RepeatedScalarFieldContainer[str]
    settings_applied: bool
    llm_used: bool
    categorizer_mode: str
    def __init__(self, categories: _Optional[_Iterable[str]] = ..., tags: _Optional[_Iterable[str]] = ..., confidence_scores: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., sentiment: _Optional[str] = ..., new_tags_added: _Optional[_Iterable[str]] = ..., new_categories_added: _Optional[_Iterable[str]] = ..., settings_applied: bool = ..., llm_used: bool = ..., categorizer_mode: _Optional[str] = ...) -> None: ...

class CreatePreviewReply(_message.Message):
    __slots__ = ("preview_id", "preview_audio_url", "preview_duration", "quality_metrics", "expires_at", "segments_applied", "track_id", "segments")
    PREVIEW_ID_FIELD_NUMBER: _ClassVar[int]
    PREVIEW_AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    PREVIEW_DURATION_FIELD_NUMBER: _ClassVar[int]
    QUALITY_METRICS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_APPLIED_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    preview_id: str
    preview_audio_url: str
    preview_duration: float
    quality_metrics: QualityMetrics
    expires_at: str
    segments_applied: int
    track_id: str
    segments: _containers.RepeatedCompositeFieldContainer[SegmentAudio]
    def __init__(self, preview_id: _Optional[str] = ..., preview_audio_url: _Optional[str] = ..., preview_duration: _Optional[float] = ..., quality_metrics: _Optional[_Union[QualityMetrics, _Mapping]] = ..., expires_at: _Optional[str] = ..., segments_applied: _Optional[int] = ..., track_id: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[SegmentAudio, _Mapping]]] = ...) -> None: ...

class ConfirmPreviewReply(_message.Message):
    __slots__ = ("audio_url", "b2_key", "duration", "track_id", "user_id", "job_type", "action", "status")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    b2_key: str
    duration: float
    track_id: str
    user_id: str
    job_type: str
    action: str
    status: str
    def __init__(self, audio_url: _Optional[str] = ..., b2_key: _Optional[str] = ..., duration: _Optional[float] = ..., track_id: _Optional[str] = ..., user_id: _Optional[str] = ..., job_type: _Optional[str] = ..., action: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class RemoveSegmentReply(_message.Message):
    __slots__ = ("audio_url", "b2_key", "duration", "segments_removed", "removed_duration", "track_id", "user_id", "job_type", "action", "status")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_REMOVED_FIELD_NUMBER: _ClassVar[int]
    REMOVED_DURATION_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    b2_key: str
    duration: float
    segments_removed: int
    removed_duration: float
    track_id: str
    user_id: str
    job_type: str
    action: str
    status: str
    def __init__(self, audio_url: _Optional[str] = ..., b2_key: _Optional[str] = ..., duration: _Optional[float] = ..., segments_removed: _Optional[int] = ..., removed_duration: _Optional[float] = ..., track_id: _Optional[str] = ..., user_id: _Optional[str] = ..., job_type: _Optional[str] = ..., action: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class RollbackPreviewReply(_message.Message):
    __slots__ = ("preview_id", "status")
    PREVIEW_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    preview_id: str
    status: str
    def __init__(self, preview_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class Preview(_message.Message):
    __slots__ = ("preview_id", "track_id", "audio_url", "b2_key", "status", "expires_at", "changes", "same_speaker", "created_at", "user_id", "quality_metrics")
    PREVIEW_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    B2_KEY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    SAME_SPEAKER_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    QUALITY_METRICS_FIELD_NUMBER: _ClassVar[int]
    preview_id: str
    track_id: str
    audio_url: str
    b2_key: str
    status: str
    expires_at: str
    changes: _containers.RepeatedCompositeFieldContainer[SegmentChange]
    same_speaker: bool
    created_at: str
    user_id: str
    quality_metrics: QualityMetrics
    def __init__(self, preview_id: _Optional[str] = ..., track_id: _Optional[str] = ..., audio_url: _Optional[str] = ..., b2_key: _Optional[str] = ..., status: _Optional[str] = ..., expires_at: _Optional[str] = ..., changes: _Optional[_Iterable[_Union[SegmentChange, _Mapping]]] = ..., same_speaker: bool = ..., created_at: _Optional[str] = ..., user_id: _Optional[str] = ..., quality_metrics: _Optional[_Union[QualityMetrics, _Mapping]] = ...) -> None: ...

class DiscoveryItem(_message.Message):
    __slots__ = ("track_id", "job_id", "discovery", "latest_at", "published_at", "trending_score", "completed_at")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    DISCOVERY_FIELD_NUMBER: _ClassVar[int]
    LATEST_AT_FIELD_NUMBER: _ClassVar[int]
    PUBLISHED_AT_FIELD_NUMBER: _ClassVar[int]
    TRENDING_SCORE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    track_id: str
    job_id: str
    discovery: _struct_pb2.Struct
    latest_at: str
    published_at: str
    trending_score: float
    completed_at: str
    def __init__(self, track_id: _Optional[str] = ..., job_id: _Optional[str] = ..., discovery: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., latest_at: _Optional[str] = ..., published_at: _Optional[str] = ..., trending_score: _Optional[float] = ..., completed_at: _Optional[str] = ...) -> None: ...

class ListDiscoveryReply(_message.Message):
    __slots__ = ("sort", "limit", "offset", "total", "items")
    SORT_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    sort: str
    limit: int
    offset: int
    total: int
    items: _containers.RepeatedCompositeFieldContainer[DiscoveryItem]
    def __init__(self, sort: _Optional[str] = ..., limit: _Optional[int] = ..., offset: _Optional[int] = ..., total: _Optional[int] = ..., items: _Optional[_Iterable[_Union[DiscoveryItem, _Mapping]]] = ...) -> None: ...

class TrainReply(_message.Message):
    __slots__ = ("status", "detail")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    status: str
    detail: str
    def __init__(self, status: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class IngestReply(_message.Message):
    __slots__ = ("status", "example_id")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLE_ID_FIELD_NUMBER: _ClassVar[int]
    status: str
    example_id: str
    def __init__(self, status: _Optional[str] = ..., example_id: _Optional[str] = ...) -> None: ...

class PlatformSettingsReply(_message.Message):
    __slots__ = ("status", "blocked_keywords_count", "auto_tag_keywords_count")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_KEYWORDS_COUNT_FIELD_NUMBER: _ClassVar[int]
    AUTO_TAG_KEYWORDS_COUNT_FIELD_NUMBER: _ClassVar[int]
    status: str
    blocked_keywords_count: int
    auto_tag_keywords_count: int
    def __init__(self, status: _Optional[str] = ..., blocked_keywords_count: _Optional[int] = ..., auto_tag_keywords_count: _Optional[int] = ...) -> None: ...

class GpuMemory(_message.Message):
    __slots__ = ("free_mb", "used_mb", "total_mb")
    FREE_MB_FIELD_NUMBER: _ClassVar[int]
    USED_MB_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MB_FIELD_NUMBER: _ClassVar[int]
    free_mb: float
    used_mb: float
    total_mb: float
    def __init__(self, free_mb: _Optional[float] = ..., used_mb: _Optional[float] = ..., total_mb: _Optional[float] = ...) -> None: ...

class HealthReply(_message.Message):
    __slots__ = ("status", "gpu_available", "gpu_name", "gpu_memory", "active_jobs", "queued_jobs")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    GPU_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_JOBS_FIELD_NUMBER: _ClassVar[int]
    QUEUED_JOBS_FIELD_NUMBER: _ClassVar[int]
    status: str
    gpu_available: bool
    gpu_name: str
    gpu_memory: GpuMemory
    active_jobs: int
    queued_jobs: int
    def __init__(self, status: _Optional[str] = ..., gpu_available: bool = ..., gpu_name: _Optional[str] = ..., gpu_memory: _Optional[_Union[GpuMemory, _Mapping]] = ..., active_jobs: _Optional[int] = ..., queued_jobs: _Optional[int] = ...) -> None: ...

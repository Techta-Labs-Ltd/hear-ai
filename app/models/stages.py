from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    description: str
    progress_start: int
    progress_end: int

    @property
    def progress_mid(self) -> int:
        return (self.progress_start + self.progress_end) // 2


PIPELINE = [
    Stage("transcribing",    "Transcribing audio",       "Converting speech to text",           0, 25),
    Stage("correcting",      "Correcting transcript",    "Applying punctuation and word fixes",  25, 30),
    Stage("moderating",      "Checking content safety",  "Running content moderation checks",    30, 45),
    Stage("categorizing",    "Tagging content",          "Categorizing by topic and theme",     45, 55),
    Stage("discovering",     "Building discovery",       "Creating content profile",            55, 60),
    Stage("compressing",     "Creating audio variants",  "Generating MP3 and speed layers",     60, 100),
]

EDIT_TRANSCRIPT = [
    Stage("downloading",       "Downloading audio",        "Fetching source audio",                0, 10),
    Stage("transcribing",      "Transcribing audio",       "Getting word timestamps from speech",  10, 25),
    Stage("diffing_transcript","Finding changes",          "Comparing original vs edited text",    25, 35),
    Stage("reconstructing_edits","Regenerating speech",    "Creating new audio for edited parts",  35, 100),
]

REBUILD = [
    Stage("rebuilding_audio",  "Rebuilding audio",         "Regenerating entire track from text",   0, 70),
    Stage("moderating",        "Checking content safety",  "Running content moderation checks",     70, 80),
    Stage("discovering",       "Building discovery",       "Creating content profile",             80, 100),
]

RECONSTRUCT = [
    Stage("downloading",    "Downloading audio",        "Fetching source audio",              0, 15),
    Stage("reconstructing", "Reconstructing segments",  "Replacing audio at edited positions", 15, 100),
]

TRANSCRIPTION = [
    Stage("transcribing", "Transcribing audio", "Converting speech to text", 0, 100),
]

CATEGORIZATION = [
    Stage("transcribing",  "Transcribing audio",       "Converting speech to text",           0, 40),
    Stage("moderating",    "Checking content safety",  "Running content moderation checks",   40, 50),
    Stage("categorizing",  "Tagging content",          "Categorizing by topic and theme",     50, 100),
]

MAGIC_CLEAN = [
    Stage("enhancing", "Enhancing audio quality", "Improving clarity and reducing noise", 0, 100),
]

AUDIO_TAG = [
    Stage("audio_tagging", "Extracting audio tags", "Identifying spoken keywords", 0, 100),
]

FLOWS: dict[str, list[Stage]] = {
    "pipeline":       PIPELINE,
    "edit_transcript": EDIT_TRANSCRIPT,
    "rebuild":        REBUILD,
    "reconstruct":    RECONSTRUCT,
    "transcription":  TRANSCRIPTION,
    "categorization": CATEGORIZATION,
    "magic_clean":    MAGIC_CLEAN,
    "audio_tag":      AUDIO_TAG,
}


def get_stage(job_type: str, stage_id: str) -> Stage | None:
    flow = FLOWS.get(job_type, [])
    for s in flow:
        if s.id == stage_id:
            return s
    return None


def get_progress(stage: Stage) -> int:
    return stage.progress_mid


def get_label(job_type: str, stage_id: str) -> str:
    s = get_stage(job_type, stage_id)
    return s.label if s else stage_id


def get_description(job_type: str, stage_id: str) -> str:
    s = get_stage(job_type, stage_id)
    return s.description if s else ""

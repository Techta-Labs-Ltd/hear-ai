from hear.models.stages import DISCOVERY, MAGIC_CLEAN


def test_magic_clean_stream_stages_are_ordered_and_cover_full_progress():
    assert [stage.id for stage in MAGIC_CLEAN] == [
        "downloading",
        "separating",
        "enhancing",
        "mixing",
        "finalizing",
    ]
    assert MAGIC_CLEAN[0].progress_start == 0
    assert MAGIC_CLEAN[-1].progress_end == 100
    assert all(
        current.progress_end == following.progress_start
        for current, following in zip(MAGIC_CLEAN, MAGIC_CLEAN[1:])
    )


def test_standalone_discovery_reuses_or_generates_transcription_first():
    assert [stage.id for stage in DISCOVERY] == ["transcribing", "discovering"]
    assert DISCOVERY[0].progress_start == 0
    assert DISCOVERY[-1].progress_end == 100

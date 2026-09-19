from hear.services.magic_clean.pipeline import MagicCleanPipeline


class Loader:
    def __init__(self):
        self.path = None

    def load(self, path=None):
        self.path = path


def test_magic_clean_pipeline_passes_configured_mossformer_path():
    mossformer = Loader()
    pipeline = MagicCleanPipeline(
        mossformer=mossformer,
        noise=object(),
        speech=object(),
        dynamics=object(),
        stem=object(),
    )

    pipeline.load("htdemucs", "/models/custom-mossformer")

    assert mossformer.path == "/models/custom-mossformer"

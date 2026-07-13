class ProcessingStage:
    name: str = ""
    _ready: bool = False

    def load(self):
        self._ready = True

    async def process(self, ctx) -> any:
        raise NotImplementedError

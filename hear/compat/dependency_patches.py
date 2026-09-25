from hear.tools.dependency_patches import DependencyPatchManager


class PatchVerifier:
    def __init__(self, manager: DependencyPatchManager | None = None) -> None:
        self._manager = manager or DependencyPatchManager()

    def verify(self) -> bool:
        self._manager.run(check=True)
        return True

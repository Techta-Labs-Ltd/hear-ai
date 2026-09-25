import threading


class HarmKeywordLoader:
    def __init__(self):
        self._harm_keywords: list[str] = []
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def load(self):
        from hear.models.database import DatabaseRuntime, HarmKeyword

        db = DatabaseRuntime.SessionLocal()
        try:
            harm = [
                row.keyword
                for row in db.query(HarmKeyword).filter(HarmKeyword.kind == "harm").all()
            ]
        finally:
            db.close()
        with self._lock:
            self._harm_keywords = harm

    def load_keywords(self, keywords: list[str]) -> None:
        normalized = [str(item).strip() for item in keywords if str(item).strip()]
        with self._lock:
            self._harm_keywords = normalized

    @property
    def harm_keywords(self) -> list[str]:
        with self._lock:
            return list(self._harm_keywords)

harm_keyword_loader = HarmKeywordLoader()

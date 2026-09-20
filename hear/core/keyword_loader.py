import threading

from hear.models.database import DatabaseRuntime, HarmKeyword

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

    @property
    def harm_keywords(self) -> list[str]:
        with self._lock:
            return list(self._harm_keywords)

harm_keyword_loader = HarmKeywordLoader()

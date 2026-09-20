import threading

from hear.models.database import AutoTagKeyword, HarmKeyword, SessionLocal


class HarmKeywordLoader:
    def __init__(self):
        self._harm_keywords: list[str] = []
        self._platform_keywords: list[str] = []
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def load(self):
        db = SessionLocal()
        try:
            harm = [row.keyword for row in db.query(HarmKeyword).filter(HarmKeyword.kind == "harm").all()]
            platform = [row.keyword for row in db.query(HarmKeyword).filter(HarmKeyword.kind == "platform").all()]
        finally:
            db.close()
        with self._lock:
            self._harm_keywords = harm
            self._platform_keywords = platform

    @property
    def harm_keywords(self) -> list[str]:
        with self._lock:
            return list(self._harm_keywords)

    @property
    def platform_keywords(self) -> list[str]:
        with self._lock:
            return list(self._platform_keywords)

    @property
    def all_keywords(self) -> list[str]:
        with self._lock:
            return list(self._harm_keywords) + list(self._platform_keywords)

    def sync_platform_keywords(self, keywords: list[str]):
        normalized = [k.strip().lower() for k in keywords if k.strip()]
        db = SessionLocal()
        try:
            db.query(HarmKeyword).filter(HarmKeyword.kind == "platform").delete()
            for kw in normalized:
                db.add(HarmKeyword(keyword=kw, kind="platform"))
            db.commit()
        finally:
            db.close()
        with self._lock:
            self._platform_keywords = normalized


harm_keyword_loader = HarmKeywordLoader()


class AutoTagKeywordLoader:
    """Platform-pushed keywords that always get applied as tags (e.g. news, breaking,
    exclusive) -- pushed via Pipeline.UpdatePlatformSettings over gRPC, comma-separated."""

    def __init__(self):
        self._keywords: list[str] = []
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def load(self):
        db = SessionLocal()
        try:
            keywords = [row.keyword for row in db.query(AutoTagKeyword).all()]
        finally:
            db.close()
        with self._lock:
            self._keywords = keywords

    @property
    def keywords(self) -> list[str]:
        with self._lock:
            return list(self._keywords)

    def sync(self, keywords: list[str]):
        normalized = [k.strip().lower() for k in keywords if k.strip()]
        db = SessionLocal()
        try:
            db.query(AutoTagKeyword).delete()
            for kw in normalized:
                db.add(AutoTagKeyword(keyword=kw))
            db.commit()
        finally:
            db.close()
        with self._lock:
            self._keywords = normalized


auto_tag_keyword_loader = AutoTagKeywordLoader()

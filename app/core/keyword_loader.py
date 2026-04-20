import os
import threading

from app.config import settings


class HarmKeywordLoader:
    def __init__(self):
        self._harm_keywords: list[str] = []
        self._platform_keywords: list[str] = []
        self._lock = threading.Lock()

    def load(self, path: str = None):
        path = path or settings.HARM_KEYWORDS_FILE
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        section = None
        harm = []
        platform = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].upper()
                continue
            if section == "HARM_KEYWORDS":
                harm.append(line.lower())
            elif section == "PLATFORM_KEYWORDS":
                platform.append(line.lower())

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

    def add_harm_keyword(self, keyword: str):
        kw = keyword.strip().lower()
        if not kw:
            return
        with self._lock:
            if kw not in self._harm_keywords:
                self._harm_keywords.append(kw)
                self._save()

    def add_platform_keyword(self, keyword: str):
        kw = keyword.strip().lower()
        if not kw:
            return
        with self._lock:
            if kw not in self._platform_keywords:
                self._platform_keywords.append(kw)
                self._save()

    def remove_platform_keyword(self, keyword: str):
        kw = keyword.strip().lower()
        with self._lock:
            if kw in self._platform_keywords:
                self._platform_keywords.remove(kw)
                self._save()

    def sync_platform_keywords(self, keywords: list[str]):
        normalized = [k.strip().lower() for k in keywords if k.strip()]
        with self._lock:
            self._platform_keywords = normalized
            self._save()

    def _save(self):
        path = settings.HARM_KEYWORDS_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[HARM_KEYWORDS]\n")
            for kw in self._harm_keywords:
                f.write(f"{kw}\n")
            f.write("\n[PLATFORM_KEYWORDS]\n")
            for kw in self._platform_keywords:
                f.write(f"{kw}\n")


harm_keyword_loader = HarmKeywordLoader()

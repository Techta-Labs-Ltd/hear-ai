from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: int = Field(ge=1)
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    keyword_rules: tuple[tuple[str, str], ...]
    harm_keywords: tuple[str, ...]
    taxonomy_paths: tuple[str, ...]

    @classmethod
    def empty(cls) -> PipelineConfiguration:
        return cls(
            version=1,
            categories=(),
            tags=(),
            keyword_rules=(),
            harm_keywords=(),
            taxonomy_paths=(),
        )

    @field_validator("categories", "tags", "harm_keywords", "taxonomy_paths", mode="before")
    @classmethod
    def normalize_string_lists(cls, values):
        if values is None:
            return ()
        if not isinstance(values, (list, tuple)):
            raise ValueError("pipeline configuration list must be an array")
        return tuple(str(value).strip() for value in values if str(value).strip())

    @field_validator("keyword_rules", mode="before")
    @classmethod
    def normalize_keyword_rules(cls, values):
        if values is None:
            return ()
        if isinstance(values, dict):
            items = values.items()
        elif isinstance(values, (list, tuple)):
            items = values
        else:
            raise ValueError("pipeline keyword rules must be an object or pairs")
        return tuple(
            sorted(
                (str(pattern).strip(), str(tag).strip())
                for pattern, tag in items
                if str(pattern).strip() and str(tag).strip()
            )
        )


class PipelineConfigurationLoader:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._configuration: PipelineConfiguration | None = None

    def load(self) -> PipelineConfiguration:
        configuration = PipelineConfiguration.model_validate(
            json.loads(self._path.read_text(encoding="utf-8"))
        )
        self._configuration = configuration
        return configuration

    @property
    def configuration(self) -> PipelineConfiguration:
        if self._configuration is None:
            raise RuntimeError("pipeline_configuration_not_loaded")
        return self._configuration


class CategoryLabels:
    @staticmethod
    def is_hierarchical_taxonomy_path(label: str) -> bool:
        return " > " in (label or "")

    @staticmethod
    def _taxonomy_path_to_tag(path: str) -> str:
        parts = [part.strip().lower() for part in (path or "").split(">") if part.strip()]
        if not parts:
            return ""
        slug_parts = []
        for part in parts:
            slug = re.sub("[^a-z0-9]+", "-", part).strip("-")
            if slug:
                slug_parts.append(slug)
        slug = re.sub("-+", "-", "-".join(slug_parts))
        return f"#{slug}" if slug else ""


@dataclass(frozen=True)
class CategoryData:
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    keyword_rules: Mapping[str, str]
    all_labels: tuple[str, ...]


@dataclass(frozen=True)
class PipelineCategoryCatalog:
    data: CategoryData

    def __init__(self, configuration: PipelineConfiguration) -> None:
        object.__setattr__(
            self,
            "data",
            CategoryData(
                categories=configuration.categories,
                tags=configuration.tags,
                keyword_rules=MappingProxyType(dict(configuration.keyword_rules)),
                all_labels=configuration.categories + configuration.tags,
            ),
        )

    def flat_catalog_categories(self) -> list[str]:
        return [
            category
            for category in self.data.categories
            if category.strip() and not CategoryLabels.is_hierarchical_taxonomy_path(category)
        ]


@dataclass(frozen=True)
class TaxonomyData:
    paths: tuple[str, ...]
    path_lookup: Mapping[str, str]


class TaxonomyLabels:
    @staticmethod
    def _norm(value: str) -> str:
        return re.sub("\\s+", " ", (value or "").strip().lower())

    @staticmethod
    def _hierarchical_segments(key: str) -> list[str]:
        return [TaxonomyLabels._norm(part) for part in key.split(" > ") if part.strip()]

    @staticmethod
    def _topic_matches_taxonomy_path(topic: str, key: str) -> bool:
        normalized_topic = TaxonomyLabels._norm(topic)
        if not normalized_topic or len(normalized_topic) < 3:
            return False
        if normalized_topic == key:
            return True
        if " > " in key:
            segments = TaxonomyLabels._hierarchical_segments(key)
            if normalized_topic in segments:
                return True
            if len(normalized_topic) >= 5 and any(normalized_topic == segment for segment in segments):
                return True
            return any(len(segment) >= 5 and segment in normalized_topic for segment in segments)
        if normalized_topic in key or key in normalized_topic:
            return True
        tokens = re.findall("[a-z]{4,}", key)
        return bool(tokens) and sum(token in normalized_topic for token in tokens) >= min(2, len(tokens))


@dataclass(frozen=True)
class PipelineTaxonomy:
    data: TaxonomyData

    def __init__(self, configuration: PipelineConfiguration) -> None:
        lookup = {TaxonomyLabels._norm(path): path for path in configuration.taxonomy_paths}
        object.__setattr__(
            self,
            "data",
            TaxonomyData(
                paths=configuration.taxonomy_paths,
                path_lookup=MappingProxyType(lookup),
            ),
        )

    def match_paths_for_topics(self, topics: list[str]) -> list[str]:
        if not topics:
            return []
        matched = []
        seen = set()
        for topic in topics:
            normalized_topic = TaxonomyLabels._norm(topic)
            if not normalized_topic:
                continue
            for key, path in self.data.path_lookup.items():
                if key not in seen and TaxonomyLabels._topic_matches_taxonomy_path(
                    normalized_topic, key
                ):
                    matched.append(path)
                    seen.add(key)
        return matched[:12]

    def canonicalize_path(self, path: str) -> str:
        cleaned = re.sub("\\s+", " ", (path or "").strip())
        if not cleaned:
            return ""
        key = TaxonomyLabels._norm(cleaned)
        hit = self.data.path_lookup.get(key)
        if hit:
            return hit
        for candidate, canonical in self.data.path_lookup.items():
            if TaxonomyLabels._topic_matches_taxonomy_path(key, candidate):
                return canonical
        return cleaned

    def taxonomy_label_terms(self) -> frozenset[str]:
        terms = set()
        for path in self.data.paths:
            terms.add(TaxonomyLabels._norm(path))
            terms.update(TaxonomyLabels._norm(part) for part in path.split(" > ") if part.strip())
        return frozenset(terms)


@dataclass(frozen=True)
class PipelineCatalog:
    configuration: PipelineConfiguration
    category_catalog: PipelineCategoryCatalog
    taxonomy: PipelineTaxonomy
    harm_keywords: tuple[str, ...]

    def __init__(self, configuration: PipelineConfiguration) -> None:
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "category_catalog", PipelineCategoryCatalog(configuration))
        object.__setattr__(self, "taxonomy", PipelineTaxonomy(configuration))
        object.__setattr__(self, "harm_keywords", configuration.harm_keywords)

    @property
    def categories(self) -> tuple[str, ...]:
        return self.configuration.categories

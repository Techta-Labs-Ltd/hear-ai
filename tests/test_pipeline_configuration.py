import json

import pytest

from hear.services.pipeline.catalog import PipelineCatalogClient
from hear.services.pipeline.configuration import (
    CategoryLabels,
    PipelineCategoryCatalog,
    PipelineConfiguration,
    PipelineConfigurationLoader,
    PipelineTaxonomy,
)


def test_taxonomy_path_to_tag():
    assert (
        CategoryLabels._taxonomy_path_to_tag("Accessibility > Guide dogs")
        == "#accessibility-guide-dogs"
    )


def test_configuration_normalizes_and_freezes_runtime_catalog():
    configuration = PipelineConfiguration.model_validate(
        {
            "version": 3,
            "categories": [" News ", ""],
            "tags": ["#local"],
            "keyword_rules": {" council ": " #local "},
            "harm_keywords": ["dangerword"],
            "taxonomy_paths": ["Civic > Parish Council"],
        }
    )
    categories = PipelineCategoryCatalog(configuration)
    taxonomy = PipelineTaxonomy(configuration)

    assert configuration.categories == ("News",)
    assert categories.data.keyword_rules == {"council": "#local"}
    with pytest.raises(TypeError):
        categories.data.keyword_rules["tree"] = "#tree"
    with pytest.raises(TypeError):
        taxonomy.data.path_lookup["tree"] = "Tree"
    with pytest.raises(ValueError):
        configuration.version = 4


def test_file_loader_does_not_mutate_process_global_state(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": ["News"],
                "tags": [],
                "keyword_rules": {},
                "harm_keywords": [],
                "taxonomy_paths": [],
            }
        ),
        encoding="utf-8",
    )
    loader = PipelineConfigurationLoader(path)

    configuration = loader.load()

    assert configuration is loader.configuration
    assert configuration.version == 1
    assert configuration.categories == ("News",)


def test_version_is_required_and_validated():
    with pytest.raises(ValueError):
        PipelineConfiguration.model_validate(
            {
                "categories": [],
                "tags": [],
                "keyword_rules": {},
                "harm_keywords": [],
                "taxonomy_paths": [],
            }
        )
    with pytest.raises(ValueError):
        PipelineConfiguration.model_validate(
            {
                "version": 0,
                "categories": [],
                "tags": [],
                "keyword_rules": {},
                "harm_keywords": [],
                "taxonomy_paths": [],
            }
        )
    with pytest.raises(ValueError):
        PipelineConfiguration.model_validate(
            {
                "version": "1",
                "categories": [],
                "tags": [],
                "keyword_rules": {},
                "harm_keywords": [],
                "taxonomy_paths": [],
            }
        )


def test_backend_catalog_fetch_requires_and_uses_versioned_configuration(monkeypatch):
    payload = {
        "version": 4,
        "categories": ["News"],
        "tags": ["#local"],
        "keyword_rules": {"council": "#local"},
        "harm_keywords": ["dangerword"],
        "taxonomy_paths": ["Civic > Parish Council"],
    }

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        @staticmethod
        def get(url, *, headers):
            assert url == "https://backend.test/internal/ai/runtime/catalog"
            assert headers == {"X-Service-Key": "service"}
            return Response()

    monkeypatch.setattr("hear.services.pipeline.catalog.httpx.Client", Client)

    catalog = PipelineCatalogClient("https://backend.test", "service").fetch()

    assert catalog.configuration.version == 4
    assert catalog.categories == ("News",)
    assert catalog.category_catalog.data.keyword_rules["council"] == "#local"

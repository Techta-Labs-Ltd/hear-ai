from hear.core.category_loader import category_loader
from hear.core.discovery_taxonomy import discovery_taxonomy_loader
from hear.core.keyword_loader import harm_keyword_loader
from hear.services.categorization.discovery import DiscoveryService
from hear.services.pipeline.catalog import PipelineCatalog, PipelineCatalogSnapshot


def test_pipeline_catalog_owns_isolated_runtime_loaders():
    catalog = PipelineCatalog(
        PipelineCatalogSnapshot(
            categories=("News",),
            tags=("#local",),
            keyword_rules={"council": "#local"},
            taxonomy_paths=("Civic > Parish Council",),
            harm_keywords=("dangerword",),
        )
    )

    assert catalog.category_loader is not category_loader
    assert catalog.taxonomy is not discovery_taxonomy_loader
    assert catalog.harm_keywords is not harm_keyword_loader
    assert catalog.category_loader.data.categories == ["News"]
    assert catalog.taxonomy.data.paths == ["Civic > Parish Council"]
    assert catalog.harm_keywords.harm_keywords == ["dangerword"]


def test_discovery_uses_injected_taxonomy():
    catalog = PipelineCatalog(
        PipelineCatalogSnapshot(
            categories=(),
            tags=(),
            keyword_rules={},
            taxonomy_paths=("Civic > Parish Council",),
            harm_keywords=(),
        )
    )
    service = DiscoveryService(taxonomy=catalog.taxonomy)
    assert service._is_plausible_speaker("Parish Council", "Parish Council") is False

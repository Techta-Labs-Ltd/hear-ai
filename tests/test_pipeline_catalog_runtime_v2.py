from hear.services.categorization.discovery import DiscoveryService
from hear.services.pipeline.configuration import PipelineCatalog, PipelineConfiguration


def test_pipeline_catalog_owns_isolated_runtime_loaders():
    catalog = PipelineCatalog(
        PipelineConfiguration(
            version=2,
            categories=("News",),
            tags=("#local",),
            keyword_rules={"council": "#local"},
            taxonomy_paths=("Civic > Parish Council",),
            harm_keywords=("dangerword",),
        )
    )

    assert catalog.category_catalog.data.categories == ("News",)
    assert catalog.taxonomy.data.paths == ("Civic > Parish Council",)
    assert catalog.harm_keywords == ("dangerword",)
    assert catalog.configuration.version == 2


def test_discovery_uses_injected_taxonomy():
    catalog = PipelineCatalog(
        PipelineConfiguration(
            version=2,
            categories=(),
            tags=(),
            keyword_rules={},
            taxonomy_paths=("Civic > Parish Council",),
            harm_keywords=(),
        )
    )
    service = DiscoveryService(taxonomy=catalog.taxonomy)
    assert service._is_plausible_speaker("Parish Council", "Parish Council") is False

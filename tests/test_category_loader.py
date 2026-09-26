from hear.core.category_loader import CategoryLabels, CategoryLoader


def test_taxonomy_path_to_tag():
    assert (
        CategoryLabels._taxonomy_path_to_tag("Accessibility > Guide dogs")
        == "#accessibility-guide-dogs"
    )


def test_snapshot_replaces_catalog_atomically():
    loader = CategoryLoader()
    loader.load_snapshot(
        ["News", "Sport"],
        ["#news", "#football"],
        {"football": "#football"},
    )
    data = loader.data
    assert data.categories == ["News", "Sport"]
    assert data.tags == ["#news", "#football"]
    assert data.keyword_rules == {"football": "#football"}
    assert data.all_labels == ["News", "Sport", "#news", "#football"]


def test_snapshot_filters_blank_values():
    loader = CategoryLoader()
    loader.load_snapshot(
        ["News", "", " "],
        ["#news", ""],
        {"": "#bad", "tree": "", "news": "#news"},
    )
    assert loader.data.categories == ["News"]
    assert loader.data.tags == ["#news"]
    assert loader.data.keyword_rules == {"news": "#news"}

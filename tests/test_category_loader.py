import tempfile
from pathlib import Path

from hear.core.category_loader import CategoryLoader, _taxonomy_path_to_tag


def test_taxonomy_path_to_tag():
    assert _taxonomy_path_to_tag("Accessibility > Guide dogs") == "#accessibility-guide-dogs"


def test_ensure_labels_persists_new_entries():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "categories.txt"
        path.write_text(
            "[CATEGORIES]\nNews\n\n[TAGS]\n#news\n\n[KEYWORDS]\n",
            encoding="utf-8",
        )
        loader = CategoryLoader()
        loader.load(str(path))
        new_tags, new_cats = loader.ensure_labels(
            ["#guidedogs", "#assistivetechnology"],
            ["Personal lived experience"],
        )
        assert "#guidedogs" in new_tags
        assert "Personal lived experience" in new_cats
        text = path.read_text(encoding="utf-8")
        assert "#guidedogs" in text
        assert "Personal lived experience" in text


def test_import_discovery_taxonomy_adds_path_and_slug_tag():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "categories.txt"
        path.write_text(
            "[CATEGORIES]\nNews\n\n[TAGS]\n#news\n\n[KEYWORDS]\n",
            encoding="utf-8",
        )
        loader = CategoryLoader()
        loader.load(str(path))
        tags_added, cats_added = loader.import_discovery_taxonomy(
            ["Accessibility > Guide dogs"]
        )
        assert "Guide dogs" in cats_added
        assert "Accessibility > Guide dogs" not in cats_added
        assert "#accessibility-guide-dogs" in tags_added
        text = path.read_text(encoding="utf-8")
        assert "Guide dogs" in text
        assert "Accessibility > Guide dogs" not in text
        assert "#accessibility-guide-dogs" in text

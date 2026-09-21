import ast
import json
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.build_cleaner_wheel import CleanerWheelBuilder

ROOT = Path(__file__).resolve().parents[1]


def test_allowlist_closes_internal_imports_without_legacy_modules():
    files = CleanerWheelBuilder.files(ROOT)
    selected = set(files)
    forbidden = {"sqlalchemy", "psycopg2", "clearvoice", "demucs", "whisperx", "qwen_asr"}
    for name in files:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                if node.module.startswith("hear"):
                    modules.extend(node.module + "." + v.name for v in node.names)
            for module in modules:
                assert module.split(".")[0] not in forbidden
                if module == "hear" or module.startswith("hear."):
                    base = module.replace(".", "/")
                    candidates = (base + ".py", base + "/__init__.py")
                    actual = [v for v in candidates if (ROOT / v).is_file()]
                    assert all(v in selected for v in actual), (name, module)
    assert "hear/proto/pipeline_pb2.py" not in selected
    assert not any("processing/" in name or "database" in name for name in selected)


def test_dependency_lock_excludes_legacy_and_database_packages():
    lock = tomllib.loads((ROOT / "deploy/cleaner/uv.lock").read_text())
    names = {v["name"] for v in lock["package"]}
    assert not names & {
        "hear-ai",
        "clearvoice",
        "demucs",
        "sqlalchemy",
        "psycopg2-binary",
        "whisperx",
        "qwen-asr",
        "transformers",
    }
    assert {"deepfilternet", "deepfilterlib", "onnxruntime", "torch"} <= names


def test_sam_extra_matches_factory_without_text_encoder_or_optional_models():
    project = tomllib.loads((ROOT / "deploy/cleaner/pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["sam"] == [
        "torch==2.8.0+cu128",
        "einops==0.8.2",
    ]
    lock = tomllib.loads((ROOT / "deploy/cleaner/uv.lock").read_text())
    versions = {package["name"]: package["version"] for package in lock["package"]}
    assert versions["einops"] == "0.8.2"
    assert versions["soundfile"] == "0.12.1"
    assert versions["torch"] == "2.8.0+cu128"
    assert not {"transformers", "audiotools", "descript-audio-tools", "dacvae"} & versions.keys()


@pytest.mark.parametrize("entry", ["../.env", "/tmp/private.py", "hear/private.env"])
def test_unsafe_allowlist_entries_rejected(tmp_path, entry):
    config = tmp_path / "deploy/cleaner"
    config.mkdir(parents=True)
    (config / "package-files.json").write_text(json.dumps([entry]))
    with pytest.raises(ValueError):
        CleanerWheelBuilder.files(tmp_path)


def test_wheel_audit_rejects_extra_application_payload(tmp_path):
    wheel = tmp_path / "unsafe.whl"
    files = CleanerWheelBuilder.files(ROOT)
    with zipfile.ZipFile(wheel, "w") as output:
        for name in files:
            output.writestr(name, (ROOT / name).read_bytes())
        output.writestr("hear/core/database.py", "")
    with pytest.raises(ValueError):
        CleanerWheelBuilder.audit(wheel, files, ROOT)


def test_build_does_not_overwrite_existing_artifacts(tmp_path):
    sentinel = tmp_path / "keep.whl"
    sentinel.write_bytes(b"existing")
    with pytest.raises(ValueError):
        CleanerWheelBuilder.build(ROOT, tmp_path)
    assert sentinel.read_bytes() == b"existing"

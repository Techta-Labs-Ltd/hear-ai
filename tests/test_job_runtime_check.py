import json
import sys

from hear.inference.manifest import ModelManifest
from hear.runtime.ownership import DeploymentOwnership
from scripts.check_job_runtime import JobRuntimeCheck


class TestJobRuntimeCheck:
    def test_invalid_settings_report_blocker_without_echoing_secret_input(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEAR_RUNTIME_MODE", "production")
        monkeypatch.setenv("HEAR_POD_STACK_ROLES", "pipeline")
        monkeypatch.setenv("HEAR_BACKEND_INTERNAL_URL", "private-invalid-backend-value")
        output = tmp_path / "check.json"
        monkeypatch.setattr(sys, "argv", ["check", "--output", str(output)])
        assert JobRuntimeCheck.main() == 2
        text = output.read_text()
        assert "private-invalid-backend-value" not in text
        assert "invalid_runtime_setting:HEAR_BACKEND_INTERNAL_URL" in json.loads(text)["blockers"]

    def test_pipeline_reports_missing_backend_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEAR_RUNTIME_MODE", "production")
        monkeypatch.setenv("HEAR_POD_STACK_ROLES", "pipeline")
        monkeypatch.delenv("HEAR_BACKEND_INTERNAL_URL", raising=False)
        monkeypatch.delenv("HEAR_BACKEND_SERVICE_KEY", raising=False)
        monkeypatch.setattr(DeploymentOwnership, "load", lambda: None)
        monkeypatch.setattr(ModelManifest, "validate_local", lambda *args, **kwargs: [])
        monkeypatch.setattr(ModelManifest, "license_blockers", lambda *args, **kwargs: [])
        output = tmp_path / "check.json"
        monkeypatch.setattr(sys, "argv", ["check", "--output", str(output)])
        assert JobRuntimeCheck.main() == 2
        report = json.loads(output.read_text())
        assert "backend_callback_url_missing" in report["blockers"]
        assert "backend_service_key_missing" in report["blockers"]
        assert "backend_ownership_policy_missing" in report["blockers"]
        assert not report["live_inference_verified"]

    def test_cleaner_does_not_require_pipeline_catalog_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEAR_RUNTIME_MODE", "production")
        monkeypatch.setenv("HEAR_POD_STACK_ROLES", "magic_clean_natural")
        monkeypatch.setenv("HEAR_BACKEND_INTERNAL_URL", "https://api.hear.media/api/v1")
        monkeypatch.setenv("HEAR_BACKEND_SERVICE_KEY", "")
        monkeypatch.setattr(DeploymentOwnership, "load", lambda: None)
        output = tmp_path / "check.json"
        monkeypatch.setattr(sys, "argv", ["check", "--output", str(output)])
        assert JobRuntimeCheck.main() == 2
        report = json.loads(output.read_text())
        assert "backend_service_key_missing" not in report["blockers"]
        assert "backend_callback_url_missing" not in report["blockers"]

from pathlib import Path

from hear.tools.check_architecture import ArchitectureChecker


class TestArchitecture:
    def test_application_functions_have_owners(self):
        assert ArchitectureChecker(Path(__file__).resolve().parents[1]).check() == []

    def test_checker_rejects_free_functions_and_nested_imports(self, tmp_path):
        (tmp_path / "main.py").write_text("def loose():\n    import os\n")
        violations = ArchitectureChecker(tmp_path).check()
        assert any("standalone function loose" in item for item in violations)
        assert any("nested import" in item for item in violations)

    def test_shared_utilities_allow_free_functions(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        utilities = tmp_path / "hear/utils"
        utilities.mkdir(parents=True)
        (utilities / "text.py").write_text("def normalize(value):\n    return value.strip()\n")
        assert ArchitectureChecker(tmp_path).check() == []

    def test_imports_precede_executable_code(self, tmp_path):
        (tmp_path / "main.py").write_text("value = 1\nimport os\n")
        assert any(
            "import after executable code" in item for item in ArchitectureChecker(tmp_path).check()
        )


import argparse
import ast
from pathlib import Path

FORBIDDEN_IMPORT_PREFIXES = frozenset(
    {
        "ray",
        "grpc",
        "sqlalchemy",
        "psycopg",
        "psycopg2",
        "asyncpg",
        "redis",
        "hear.proto",
        "hear.deployments",
        "hear.orchestrator",
    }
)


class ArchitectureChecker:
    def __init__(self, root: Path):
        self.root = root

    def check(self) -> list[str]:
        paths = []
        main = self.root / "main.py"
        if main.is_file():
            paths.append(main)
        for directory in ("hear", "scripts"):
            paths.extend((self.root / directory).rglob("*.py"))
        violations = []
        for path in paths:
            relative = path.relative_to(self.root)
            if "proto" in relative.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(relative))
            is_utility = relative.parts[:2] == ("hear", "utils")
            allows_lazy_imports = relative in {
                Path("hear/bootstrap.py"),
                Path("hear/inference/fish_speech.py"),
                # Optional Fish dependencies load only in the supervised child.
                Path("hear/inference/fish_nf4_loader.py"),
                Path("hear/inference/text_generation.py"),
            }
            executable_seen = False
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if executable_seen:
                        violations.append(f"{relative}:{node.lineno}: import after executable code")
                elif not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    executable_seen = True
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not is_utility:
                    violations.append(f"{relative}:{node.lineno}: standalone function {node.name}")
            for owner in ast.walk(tree):
                imported_modules: list[str] = []
                if isinstance(owner, ast.Import):
                    imported_modules.extend(alias.name for alias in owner.names)
                elif isinstance(owner, ast.ImportFrom) and owner.module:
                    imported_modules.append(owner.module)
                elif isinstance(owner, ast.Call):
                    function = owner.func
                    is_dynamic_import = isinstance(function, ast.Attribute) and (
                        function.attr == "import_module"
                    )
                    if (
                        is_dynamic_import
                        and owner.args
                        and isinstance(owner.args[0], ast.Constant)
                        and isinstance(owner.args[0].value, str)
                    ):
                        imported_modules.append(owner.args[0].value)
                for module in imported_modules:
                    if any(
                        module == prefix or module.startswith(prefix + ".")
                        for prefix in FORBIDDEN_IMPORT_PREFIXES
                    ):
                        line = getattr(owner, "lineno", 0)
                        violations.append(f"{relative}:{line}: forbidden legacy import {module}")
                if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for descendant in ast.walk(owner):
                        if (
                            isinstance(descendant, (ast.Import, ast.ImportFrom))
                            and not allows_lazy_imports
                        ):
                            violations.append(
                                f"{relative}:{descendant.lineno}: nested import"
                            )
        return sorted(set(violations))

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description="Check class ownership and utility boundaries")
        parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
        args = parser.parse_args()
        violations = cls(args.root).check()
        print("\n".join(violations) if violations else "Architecture checks passed")
        return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(ArchitectureChecker.main())

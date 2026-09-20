import argparse
import ast
from pathlib import Path


class ArchitectureChecker:
    def __init__(self, root: Path):
        self.root = root

    def check(self) -> list[str]:
        paths = [self.root / "main.py"]
        for directory in ("hear", "scripts"):
            paths.extend((self.root / directory).rglob("*.py"))
        violations = []
        for path in paths:
            relative = path.relative_to(self.root)
            if "proto" in relative.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(relative))
            is_utility = relative.parts[:2] == ("hear", "utils")
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
                if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for node in ast.walk(owner):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            violations.append(f"{relative}:{node.lineno}: nested import")
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

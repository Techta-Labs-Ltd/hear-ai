import json
from pathlib import Path

from hear.contracts.scope import ExecutionScope
from scripts.contract_vectors import ContractVectors


def test_published_vectors_match_the_code():
    published = json.loads((Path(__file__).resolve().parents[1] / "docs" / "contract-vectors.json").read_text())
    built = ContractVectors.build()
    assert published["scope_sha256"] == built["scope_sha256"] == ExecutionScope.digest(built["envelope"] | {"reporting_grant": "x"})
    assert published["grant"]["token"] == built["grant"]["token"]
    assert published["envelope"] == built["envelope"]

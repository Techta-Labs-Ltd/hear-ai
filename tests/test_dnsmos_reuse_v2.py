from pathlib import Path
from unittest.mock import MagicMock, patch

from hear.services.reconstruction.dnsmos import DNSMOSScorer


def test_dnsmos_session_load_is_idempotent(tmp_path: Path):
    model = tmp_path / "dnsmos.onnx"
    model.write_bytes(b"model")
    session = MagicMock()

    with patch(
        "hear.services.reconstruction.dnsmos.ort.InferenceSession",
        return_value=session,
    ) as create:
        scorer = DNSMOSScorer(model)
        assert scorer.load() is True
        assert scorer.load() is True

    create.assert_called_once()

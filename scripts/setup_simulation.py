"""Create isolated local test settings without reading production credentials."""

import hashlib
import ipaddress
import json
import secrets
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


class SetupSimulation:
    @staticmethod
    def main():
        root = Path("/root/hear-ai-v11/simulation-20260928")
        root.mkdir(parents=True, exist_ok=True)
        if (root / "config.json").exists():
            raise RuntimeError("simulation already configured; existing data retained")
        for name in ("tls", "sources", "objects", "logs", "scratch"):
            (root / name).mkdir(exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Hear local evaluation")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
            .not_valid_after(datetime.now(UTC) + timedelta(days=7))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]
                ),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        (root / "tls/cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        (root / "tls/key.pem").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        (root / "tls/key.pem").chmod(0o600)
        token, service = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        config = {"ingress_token": token, "service_key": service, "bucket": "hear-simulation-local"}
        (root / "config.json").write_text(json.dumps(config))
        (root / "config.json").chmod(0o600)
        shutil.copyfile(
            "/workspace/hear-ai-v11/Bad Quality Tracks 0406 - Track 15.mp3",
            root / "sources/input.mp3",
        )
        shutil.copyfile(
            "/workspace/hear-ai-v11/clean/fish-nf4-20260928/01_real_fish_nf4.wav",
            root / "sources/fish-reference.wav",
        )
        base = "https://127.0.0.1:18081"
        policy = {
            "backend_id": "simulation-local",
            "backend_base_urls": [base + "/api/v1"],
            "bucket_name": config["bucket"],
            "storage_endpoint": base + "/s3",
            "public_base_url": base + "/s3/" + config["bucket"],
            "source_hosts": ["127.0.0.1"],
        }
        registry = {
            "backends": [
                {
                    "policy": policy,
                    "callback_base_url": base + "/api/v1",
                    "ingress_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                }
            ]
        }
        env = {
            "HEAR_RUNTIME_MODE": "simulation",
            "HEAR_SIMULATION_ROOT": str(root),
            "HEAR_BACKEND_REGISTRY_JSON": json.dumps(registry, separators=(",", ":")),
            "HEAR_BACKEND_INTERNAL_URL": base + "/api/v1",
            "HEAR_BACKEND_SERVICE_KEY": service,
            "HEAR_PIPELINE_CATALOG_BACKEND_ID": "simulation-local",
            "HEAR_POD_API_KEY": token,
            "HEAR_MODEL_ROOT": "/models",
            "HEAR_TEMP_DIR": str(root / "scratch"),
            "FISH_SPEECH_HOME": "/root/hear-ai-v11/models/fish-speech/source-nf4",
            "FISH_SPEECH_MODEL_ROOT": "/root/hear-ai-v11/models",
            "FISH_SPEECH_BNB_MODE": "nf4",
            "HEAR_OPTIONAL_ENGINE_MODE": "available",
            "HEAR_MAGIC_CLEAN_MODEL_DEVICE": "cuda:0",
            "HEAR_RABBITMQ_URL": "amqp://guest:guest@127.0.0.1:5672/%2F",
            "HEAR_QUEUE_EXCHANGE": "hear.simulation.jobs",
            "HEAR_QUEUE_PREFIX": "hear.simulation",
            "HEAR_POD_MAX_CONCURRENT_JOBS": "1",
            "HEAR_HOST_MAX_CONCURRENT_JOBS": "2",
            "HEAR_HOST_JOB_LOCK_PATH": str(root / "admission.lock"),
            "HEAR_POD_STACK_ROLES": "reconstruction,pipeline,transcription,magic_clean_natural",
            "HEAR_IMAGE_REVISION": "evaluation-current",
            "HEAR_ENGINE_REVISION": "deepfilter-fish-nf4-qwen",
            "HEAR_ENABLE_DOCS": "true",
            "HEAR_GATEWAY_PORT": "8000",
            "HEAR_GATEWAY_HOST": "0.0.0.0",
            "SSL_CERT_FILE": str(root / "tls/cert.pem"),
            "REQUESTS_CA_BUNDLE": str(root / "tls/cert.pem"),
            "AWS_CA_BUNDLE": str(root / "tls/cert.pem"),
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
            "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
            "AWS_DEFAULT_REGION": "us-east-1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "XDG_CACHE_HOME": "/root/.cache",
            "HF_HOME": "/root/.cache/huggingface",
            "TORCH_HOME": "/root/.cache/torch",
            "UV_CACHE_DIR": "/root/.cache/uv",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "2",
            "WHISPER_BATCH_SIZE": "2",
            "WHISPER_LONG_AUDIO_BATCH_SIZE": "1",
            "HEAR_GATEWAY_PYTHON_BIN": "/opt/hear-ai-v11/venvs/test/bin/python",
        }
        env_file = root / "runtime.env"
        env_file.write_text("".join(k + "=" + "'" + v + "'" + "\n" for k, v in env.items()))
        env_file.chmod(0o600)
        print("Configured local simulation:", root)
        print("No production backend or storage credentials were read.")


if __name__ == "__main__":
    SetupSimulation.main()

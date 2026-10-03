"""Create isolated local test settings without reading production credentials."""

import argparse
import hashlib
import ipaddress
import json
import secrets
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


class SetupSimulation:
    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, default=Path("/root/hear-ai-runtime/canary"))
        parser.add_argument("--audio-file", type=Path, required=True)
        parser.add_argument("--fish-reference", type=Path)
        args = parser.parse_args()
        root = args.root.resolve()
        if not args.audio_file.is_file():
            parser.error("audio-file must be an existing real audio recording")
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
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.audio_file),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(root / "sources/input.mp3"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.fish_reference or args.audio_file),
                "-ar",
                "44100",
                "-ac",
                "1",
                str(root / "sources/fish-reference.wav"),
            ],
            check=True,
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
            "FISH_SPEECH_HOME": "/opt/fish-speech",
            "FISH_SPEECH_MODEL_ROOT": "/models",
            "HEAR_MAGIC_CLEAN_MODEL_DEVICE": "cuda:0",
            "HEAR_RABBITMQ_URL": "amqp://guest:guest@127.0.0.1:5672/%2F",
            "HEAR_QUEUE_EXCHANGE": "hear.simulation.jobs",
            "HEAR_QUEUE_PREFIX": "hear.simulation",
            "HEAR_POD_MAX_CONCURRENT_JOBS": "1",
            "HEAR_HOST_MAX_CONCURRENT_JOBS": "10",
            "HEAR_HOST_JOB_LOCK_PATH": str(root / "admission.lock"),
            "HEAR_POD_STACK_ROLES": "pipeline,magic_clean_natural,reconstruction",
            "HEAR_POD_ROLE_LIMITS": '{"pipeline":7,"magic_clean_natural":4,"reconstruction":2}',
            "HEAR_POD_PROCESS_LIMITS": '{"pipeline":7,"magic_clean_natural":1,"reconstruction":1}',
            "HEAR_WORKER_REPLICAS": '{"pipeline":1,"magic_clean_natural":4,"reconstruction":2}',
            "HEAR_IMAGE_REVISION": "evaluation-current",
            "HEAR_ENGINE_REVISION": "deepfilter-fish-bf16-qwen",
            "HEAR_ENABLE_DOCS": "true",
            "HEAR_GATEWAY_PORT": "8000",
            "HEAR_GATEWAY_HOST": "127.0.0.1",
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
            "HEAR_GPU_IDLE_EVICTION_ENABLED": "true",
            "HEAR_PIPELINE_IDLE_TTL_SECONDS": "60",
            "HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS": "60",
            "HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS": "60",
            "HEAR_AUDIOSEP_IDLE_TTL_SECONDS": "30",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "2",
            "WHISPER_BATCH_SIZE": "8",
            "WHISPER_LONG_AUDIO_BATCH_SIZE": "8",
            "WHISPER_CHUNK_SECONDS": "240",
            "HEAR_GATEWAY_PYTHON_BIN": "/opt/hear-ai-v11/venvs/pipeline/bin/python",
        }
        release = Path("/models/sound-cleanup-release.env")
        if release.is_file():
            env.update(
                HEAR_SOUND_CLEANUP_BUNDLE="/models/sound-cleanup-v1-runtime",
                HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE="/models/sound-cleanup-specialist/runtime",
            )
            for line in release.read_text().splitlines():
                key, value = line.split("=", 1)
                env[key] = value
        env_file = root / "runtime.env"
        env_file.write_text("".join(k + "=" + "'" + v + "'" + "\n" for k, v in env.items()))
        env_file.chmod(0o600)
        print("Configured local simulation:", root)
        print("No production backend or storage credentials were read.")


if __name__ == "__main__":
    SetupSimulation.main()

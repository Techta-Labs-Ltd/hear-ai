import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hear.runtime.cleaner.model_registry import CertifiedRuntime
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    total_bytes: int
    used_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        if self.total_bytes <= 0 or min(self.used_bytes, self.free_bytes) < 0:
            raise ValueError("invalid GPU memory snapshot")
        if self.used_bytes > self.total_bytes or self.free_bytes > self.total_bytes:
            raise ValueError("GPU memory snapshot exceeds device total")


class DeviceMemoryProbe(Protocol):
    def snapshot(self, device_index: int) -> DeviceMemorySnapshot: ...


class NvidiaSmiMemoryProbe:
    MIB = 1024**2

    def __init__(
        self,
        executable: Path = Path("/usr/bin/nvidia-smi"),
        *,
        timeout_seconds: float = 2.0,
    ):
        if not executable.is_absolute():
            raise ValueError("nvidia-smi path must be absolute")
        if not 0 < timeout_seconds <= 10:
            raise ValueError("nvidia-smi timeout must be between 0 and 10 seconds")
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def snapshot(self, device_index: int) -> DeviceMemorySnapshot:
        if not isinstance(device_index, int) or device_index < 0:
            raise ValueError("GPU device index must be a non-negative integer")
        command = [
            str(self.executable),
            f"--id={device_index}",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe unavailable"
            ) from None
        if result.returncode != 0:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe failed")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe returned invalid output"
            )
        try:
            values = [int(value.strip()) for value in lines[0].split(",")]
        except ValueError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe returned invalid output"
            ) from None
        if len(values) != 3:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe returned invalid output"
            )
        try:
            return DeviceMemorySnapshot(*(value * self.MIB for value in values))
        except ValueError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "GPU memory probe returned invalid values"
            ) from None


class GpuAdmissionController:
    def __init__(
        self,
        probe: DeviceMemoryProbe,
        *,
        device_index: int = 0,
        safety_reserve_bytes: int = 2_000_000_000,
    ):
        if not isinstance(device_index, int) or device_index < 0:
            raise ValueError("GPU device index must be a non-negative integer")
        if not isinstance(safety_reserve_bytes, int) or safety_reserve_bytes < 0:
            raise ValueError("GPU safety reserve is invalid")
        self.probe = probe
        self.device_index = device_index
        self.safety_reserve_bytes = safety_reserve_bytes

    def admit(self, runtime: CertifiedRuntime) -> DeviceMemorySnapshot:
        if runtime.lane == "cpu":
            raise ValueError("CPU runtime cannot use GPU admission")
        required = runtime.certified_peak_device_bytes
        if required <= 0:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "GPU runtime has no certified device-memory peak",
            )
        snapshot = self.probe.snapshot(self.device_index)
        if snapshot.free_bytes < required + self.safety_reserve_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED,
                "insufficient device-wide GPU memory for cleaner runtime",
            )
        return snapshot

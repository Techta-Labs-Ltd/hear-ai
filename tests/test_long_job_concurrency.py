import hashlib
import json

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from hear.runtime.concurrency import RoleConcurrency
from hear.runtime.host_admission import HostJobAdmission
from hear.runtime.roles import WorkerRole
from scripts.simulation_multipart import SimulationMultipart


def test_ten_pipeline_jobs_share_safe_process_and_global_budget():
    env = {
        "HEAR_POD_PROCESS_LIMITS": '{"pipeline":10}',
        "HEAR_POD_ROLE_LIMITS": '{"pipeline":10,"magic_clean_natural":10}',
    }
    pipeline = RoleConcurrency.load(WorkerRole.PIPELINE, 1, 10, env)
    cleaner = RoleConcurrency.load(WorkerRole.MAGIC_CLEAN_NATURAL, 1, 10, env)
    assert (pipeline.process_jobs, pipeline.role_jobs) == (10, 10)
    assert (cleaner.process_jobs, cleaner.role_jobs) == (1, 10)


@pytest.mark.parametrize("value", ['{"pipeline":true}', '{"pipeline":17}', '{"other":2}'])
def test_invalid_limit_map_rejected(value):
    with pytest.raises(ValueError):
        RoleConcurrency.limits(value)


def test_shared_cleaning_session_cannot_be_multithreaded():
    with pytest.raises(ValueError, match="one_job_per_process"):
        RoleConcurrency.load(WorkerRole.MAGIC_CLEAN_NATURAL, 10, 10, {})


def test_ten_role_slots_do_not_allow_an_eleventh_job(tmp_path):
    admission = HostJobAdmission(tmp_path / "lock", 10, "pipeline", 10)
    permits = [admission.try_acquire() for _ in range(10)]
    try:
        assert all(permits)
        assert admission.try_acquire() is None
        cleaner = HostJobAdmission(tmp_path / "lock", 10, "magic_clean_natural", 10)
        assert cleaner.try_acquire() is None
    finally:
        for permit in permits:
            if permit:
                permit.close()


def test_multipart_bytes_and_metadata_are_verified(tmp_path):
    (tmp_path / "metadata").mkdir()
    store = SimulationMultipart(tmp_path)
    app = FastAPI()
    target = tmp_path / "objects" / "audio.flac"

    @app.api_route("/object", methods=["POST", "PUT", "DELETE"])
    async def handle(request: Request):
        return await store.handle(request, "test-bucket", "audio.flac", target)

    client = TestClient(app)
    first, second = b"audio-part-one", b"audio-part-two"
    digest = hashlib.sha256(first + second).hexdigest()
    response = client.post(
        "/object?uploads", headers={"x-amz-meta-sha256": digest, "content-type": "audio/flac"}
    )
    import xml.etree.ElementTree as ET

    identity = ET.fromstring(response.content).findtext("{*}UploadId")
    parts = []
    for number, data in enumerate((first, second), 1):
        response = client.put(f"/object?uploadId={identity}&partNumber={number}", content=data)
        assert response.status_code == 200
        parts.append(
            f"<Part><PartNumber>{number}</PartNumber><ETag>{response.headers['etag']}</ETag></Part>"
        )
    body = "<CompleteMultipartUpload>" + "".join(parts) + "</CompleteMultipartUpload>"
    response = client.post(f"/object?uploadId={identity}", content=body)
    assert response.status_code == 200 and target.read_bytes() == first + second
    record = json.loads(next((tmp_path / "metadata").glob("*.json")).read_text())
    assert record["sha256"] == digest and record["size"] == len(first + second)
    assert not (tmp_path / "multipart" / identity).exists()


@pytest.mark.parametrize(
    "role,expected",
    [
        (WorkerRole.PIPELINE, (7, 7)),
        (WorkerRole.MAGIC_CLEAN_NATURAL, (1, 4)),
        (WorkerRole.RECONSTRUCTION, (1, 2)),
        (WorkerRole.TRANSCRIPTION, (1, 1)),
    ],
)
def test_requested_seven_four_two_limits(role, expected):
    environment = {
        "HEAR_POD_PROCESS_LIMITS": '{"pipeline":7,"magic_clean_natural":1,"reconstruction":1,"transcription":1}',
        "HEAR_POD_ROLE_LIMITS": '{"pipeline":7,"magic_clean_natural":4,"reconstruction":2,"transcription":1}',
    }
    result = RoleConcurrency.load(role, 1, 10, environment)
    assert (result.process_jobs, result.role_jobs) == expected


def test_fish_two_jobs_require_two_processes():
    with pytest.raises(ValueError, match="one_job_per_process"):
        RoleConcurrency.load(
            WorkerRole.RECONSTRUCTION,
            1,
            10,
            {
                "HEAR_POD_PROCESS_LIMITS": '{"reconstruction":2}',
                "HEAR_POD_ROLE_LIMITS": '{"reconstruction":2}',
            },
        )


@pytest.mark.parametrize(
    "role,limit", [("pipeline", 7), ("magic_clean_natural", 4), ("reconstruction", 2)]
)
def test_new_role_ceilings_reject_the_next_job(tmp_path, role, limit):
    admission = HostJobAdmission(tmp_path / "revised.lock", 10, role, limit)
    permits = [admission.try_acquire() for _ in range(limit)]
    try:
        assert all(permits)
        assert admission.try_acquire() is None
    finally:
        for permit in permits:
            if permit:
                permit.close()

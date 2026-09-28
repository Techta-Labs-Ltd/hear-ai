"""Bounded local S3 multipart emulation for long-file canaries, not cloud storage."""

import asyncio
import hashlib
import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import HTTPException, Request, Response


class SimulationMultipart:
    def __init__(self, root: Path):
        self.root = root
        self.uploads = root / "multipart"
        self.uploads.mkdir(exist_ok=True)

    @staticmethod
    def xml(name: str, values: dict[str, str]) -> bytes:
        element = ET.Element(name, xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
        for key, value in values.items():
            ET.SubElement(element, key).text = value
        return ET.tostring(element, encoding="utf-8", xml_declaration=True)

    async def handle(self, request: Request, bucket: str, key: str, target: Path):
        if request.method == "POST" and "uploads" in request.query_params:
            identity = uuid.uuid4().hex
            folder = self.uploads / identity
            folder.mkdir()
            metadata = {
                "key": key,
                "bucket": bucket,
                "content_type": request.headers.get("content-type", "application/octet-stream"),
                "sha256": request.headers.get("x-amz-meta-sha256", ""),
            }
            (folder / "metadata.json").write_text(json.dumps(metadata))
            return Response(
                self.xml(
                    "InitiateMultipartUploadResult",
                    {"Bucket": bucket, "Key": key, "UploadId": identity},
                ),
                media_type="application/xml",
            )
        identity = request.query_params.get("uploadId", "")
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise HTTPException(400, "invalid_simulation_upload_id")
        folder = self.uploads / identity
        if not (folder / "metadata.json").is_file():
            raise HTTPException(404, "multipart_upload_not_found")
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["key"] != key or metadata["bucket"] != bucket:
            raise HTTPException(403, "multipart_scope_mismatch")
        if request.method == "DELETE":
            shutil.rmtree(folder)
            return Response(status_code=204)
        if request.method == "PUT":
            number = request.query_params.get("partNumber", "")
            if not number.isdigit() or not 1 <= int(number) <= 128:
                raise HTTPException(400, "invalid_part_number")
            part = folder / f"{int(number):04d}.part"
            digest = hashlib.md5(usedforsecurity=False)
            count = 0
            with part.open("wb") as stream:
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > 64 * 1024**2:
                        raise HTTPException(413, "multipart_part_limit")
                    stream.write(chunk)
                    digest.update(chunk)
            return Response(headers={"ETag": '"' + digest.hexdigest() + '"'})
        if request.method == "POST":
            body = await request.body()
            if len(body) > 64 * 1024:
                raise HTTPException(413, "multipart_manifest_limit")
            return await asyncio.to_thread(self.complete, folder, target, metadata, body)
        raise HTTPException(405, "unsupported_multipart_method")

    def complete(self, folder: Path, target: Path, metadata: dict, body: bytes):
        try:
            tree = ET.fromstring(body)
            rows = [
                (int(p.findtext("{*}PartNumber", "0")), p.findtext("{*}ETag", "").strip('"'))
                for p in tree.findall("{*}Part")
            ]
        except (ET.ParseError, ValueError) as exc:
            raise HTTPException(400, "invalid_multipart_manifest") from exc
        if not rows or [n for n, _ in rows] != list(range(1, len(rows) + 1)):
            raise HTTPException(400, "multipart_sequence_invalid")
        staged = folder / "assembled"
        total = 0
        with staged.open("xb") as output:
            for number, expected in rows:
                part = folder / f"{number:04d}.part"
                digest = hashlib.md5(usedforsecurity=False)
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        total += len(chunk)
                        if total > 2 * 1024**3:
                            raise HTTPException(413, "multipart_object_limit")
                        output.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise HTTPException(422, "multipart_part_hash_mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
        record = {
            "size": total,
            "content_type": metadata["content_type"],
            "sha256": metadata["sha256"],
        }
        meta = (
            self.root
            / "metadata"
            / (hashlib.sha256(metadata["key"].encode()).hexdigest() + ".json")
        )
        meta.write_text(json.dumps(record))
        combined = hashlib.md5(
            b"".join(bytes.fromhex(tag) for _, tag in rows), usedforsecurity=False
        ).hexdigest()
        result = self.xml(
            "CompleteMultipartUploadResult",
            {
                "Bucket": metadata["bucket"],
                "Key": metadata["key"],
                "ETag": f'"{combined}-{len(rows)}"',
            },
        )
        shutil.rmtree(folder)
        return Response(result, media_type="application/xml")

import hashlib
import os
import tempfile
from pathlib import Path

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.artifacts import (
    ArtifactWriter,
    ManifestVerifier,
    PublishedBundle,
    StoredObject,
)
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError, ErrorCode


class S3BundleVerifier:
    def __init__(self, client, bucket: str):
        if not bucket:
            raise ValueError("a scoped bucket is required")
        self.client = client
        self.bucket = bucket

    def verify(
        self, ticket: AttemptTicket, reference: StoredObject, guard: ResourceGuard
    ) -> PublishedBundle:
        guard.check()
        if reference.key != ticket.manifest_key or not reference.version:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "terminal reference mismatch")
        if not 0 < reference.size_bytes <= ArtifactWriter.MAX_MANIFEST_BYTES:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "manifest size limit exceeded")
        payload = self._read(reference, "application/json", guard, collect=True)
        manifest = ManifestVerifier.verify(payload, ticket, reference.sha256)
        for artifact in manifest.artifacts:
            self._read(
                StoredObject(
                    artifact.object_key,
                    artifact.object_version,
                    artifact.sha256,
                    artifact.size_bytes,
                ),
                artifact.content_type,
                guard,
                collect=False,
            )
        guard.check()
        return PublishedBundle(manifest, reference)

    def _read(self, reference, content_type, guard, *, collect, sink=None):
        guard.check()
        if not 0 < reference.size_bytes <= guard.budget.scratch_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "object verification limit exceeded"
            )
        body = None
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=reference.key,
                VersionId=reference.version,
            )
            body = response["Body"]
            guard.check()
            if (
                response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200
                or str(
                    response.get("ResponseMetadata", {})
                    .get("HTTPHeaders", {})
                    .get("x-backblaze-live-read-enabled", "false")
                ).lower()
                == "true"
                or "ContentRange" in response
                or response.get("VersionId") != reference.version
                or response.get("ContentLength") != reference.size_bytes
                or (content_type is not None and response.get("ContentType") != content_type)
            ):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "remote object identity mismatch"
                )
            digest = hashlib.sha256()
            size = 0
            chunks = []
            while True:
                guard.check()
                chunk = body.read(min(1024 * 1024, reference.size_bytes - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > reference.size_bytes:
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "remote object length mismatch"
                    )
                digest.update(chunk)
                if sink is not None:
                    sink.write(chunk)
                if collect:
                    chunks.append(chunk)
            guard.check()
            if size != reference.size_bytes or digest.hexdigest() != reference.sha256:
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "remote object checksum mismatch"
                )
            return b"".join(chunks) if collect else b""
        except CleanExecutionError:
            raise
        except Exception as exc:
            raise CleanExecutionError(
                ErrorCode.STORAGE_FAILED, "versioned object verification failed"
            ) from exc
        finally:
            if body is not None:
                try:
                    body.close()
                except OSError:
                    pass


class S3SourceStager:
    """Stage only an authenticated ticket's exact source; no arbitrary URL fetch.

    SourceInspector must still validate the decoded audio after staging. Source
    MIME metadata is not trusted as a format declaration. Credentials, bucket
    scope and endpoint allowlisting belong to the authenticated ingress client.
    """

    def __init__(self, client, bucket: str):
        self.reader = S3BundleVerifier(client, bucket)

    def stage(self, ticket: AttemptTicket, destination: Path, guard: ResourceGuard) -> Path:
        guard.bind_deadline(ticket.deadline)
        guard.check()
        if destination.is_symlink() or not destination.resolve().is_relative_to(
            guard.workspace.resolve()
        ):
            raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "source path outside workspace")
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "source destination exists")
        source = ticket.input
        occupied = sum(path.stat().st_size for path in guard.workspace.rglob("*") if path.is_file())
        if (
            source.size_bytes > guard.budget.max_input_bytes
            or occupied + source.size_bytes > guard.budget.scratch_bytes
        ):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "source exceeds download budget"
            )
        reference = StoredObject(
            source.object_key, source.object_version, source.sha256, source.size_bytes
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix="source-stage-", dir=guard.workspace
            ) as temporary:
                self.reader._read(reference, None, guard, collect=False, sink=temporary)
                temporary.flush()
                guard.check()
                # Same-filesystem hard link makes a verified source visible
                # atomically and refuses concurrent destination replacement.
                os.link(temporary.name, destination)
        except FileExistsError as exc:
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "source destination exists"
            ) from exc
        except CleanExecutionError as exc:
            if exc.code == ErrorCode.ARTIFACT_CONFLICT:
                raise CleanExecutionError(
                    ErrorCode.SOURCE_MISMATCH, "pinned source mismatch"
                ) from exc
            raise
        except OSError as exc:
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "source staging failed") from exc
        return destination

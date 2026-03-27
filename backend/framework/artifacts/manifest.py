"""
Manifest I/O helpers for Pitch Accelerator node runs.

Usage
-----
    run_id  = generate_run_id()
    started = utc_now()

    # ... do work, collect NodeArtifact objects ...

    manifest = RunManifest(
        run_id=run_id, node_id="cep_sim_uk", node_type="cep_sim",
        started_at=started, artifacts=[art1, art2, ...],
        config_summary={"market": "UK", "respondents": 1315},
    )
    path = write_manifest(manifest, out_dir)
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.framework.schemas.artifact import NodeArtifact, RunManifest


def generate_run_id() -> str:
    """Return a fresh UUID-based run identifier."""
    return str(uuid.uuid4())


def utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_artifact(
    *,
    artifact_type,
    run_id: str,
    node_id: str,
    node_type,
    title: str,
    description: str,
    storage_path: str | Path,
    preview_type,
    row_count: int | None = None,
    metadata: dict | None = None,
) -> NodeArtifact:
    """Convenience constructor that fills file_size_bytes automatically."""
    p = Path(storage_path)
    size = p.stat().st_size if p.exists() else None
    return NodeArtifact(
        artifact_type=artifact_type,
        run_id=run_id,
        node_id=node_id,
        node_type=node_type,
        title=title,
        description=description,
        storage_path=str(storage_path),
        preview_type=preview_type,
        row_count=row_count,
        file_size_bytes=size,
        metadata=metadata or {},
    )


def write_manifest(manifest: RunManifest, out_dir: str | Path) -> Path:
    """
    Serialise a RunManifest to ``run_manifest.json`` inside *out_dir*.

    Returns the path to the written file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    p = out_path / "run_manifest.json"
    p.write_text(manifest.model_dump_json(indent=2))
    return p


def read_manifest(path: str | Path) -> RunManifest:
    """Load a RunManifest from a ``run_manifest.json`` file."""
    return RunManifest.model_validate_json(Path(path).read_text())


def artifacts_by_type(manifest: RunManifest, artifact_type: str) -> list[NodeArtifact]:
    """Filter a manifest's artifacts by type."""
    return [a for a in manifest.artifacts if a.artifact_type == artifact_type]

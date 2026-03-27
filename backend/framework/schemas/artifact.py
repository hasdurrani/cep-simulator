"""
Shared artifact contract for all Pitch Accelerator analysis nodes.

Every file a node produces is wrapped in a NodeArtifact.
Every run produces a RunManifest that lists all its artifacts.

The manifest is the contract between a node and the frontend / downstream nodes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


ArtifactType = Literal[
    # cep_sim
    "memory_graph",
    "scenario_recall",
    "ad_impact",
    "flight_simulator",
    "segment_summary",
    "model_diagnostics",
    "summary_bundle",
    # purchase_signal
    "audience_profile",
    "feature_comparison",
    "demographics",
    "anchor_timeline",
    "statsocial",
    "run_metadata",
    # shared / generic
    "chart",
    "report",
]

PreviewType = Literal[
    "heatmap",
    "leaderboard",
    "bar_chart",
    "scatter",
    "table",
    "json",
    "markdown",
    "image",
]

NodeType = Literal["cep_sim", "purchase_signal"]


class NodeArtifact(BaseModel):
    """Typed envelope for a single output file produced by a node run."""

    artifact_type:  ArtifactType
    schema_version: str = "1.0"
    run_id:         str
    node_id:        str
    node_type:      NodeType
    title:          str
    description:    str
    storage_path:   str            # relative to project root
    preview_type:   PreviewType
    row_count:      int | None = None
    file_size_bytes: int | None = None
    metadata:       dict[str, Any] = Field(default_factory=dict)
    generated_at:   str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RunManifest(BaseModel):
    """Complete record of a single node execution and everything it produced."""

    run_id:       str
    node_id:      str
    node_type:    NodeType
    status:       Literal["success", "partial", "failed"] = "success"
    started_at:   str
    completed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts:    list[NodeArtifact] = Field(default_factory=list)

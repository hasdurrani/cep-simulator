import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class RunMetadata:
    run_name: str
    output_schema: str


@dataclass
class RunConfigBase:

    run: RunMetadata

    purchase_signal: Optional[Dict[str, Any]] = None

    duplication_of_purchase: Optional[Dict[str, Any]] = None


def load_run_config(path: str) -> RunConfigBase:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    run_section = data.get("run", {})

    return RunConfigBase(
        run=RunMetadata(
            run_name=run_section["run_name"],
            output_schema=run_section["output_schema"],
        ),
        purchase_signal=data.get("purchase_signal"),
        duplication_of_purchase=data.get("duplication_of_purchase"),
    )

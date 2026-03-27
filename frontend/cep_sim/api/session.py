"""In-memory session store for CEP sim UI sessions."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class Session:
    session_id: str
    country: str
    config: object                      # CepSimConfig
    rbc_df: pd.DataFrame                # pre-ad memory graph
    cep_master_df: pd.DataFrame
    raw_map_df: pd.DataFrame
    long_df: pd.DataFrame
    scenario_recall_df: pd.DataFrame
    brand_name_map: dict[str, str]
    respondent_ids: list[str]
    respondents_df: pd.DataFrame
    brand_priors: dict | None = None
    responsiveness_map: dict | None = None
    brand_similarity: dict | None = None
    fitted_params: dict | None = None
    mae: float | None = None
    holdout_mae: float | None = None
    holdout_rho: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


_store: dict[str, Session] = {}
_lock  = threading.Lock()


def put(session: Session) -> None:
    with _lock:
        _store[session.session_id] = session


def get(session_id: str) -> Session | None:
    return _store.get(session_id)


def delete(session_id: str) -> None:
    with _lock:
        _store.pop(session_id, None)

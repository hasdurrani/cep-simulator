import logging
import pandas as pd
from pathlib import Path

from backend.schemas.config import CepSimConfig
from backend.schemas.ontology import CEPNode
from backend.service.utils import normalize_cep_text, cep_label_to_id

logger = logging.getLogger(__name__)

# Keyword → cep_family mapping (order matters — first match wins)
_BRAZIL_FAMILY_RULES: list[tuple[str, str]] = [
    ("bar",            "on_premise"),
    ("churrasco",      "social_occasion"),
    ("festival",       "social_occasion"),
    ("futebol",        "social_occasion"),
    ("casamento",      "social_occasion"),
    ("jantar",         "social_occasion"),
    ("amigos",         "social_occasion"),
    ("ao ar livre",    "outdoor"),
    ("quente",         "outdoor"),
    ("férias",         "lifestyle"),
    ("trabalho",       "lifestyle"),
    ("garrafa grande", "sharing"),
    ("impressão",      "premium"),
    ("especial",       "premium"),
]

_UK_FAMILY_RULES: list[tuple[str, str]] = [
    ("pub",             "on_premise"),
    ("bar",             "on_premise"),
    ("meal",            "on_premise"),
    ("gig",             "on_premise"),
    ("club",            "on_premise"),
    ("BBQ",             "outdoor"),
    ("outdoors",        "outdoor"),
    ("hot",             "outdoor"),
    ("sunny",           "outdoor"),
    ("sport",           "social_occasion"),
    ("dinner",          "social_occasion"),
    ("party",           "social_occasion"),
    ("co-workers",      "social_occasion"),
    ("special occasion","social_occasion"),
    ("wedding",         "social_occasion"),
    ("fun",             "social_occasion"),
    ("hosting",         "social_occasion"),
    ("home",            "social_occasion"),
    ("group",           "sharing"),
    ("holiday",         "lifestyle"),
    ("heading out",     "lifestyle"),
    ("quick drink",     "lifestyle"),
    ("long day",        "lifestyle"),
    ("easy to drink",   "functional"),
    ("pace yourself",   "functional"),
    ("different",       "discovery"),
]

# Default (backwards-compatible)
_FAMILY_RULES = _BRAZIL_FAMILY_RULES


def _infer_family(normalized: str, rules: list[tuple[str, str]] = _FAMILY_RULES) -> str:
    for keyword, family in rules:
        if keyword in normalized:
            return family
    return "general"


def _get_family_rules(country: str) -> list[tuple[str, str]]:
    if country.upper() == "UK":
        return _UK_FAMILY_RULES
    return _BRAZIL_FAMILY_RULES


def build_ontology(long_df: pd.DataFrame, config: CepSimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cep_master and cep_raw_to_master_map from unique raw CEP stems.
    Returns (cep_master_df, raw_map_df).
    """
    raw_stems = long_df["cep_raw"].unique()
    family_rules = _get_family_rules(config.survey.country)

    cep_nodes: list[CEPNode] = []
    map_records = []

    for idx, raw in enumerate(sorted(raw_stems), start=1):
        normalized = normalize_cep_text(raw)
        cep_id = cep_label_to_id(normalized, idx)
        family = _infer_family(normalized, family_rules)

        # Make a short readable label: title-case the normalized text, max 60 chars
        label = normalized.title()[:60]

        node = CEPNode(
            cep_id=cep_id,
            cep_family=family,
            cep_label=label,
            cep_description=raw,
            active=True,
        )
        cep_nodes.append(node)

        map_records.append({
            "raw_response":      raw,
            "normalized_text":   normalized,
            "cep_id":            cep_id,
            "method":            "rule_based",
            "confidence":        1.0,
        })

    cep_master_df = pd.DataFrame([n.model_dump() for n in cep_nodes])
    raw_map_df = pd.DataFrame(map_records)

    logger.info("Built ontology: %d CEPs across %d families",
                len(cep_master_df), cep_master_df["cep_family"].nunique())
    return cep_master_df, raw_map_df


def save_ontology(
    cep_master_df: pd.DataFrame,
    raw_map_df: pd.DataFrame,
    config: CepSimConfig,
) -> tuple[Path, Path]:
    out_dir = Path(config.output.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = out_dir / "cep_master.csv"
    map_path = out_dir / "cep_raw_to_master_map.csv"

    cep_master_df.to_csv(master_path, index=False)
    raw_map_df.to_csv(map_path, index=False)

    logger.info("Saved cep_master: %s", master_path)
    logger.info("Saved cep_raw_to_master_map: %s", map_path)
    return master_path, map_path

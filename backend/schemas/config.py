"""Config schema for CEP simulator — Dynata coded-variable format."""
import tomllib
from pathlib import Path
from pydantic import BaseModel, field_validator


class RecallConfig(BaseModel):
    cep_blocks: list[str]          # e.g. ["Q10", "Q11", ..., "Q20"]
    exclude_brands: list[str] = [] # e.g. ["Nenhuma das opções acima"]


class DemographicColumns(BaseModel):
    gender: str = "Q2"
    age: str = "Q3"
    income: str | None = "Q6"
    has_children: str | None = "Q7"


class SurveyConfig(BaseModel):
    zip_path: str
    data_file: str           # path inside zip to the .csv
    codebook_file: str       # path inside zip to the .txt codebook
    respondent_id_column: str = "CID"
    country: str = "Brazil"
    demographic_columns: DemographicColumns = DemographicColumns()
    recall: RecallConfig


class DefaultsConfig(BaseModel):
    assoc_strength_if_mentioned: float = 1.0
    base_usage_default: float = 0.2        # fallback when no brand prior available
    learning_rate: float = 0.1
    competition_penalty_weight: float = 0.05
    softmax_temperature: float = 1.0
    w_max: float = 5.0                     # saturation ceiling for ad updates
    new_edge_weight: float = 0.3           # friction for creating brand-new CEP edges
    base_prior_weight: float = 1.0         # scalar multiplier on brand awareness priors


class OutputConfig(BaseModel):
    processed_dir: str = "backend/data/processed"
    outputs_dir: str = "outputs/cep_sim"


class AdConfig(BaseModel):
    ad_id: str
    brand_id: str                        # e.g. "brand_heineken"
    brand_name: str                      # display name, e.g. "Heineken"
    focal_scenario: str                  # scenario slug used for output labelling
    focal_cep_keywords: list[str]        # substrings matched against cep_description
    secondary_cep_keywords: list[str] = []
    branding_clarity: float = 0.9
    attention_weight: float = 1.0
    channel: str = "digital"
    emotion: str = "positive"


class CepSimConfig(BaseModel):
    survey: SurveyConfig
    defaults: DefaultsConfig = DefaultsConfig()
    output: OutputConfig = OutputConfig()
    ad: AdConfig | None = None


def load_cep_sim_config(path: str) -> CepSimConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return CepSimConfig(**data)

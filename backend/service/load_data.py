"""Load raw survey data from Dynata zip export."""
import logging
import zipfile
import io
import pandas as pd
from backend.schemas.config import CepSimConfig

logger = logging.getLogger(__name__)


def load_survey(config: CepSimConfig) -> pd.DataFrame:
    """Load the raw survey CSV from the zip file. Returns all complete rows."""
    with zipfile.ZipFile(config.survey.zip_path) as z:
        with z.open(config.survey.data_file) as f:
            df = pd.read_csv(f)
    # All rows in this file are STATUS=2 (complete), but filter defensively
    if "STATUS" in df.columns:
        df = df[df["STATUS"] == 2].copy()
    df[config.survey.respondent_id_column] = df[config.survey.respondent_id_column].astype(str)
    logger.info("Loaded survey: %d rows from %s", len(df), config.survey.data_file)
    return df


def inspect_survey(df: pd.DataFrame, config: CepSimConfig) -> dict:
    """Return basic stats about the survey."""
    cep_cols = [
        c for c in df.columns
        if any(c.startswith(block + "_") for block in config.survey.recall.cep_blocks)
    ]
    return {
        "row_count": len(df),
        "recall_column_count": len(cep_cols),
        "cep_blocks": config.survey.recall.cep_blocks,
    }

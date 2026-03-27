"""
Dynata codebook parser.

Parses the variable-coded .txt codebook from a Dynata survey zip and returns
a DataFrame with one row per checkbox variable, containing:
  var_name, q_block, option_index, question_stem, brand_name

Also provides build_column_map() which returns a dict:
  {col_name: {"cep_block": "Q10", "cep_description": "...", "brand_name": "Heineken"}}
"""
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HTML_PAT = re.compile(r"<[^>]+>")
_VAR_PAT = re.compile(r"^\d+\.\s+(Q\w+)\s+\[(\w+)\]\s+(.*)")


def parse_codebook(zip_path: str, codebook_zip_path: str) -> pd.DataFrame:
    """
    Parse codebook txt from inside the zip.

    Returns a DataFrame with columns:
        var_name, q_type, q_block, option_index, question_stem, brand_name
    Only returns CHECKBOX rows.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(codebook_zip_path) as f:
            content = f.read().decode("utf-8", errors="replace")

    records = []
    for line in content.split("\n"):
        m = _VAR_PAT.match(line)
        if not m:
            continue
        var_name, q_type, text = m.groups()
        if q_type != "CHECKBOX":
            continue
        text = _HTML_PAT.sub("", text).strip()
        # Split on ' - ' (last occurrence) to get stem + brand
        idx = text.rfind(" - ")
        if idx == -1:
            continue
        stem = text[:idx].strip()
        brand = text[idx + 3:].strip()
        # Parse q_block and option_index from var_name like Q10_1
        m2 = re.match(r"(Q\d+)_(\d+)", var_name)
        if not m2:
            continue
        q_block, opt_idx = m2.groups()
        records.append({
            "var_name": var_name,
            "q_type": q_type,
            "q_block": q_block,
            "option_index": int(opt_idx),
            "question_stem": stem,
            "brand_name": brand,
        })
    return pd.DataFrame(records)


def build_column_map(
    codebook_df: pd.DataFrame,
    cep_blocks: list[str],
    exclude_brands: list[str] | None = None,
) -> dict[str, dict]:
    """
    Build a mapping from CSV column name to CEP/brand metadata.

    Returns:
        {
          "Q10_13": {
            "cep_block": "Q10",
            "cep_description": "Quando você está em um bar da moda...",
            "brand_name": "Heineken",
          },
          ...
        }
    Only includes columns in the requested cep_blocks and not in exclude_brands.
    """
    exclude = set(exclude_brands or [])
    col_map = {}
    for _, row in codebook_df.iterrows():
        if row["q_block"] not in cep_blocks:
            continue
        if row["brand_name"] in exclude:
            continue
        col_map[row["var_name"]] = {
            "cep_block": row["q_block"],
            "cep_description": row["question_stem"],
            "brand_name": row["brand_name"],
        }
    return col_map


def save_question_metadata(
    codebook_df: pd.DataFrame,
    cep_blocks: list[str],
    out_dir: str | Path,
) -> Path:
    """
    Export one row per CEP block (Q10, Q11, …) showing the question stem
    and the count of brands / options.

    Columns: q_block, question_stem, n_brands

    Saved to <out_dir>/question_metadata.csv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_df = codebook_df[codebook_df["q_block"].isin(cep_blocks)].copy()

    meta = (
        block_df.groupby("q_block")
        .agg(
            question_stem=("question_stem", "first"),
            n_brands=("brand_name", "nunique"),
        )
        .reset_index()
        .sort_values("q_block")
    )

    out_path = out_dir / "question_metadata.csv"
    meta.to_csv(out_path, index=False)
    logger.info("Saved question_metadata: %s (%d rows)", out_path, len(meta))
    return out_path

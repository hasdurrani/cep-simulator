import math
import re


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """
    Softmax with temperature scaling.

    temperature < 1  → sharper distribution (amplifies differences)
    temperature > 1  → flatter distribution (compresses differences)
    temperature = 1  → standard softmax
    """
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def parse_mention(value) -> int:
    """Convert a survey cell value to binary 1/0."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return 1 if value > 0 else 0
    s = str(value).strip().lower()
    if s in ("yes", "1", "true", "checked", "selected", "x"):
        return 1
    return 0


def brand_to_id(brand_name: str) -> str:
    """Convert a brand name to a stable snake_case id."""
    s = brand_name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return f"brand_{s}"


def normalize_cep_text(raw: str) -> str:
    """
    Strip survey boilerplate from a CEP question stem and return the
    core need-state phrase.
    """
    text = raw.strip().lower()

    boilerplate = [
        r"^when you need\s+",
        r"^when you think of\s+",
        r"^when you want to\s+",
        r",?\s*which brands come to mind\??$",
        r"\?$",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text).strip()

    # Collapse whitespace and strip punctuation at boundaries
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    return text


def cep_label_to_id(label: str, index: int) -> str:
    """Generate a stable cep_id like cep_001 from a label and its index."""
    return f"cep_{index:03d}"

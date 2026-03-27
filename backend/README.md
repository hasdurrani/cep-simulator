# cep_sim — CEP Memory Simulator

Simulates how advertising shifts brand recall probabilities across purchase occasions (Category Entry Points). Built on a real-world Dynata survey of Brazilian beer drinkers (N=1,325).

---

## What it does

1. **Parses** a Dynata coded-variable survey (zip export) using its codebook
2. **Builds** a memory model: each respondent × brand × CEP triple gets an association strength derived from their survey mentions
3. **Scores** brand recall probabilities for any purchase occasion using an additive model (semantic + base prior − competition)
4. **Simulates** ad exposure by updating association strengths and measuring the resulting shift in recall probabilities
5. **Validates** predictions against observed mention rates

---

## Module layout

```
backend/
├── data/
│   ├── raw/          # Dynata zip file (not committed)
│   └── processed/    # long_survey.csv (generated)
├── docs/
│   └── model_spec.md # Formal mathematical specification
├── notebooks/
│   └── cep_sim_mvp.ipynb
├── schemas/
│   ├── config.py     # Pydantic models + load_cep_sim_config()
│   ├── ontology.py   # CEPNode
│   ├── respondent.py # Respondent, RespondentBrandCEP
│   └── events.py     # EpisodicEvent
├── service/
│   ├── codebook_parser.py   # Parse Dynata .txt codebook → column map
│   ├── load_data.py         # Load CSV from zip
│   ├── reshape_survey.py    # Wide coded → long respondent-brand-CEP
│   ├── ontology_builder.py  # CEP deduplication + family inference
│   ├── respondent_builder.py# Memory edge table + respondent demographics
│   ├── recall_engine.py     # Scoring, softmax, SCENARIOS library
│   ├── ad_engine.py         # Ad exposure update rule
│   ├── utils.py             # softmax, brand_to_id, normalize_cep_text
│   └── validator.py         # Calibration, Spearman validity, sanity checks
├── framework/
│   └── artifacts/manifest.py # Artifact writing + run manifests
├── configs/          # TOML configs per market
└── tests/
    ├── test_recall_engine.py
    ├── test_ad_engine.py
    └── test_calibration.py
```

---

## Quickstart

```python
from backend.schemas.config import load_cep_sim_config
from backend.service.load_data import load_survey
from backend.service.reshape_survey import reshape_wide_to_long
from backend.service.ontology_builder import build_ontology
from backend.service.respondent_builder import build_respondents, build_respondent_brand_cep
from backend.service.recall_engine import get_recall_probs, rank_brands, SCENARIOS
from backend.service.ad_engine import Ad, apply_ad_to_population
from backend.service.validator import run_scenario_recall, run_ad_impact, run_calibration_check

config = load_cep_sim_config("backend/configs/cep_sim_config.toml")

df       = load_survey(config)
long_df  = reshape_wide_to_long(df, config)
cep_master_df, raw_map_df = build_ontology(long_df, config)
respondents_df = build_respondents(df, config)
rbc_df   = build_respondent_brand_cep(long_df, raw_map_df, config)

brand_name_map  = rbc_df[["brand_id","brand_name"]].drop_duplicates().set_index("brand_id")["brand_name"].to_dict()
respondent_ids  = respondents_df["respondent_id"].astype(str).tolist()

# Baseline recall across all scenarios
scenario_recall_df = run_scenario_recall(respondent_ids, SCENARIOS, rbc_df, cep_master_df, brand_name_map, config)

# Calibration: predicted recall prob vs. observed mention rate
cal_df = run_calibration_check(scenario_recall_df, long_df)
print(f"MAE: {cal_df.attrs['mae']:.4f}")

# Ad simulation
ad = Ad(
    ad_id="test_ad",
    brand_id="brand_heineken",
    brand_name="Heineken",
    focal_ceps=["cep_001"],
    secondary_ceps=["cep_002"],
    branding_clarity=0.9,
)
rbc_post, events = apply_ad_to_population(respondent_ids, ad, rbc_df, config)
impact_df = run_ad_impact(respondent_ids, SCENARIOS, rbc_df, rbc_post, cep_master_df, brand_name_map, config)
```

The full annotated walkthrough is in [notebooks/cep_sim_mvp.ipynb](notebooks/cep_sim_mvp.ipynb).

---

## Survey format

The simulator is built around Dynata's coded-variable export format:

- Column names are coded (`Q10_1`, `Q10_2`, …) — never descriptive
- A `.txt` codebook inside the same zip maps variable names to question text and option labels
- CEP recall blocks are checkbox grids: one column per brand, value 1 = mentioned, 0 = not mentioned

`codebook_parser.py` converts the codebook into a column map before any reshaping happens. The config (`cep_sim_config.toml`) points at the zip and names the inner files explicitly.

---

## Scoring model

```
score(r, b, S) = Σ_{c∈S} w(r,b,c)  +  β  −  γ · (|B_S| − 1)

P(r recalls b | S) = softmax_τ(score(r, b, S))
```

| Symbol | Meaning | Default |
|---|---|---|
| `w(r,b,c)` | Association strength of respondent r for brand b at CEP c | 1.0 if mentioned in survey |
| `β` | Uniform base prior | 0.2 |
| `γ` | Per-competitor penalty | 0.05 |
| `τ` | Softmax temperature | 1.0 |

See [docs/model_spec.md](docs/model_spec.md) for full mathematical specification, assumptions, and known limitations.

---

## Ad update rule

```
w_new(r, b, c) = w_old(r, b, c) + λ · e · δ · φ(c)
```

| Symbol | Meaning | Default |
|---|---|---|
| `λ` | Learning rate | 0.1 |
| `e` | Exposure strength | 1.0 |
| `δ` | Branding clarity (how clearly the ad links brand to occasion) | Ad-level |
| `φ(c)` | CEP fit: 1.0 focal, 0.5 secondary, 0.0 otherwise | Hardcoded |

---

## Configuration

All parameters are in `backend/configs/cep_sim_config.toml`. Key sections:

```toml
[survey]
zip_path      = "..."  # path to Dynata zip
data_file     = "..."  # CSV path inside zip
codebook_file = "..."  # .txt codebook path inside zip

[survey.recall]
cep_blocks     = ["Q10", ..., "Q20"]
exclude_brands = ["Nenhuma das opções acima"]

[defaults]
base_usage_default         = 0.2
learning_rate              = 0.1
competition_penalty_weight = 0.05
softmax_temperature        = 1.0
```

---

## Running tests

```bash
pytest backend/tests/ -v
```

Tests use synthetic fixtures (3 respondents, 3 brands, 2 CEPs) and do not require the survey data file.

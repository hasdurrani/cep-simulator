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
│   ├── codebook_parser.py    # Parse Dynata .txt codebook → column map
│   ├── load_data.py          # Load CSV from zip
│   ├── reshape_survey.py     # Wide coded → long respondent-brand-CEP
│   ├── ontology_builder.py   # CEP deduplication + family inference
│   ├── respondent_builder.py # Memory edge table + respondent demographics
│   ├── recall_engine.py      # Scoring + softmax
│   ├── ad_engine.py          # Ad exposure update rule
│   ├── calibration.py        # Parameter fitting + hold-out validation
│   ├── validator.py          # Calibration checks, Spearman validity, sanity checks
│   ├── scenario_library.py   # Purchase occasion scenario definitions
│   ├── runner.py             # End-to-end pipeline orchestration
│   ├── plotting.py           # Visualisation utilities
│   ├── output_builder.py     # Artifact writing
│   └── utils.py              # Shared utilities (softmax, normalisation, etc.)
├── framework/
│   └── artifacts/manifest.py # Run manifests
├── configs/          # TOML configs per market
└── tests/            # 65 unit tests across 5 modules (no data file required)
    ├── test_recall_engine.py
    ├── test_ad_engine.py
    ├── test_calibration.py
    ├── test_validator.py
    └── test_respondent_builder.py
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

config = load_cep_sim_config("backend/configs/cep_sim_config_uk.toml")

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

`codebook_parser.py` converts the codebook into a column map before any reshaping happens. The config (`cep_sim_config_uk.toml`) points at the zip and names the inner files explicitly.

---

## Scoring model

```
score(r, b, S) = Σ_{c∈S} w(r,b,c)  +  β(b)  −  γ · Σ_{b′≠b} sim(b,b′) · Σ_{c∈S} w(r,b′,c)

P(r recalls b | S) = softmax_τ(score(r, b, S))
```

| Symbol | Meaning | How set |
|---|---|---|
| `w(r,b,c)` | Association strength of respondent r for brand b at CEP c | breadth-scaled from survey |
| `β(b)` | Brand-specific awareness prior | fitted from population mention rates |
| `sim(b,b′)` | CEP-profile cosine similarity between brands b and b′ | computed from survey |
| `γ` | Competition weight | grid-search fitted |
| `τ` | Softmax temperature (lower = sharper distribution) | grid-search fitted |

See [docs/model_spec.md](docs/model_spec.md) for full mathematical specification, assumptions, and known limitations.

---

## Ad update rule

```
w_new(r, b, c) = w_old(r, b, c) + λ · ρ(r) · e · δ · φ(c) · (1 − w_old / w_max)
```

| Symbol | Meaning | How set |
|---|---|---|
| `λ` | Base learning rate | config (default 0.1) |
| `ρ(r)` | Per-respondent responsiveness multiplier | computed from survey engagement |
| `e` | Exposure strength | ad-level |
| `δ` | Branding clarity (how clearly the ad links brand to occasion) | set in UI |
| `φ(c)` | CEP fit: 1.0 focal, 0.5 secondary, 0.0 otherwise | fixed |
| `w_max` | Saturation ceiling for association weights | config (default 5.0) |

New edges (brand–CEP pairs with no prior survey association) receive an additional friction factor (`new_edge_weight = 0.3`), reflecting the greater cognitive cost of forming a new association vs. reinforcing an existing one.

---

## Configuration

All parameters are in `backend/configs/cep_sim_config_uk.toml`. Key sections:

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

# CEP Simulator

Simulate how advertising shifts brand recall across purchase occasions — an interactive tool for modelling mental availability using real survey data.

Built with FastAPI + vanilla React. No build step required.

---

## Theoretical grounding

This tool is built on the **mental availability** framework from the Ehrenberg-Bass Institute for Marketing Science (Byron Sharp, *How Brands Grow*, 2010).

Two EBI concepts are central:

- **Mental availability** — the probability that a buyer thinks of a brand in buying situations. EBI research establishes this as the primary driver of market share, built by creating and refreshing brand-relevant memory structures across a wide range of purchase occasions.
- **Category Entry Points (CEPs)** — the situational, social, and emotional cues that trigger category and brand retrieval at the moment of purchase. Brands with strong, broad CEP associations are recalled more often and by more buyers.

The mathematical model (scoring function, weight update rule, softmax) is a quantitative operationalisation of that framework — a formalisation authored here, not published by EBI.

---

## What it does

Advertising works by strengthening the mental link between a brand and the situations in which people buy. This simulator makes that effect concrete and measurable.

Given real survey data on which brands consumers associate with which purchase occasions, the tool lets you:

1. **See the baseline market** — which brands own which occasions, and by how much
2. **Design an ad scenario** — pick a brand, choose the occasions the ad will address, and set creative parameters
3. **Run the simulation** — apply the ad to the full population, re-score recall, and measure the competitive shift

The output is a recall probability distribution: *given that a consumer is at this purchase occasion, how likely are they to think of brand B before brand B′?* It is a relative accessibility measure, not a purchase probability.

---

## Getting started

### Prerequisites

- Python 3.11+
- Survey data file (Dynata zip export — not committed to the repo)

### Install

```bash
git clone https://github.com/hasdurrani/cep-simulator.git
cd cep-simulator
pip install -e .
```

### Place your data

Put the Dynata zip file in `backend/data/raw/` and update the paths in `backend/configs/cep_sim_config_uk.toml`:

```toml
[survey]
zip_path      = "backend/data/raw/your_file.zip"
data_file     = "your_data.csv"
codebook_file = "your_codebook.txt"
```

### Start the server

```bash
uvicorn frontend.cep_sim.api.app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Using the UI

### Step 1 — Select country

On first load, pick a market (UK or Brazil). This triggers `/api/setup`, which:

- Loads and reshapes the survey data
- Builds the CEP ontology (deduplication + family grouping)
- Initialises the memory model for every respondent × brand × CEP triple, with initial weights breadth-scaled by each respondent's CEP coverage
- Computes awareness-weighted brand priors from population-level survey mention rates
- Computes per-respondent responsiveness multipliers (how open each respondent is to new associations)
- Builds a brand-similarity matrix from pairwise CEP-profile cosine similarities
- Fits τ (softmax temperature), γ (competition weight), and prior_weight jointly via grid search to minimise calibration MAE
- Runs an 80/20 hold-out validation and stores held-out MAE and Spearman ρ
- Computes baseline recall across all purchase occasion scenarios

### Step 2 — Explore the Market tab

The **Market** tab shows the baseline state of the category before any ad is applied.

| Element | What it shows |
|---|---|
| **Respondents / Brands / Scenarios** | Summary counts for the loaded dataset |
| **MAE** | Mean absolute error between the model's predicted recall probabilities and the observed survey mention rates — a measure of how well the fitted model matches the data |
| **Hold-out MAE / ρ** | Same metrics computed on the 20% of respondents held back from parameter fitting — a check that fitted parameters generalise |
| **Brand × Scenario Recall heatmap** | Each cell is the population-average recall probability (%) for a brand in a given purchase occasion. Click a column header to sort by that scenario. Hover a cell for the exact value. |
| **CEP Opportunity Map** | Select a brand from the dropdown to see a horizontal bar chart of that brand's recall vs. the category average per scenario. Green bars = above average (strength to defend); red bars = below average (opportunity to grow). |
| **Brand Recall Leaderboard** | A ranked bar chart of mean recall across all scenarios — shows which brands have the strongest overall mental availability |

### Step 3 — Configure a simulation

The left panel has three collapsible sections:

#### Brand
Select the brand the ad is promoting.

#### Purchase occasions
Every CEP (Category Entry Point) in the dataset is listed, grouped by family. Assign each one a role:

- **P (Primary)** — the ad is strongly linked to this occasion; CEP fit weight φ = 1.0
- **S (Secondary)** — the ad weakly references this occasion; φ = 0.5
- Unselected CEPs are unaffected (φ = 0.0)

At least one Primary occasion is required to run.

#### Ad settings

| Parameter | Range | What it controls |
|---|---|---|
| **Branding clarity** (δ) | 0.1 – 1.0 | How clearly the ad links the brand to the selected occasion. An ad that prominently features the brand at a specific moment scores high; a vague lifestyle ad scores low. |
| **Attention weight** | 0.5 – 2.0 | Encoding efficiency — how much of the ad is actually processed by the viewer. Modulates the episodic memory boost (not the weight update in the current model). |
| **Channel** | digital video, social media, TV, OOH, radio, cinema | Context for the ad placement (metadata; does not currently modify the update rule). |
| **Emotion** | social warmth, confidence, pride, fun, depth, nostalgia | Emotional tone of the creative (metadata). |

### Step 4 — Run and read the results

Click **Run simulation**. The **Simulate** tab opens automatically.

### Step 5 — Compare two ads (optional)

Switch to the **Compare A vs B** tab to run two ad variants side by side against the same market baseline.

#### Metric badges

| Badge | What it means |
|---|---|
| **Ad lift** | Change in the focal brand's mean recall probability across all scored scenarios (in percentage points). Positive = the ad grew mental availability. |
| **Rank pre → post** | The focal brand's competitive rank before and after the ad, within the simulated scenario. A rank improvement (e.g. 3 → 1) means the brand displaced competitors for that occasion. |
| **MAE** | Model calibration error — same metric as the Market tab. Green < 0.05, amber otherwise. |
| **Median Spearman ρ** | Median rank correlation between the model's predicted brand order and the observed survey ranking across scenarios. A construct-validity check; higher is better. Green > 0.6. |

#### Flight Simulator tab

A horizontal bar chart showing each brand's pre- and post-ad recall probability for the simulated occasion, with the displacement between them highlighted. Below the chart, a sortable table shows:

| Column | Meaning |
|---|---|
| Brand | Brand name |
| Pre | Recall probability before the ad (%) |
| Post | Recall probability after the ad (%) |
| Δ Recall | Change in percentage points (positive = gained, negative = displaced) |
| Rank pre / post | Competitive rank before and after |
| Rank Δ | Movement in rank (↑ = improved) |

#### Memory Map tab

Two stacked heatmaps showing CEP-level recall across brands, split by respondent segment:

- **Brand loyalists** — respondents who already associate the focal brand with purchase occasions
- **Competitor loyalists** — respondents who primarily associate competitor brands

The colour scale shows recall strength per CEP. CEP labels on the axis are truncated to 32 characters; hover for the full text.

#### Calibration tab

Two panels:

- **Predicted vs. observed scatter** — each point is a brand × CEP cell. A well-calibrated model clusters around the diagonal. Points far from the diagonal indicate scenarios where the model over- or under-predicts.
- **Spearman ρ by scenario** — bar chart of rank correlation between predicted and observed brand ordering per purchase occasion. Scenarios below 0.2 are flagged as potentially misspecified.

### Step 6 — Export results (optional)

Click the **↓ Export CSV** button in the top-right of the tab bar at any point after setup. Downloads a ZIP named `cep_sim_{market}_export.zip` containing:

| File | Contents |
|---|---|
| `market_baseline.csv` | Population-average recall probability (%) per brand × scenario (pivot table) |
| `model_params.json` | Fitted τ, γ, prior_weight; MAE; hold-out MAE / ρ; respondent and brand counts |
| `flight_summary.csv` | Pre / post recall and rank for all brands in the simulated scenario (present if simulate has been run) |
| `ad_impact_summary.csv` | Per-scenario mean recall delta per brand across all occasions |
| `scenario_diagnostics.csv` | Per-scenario MAE and Spearman ρ |
| `scenario_recall.csv` | Full per-respondent recall scores across all scenarios |

---

### Compare A vs B tab

Configure two independent ads — each with its own brand, primary/secondary CEPs, and branding clarity — then click **Compare A vs B**.

Both ads are applied to the same unmodified market baseline (neither ad affects the other's starting point). The output is:

| Element | What it shows |
|---|---|
| **Comparison chart** | Grouped horizontal bar chart with two bars per brand: Ad A delta (cyan) and Ad B delta (orange), sorted by Ad A displacement |
| **Ad A table** | Full flight table for Ad A: pre/post recall, Δ recall, rank shift |
| **Ad B table** | Full flight table for Ad B |

This answers: *"Is it better to run an ad targeting occasion X or occasion Y? Which moves this brand's recall more? Which displaces more competitors?"*

---

## The model

### Memory initialisation

The survey records a binary recall indicator for each respondent–brand–CEP triple:

```
m(r, b, c) ∈ {0, 1}
```

Initial weights are breadth-scaled by each respondent's CEP coverage for that brand:

```
w₀(r, b, c) = [Σ_c m(r,b,c) / |C|] / k(r,b)   if m(r,b,c) = 1
```

where `k(r,b)` is the number of CEPs in which respondent `r` mentioned brand `b`. A brand mentioned in many CEPs gets a higher total weight than one mentioned in only one.

### Recall scoring

For respondent `r`, brand `b`, and a set of active CEPs `S`:

```
score(r, b, S) = Σ_{c∈S} w(r,b,c)  +  β(b)  −  γ · Σ_{b′≠b} sim(b,b′) · Σ_{c∈S} w(r,b′,c)
```

Scores are converted to probabilities via softmax with temperature `τ`:

```
P(r recalls b | S) = exp(score(r, b, S) / τ) / Σ_{b'} exp(score(r, b', S) / τ)
```

| Symbol | Meaning | How set |
|---|---|---|
| `w(r,b,c)` | Association strength of respondent r for brand b at CEP c | breadth-scaled from survey |
| `β(b)` | Brand-specific awareness prior | fitted from population mention rates |
| `sim(b,b′)` | CEP-profile cosine similarity between brands b and b′ | computed from survey |
| `γ` | Competition weight | grid-search fitted |
| `τ` | Softmax temperature (lower = sharper distribution) | grid-search fitted |

### Ad exposure update

When the population is exposed to an ad, each respondent's association weights are updated with saturation and per-respondent responsiveness:

```
w_new(r, b, c) = w_old(r, b, c) + λ · ρ(r) · e · δ · φ(c) · (1 − w_old / w_max)
```

| Symbol | Meaning | How set |
|---|---|---|
| `λ` | Base learning rate | config (default 0.1) |
| `ρ(r)` | Per-respondent responsiveness multiplier | computed from survey engagement |
| `e` | Exposure strength | ad-level |
| `δ` | Branding clarity | set in UI |
| `φ(c)` | CEP fit: 1.0 Primary, 0.5 Secondary, 0.0 otherwise | fixed |
| `w_max` | Saturation ceiling | config (default 5.0) |

New edges (brand–CEP pairs with no prior survey association) receive an additional friction factor (`new_edge_weight = 0.3`), reflecting the greater cognitive cost of forming a new association vs. reinforcing an existing one.

### Validation

The model is validated against the survey data in three ways:

- **Calibration (MAE):** Mean absolute error between the model's predicted population-average recall probabilities and the observed brand mention rates per CEP. Computed on training respondents.
- **Hold-out MAE / ρ:** Same metrics on the 20% of respondents held back from parameter fitting — confirms that fitted parameters (τ, γ, prior_weight) generalise.
- **Construct validity (Spearman ρ):** Rank correlation between the model's predicted brand ordering and the survey-observed ordering per scenario.

For the full mathematical specification, assumptions, known limitations, and upgrade paths see [backend/docs/model_spec.md](backend/docs/model_spec.md).

---

## Repo structure

```
cep-simulator/
├── backend/
│   ├── schemas/        # Pydantic models (config, respondent, events, ontology)
│   ├── service/        # Core engine
│   │   ├── recall_engine.py      # Scoring + softmax
│   │   ├── ad_engine.py          # Weight update rule
│   │   ├── validator.py          # Calibration, Spearman validity, sanity checks
│   │   ├── respondent_builder.py # Memory edge table
│   │   ├── ontology_builder.py   # CEP deduplication + family inference
│   │   ├── runner.py             # End-to-end pipeline runner
│   │   └── output_builder.py     # Artifact writing
│   ├── tests/          # 65 unit tests (no data file required)
│   ├── notebooks/      # Annotated analysis notebooks (UK, Brazil)
│   ├── docs/
│   │   └── model_spec.md         # Full mathematical specification
│   ├── data/           # Survey data (gitignored — not committed)
│   ├── framework/      # Artifact manifest utilities
│   └── configs/        # TOML configs per market
├── frontend/
│   └── cep_sim/
│       ├── api/
│       │   ├── app.py            # FastAPI app
│       │   ├── session.py        # In-memory session store
│       │   ├── plotly_charts.py  # Plotly JSON builders
│       │   └── routes/           # setup, simulate, baseline, compare, export endpoints
│       └── ui/
│           ├── index.html
│           └── src/
│               ├── deps.js       # All CDN imports (React 18, htm, Plotly)
│               ├── app.js
│               ├── api.js
│               └── components/
│                   ├── setup-panel.js
│                   ├── config-panel.js
│                   └── results-panel.js
└── pyproject.toml
```

---

## Running tests

Tests use synthetic fixtures (3 respondents, 3 brands, 2 CEPs) and do not require the survey data file. 65 tests across 5 modules.

```bash
pytest backend/tests/ -v
```

---

## Configuration reference

All model parameters are in `backend/configs/cep_sim_config_uk.toml`. The `[ad]` section sets the default brand and CEPs pre-selected in the UI on load.

```toml
[defaults]
base_usage_default         = 0.2   # β floor — minimum prior for brands with low survey coverage
learning_rate              = 0.1   # λ — base ad update step size (scaled per respondent at runtime)
competition_penalty_weight = 0.05  # γ — competition weight (overridden by grid-search fit at setup)
softmax_temperature        = 1.0   # τ — probability sharpness (overridden by grid-search fit at setup)
w_max                      = 5.0   # saturation ceiling for association weights
new_edge_weight            = 0.3   # friction for forming brand-new CEP associations
base_prior_weight          = 1.0   # scalar on awareness priors (overridden by grid-search fit at setup)

[ad]
focal_brand_name          = "Heineken"
focal_cep_keywords        = ["watching sport"]
secondary_cep_keywords    = ["hosting friends", "out with co-workers"]
```

**Note:** τ, γ, and `base_prior_weight` are overridden at setup time by a joint grid search that minimises calibration MAE. Config values serve as defaults only if fitting is skipped.

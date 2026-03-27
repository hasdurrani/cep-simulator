# CEP Simulator — Model Specification

**Version:** 0.3
**Status:** Active — Layer 1 and Layer 2 enhancements implemented; export and performance optimisations complete

---

## Theoretical grounding

The concepts underlying this simulator come from the **Ehrenberg-Bass Institute for Marketing Science** (University of South Australia), principally from Byron Sharp's *How Brands Grow* (2010) and subsequent EBI research on mental availability.

Two EBI constructs are central:

**Mental availability** — the probability that a consumer thinks of a brand in buying situations. EBI research establishes that mental availability is the primary driver of market share, and that it is built by creating and refreshing brand-relevant memory structures across a wide range of purchase occasions.

**Category Entry Points (CEPs)** — the situational, social, and emotional cues that trigger category and brand retrieval at the moment of purchase. EBI research shows that brands with strong and broad CEP associations are recalled more often and by more buyers. The goal of advertising, on this view, is to create and reinforce links between the brand and the CEPs that matter in the category.

**What this simulator adds**: EBI's framework is primarily empirical and descriptive. This simulator provides a mathematical operationalisation of that framework — a quantitative model of how CEP-linked memory structures are initialised from survey data, how they produce recall probabilities, and how advertising updates them. The mathematical choices (scoring function, softmax, weight update rule) are the author's formalisation, not EBI-published models.

---

## What this model predicts

This model predicts **relative brand accessibility under a CEP cue** — given that a consumer is at a specific purchase occasion, how likely are they to think of brand B before brand B′?

It does **not** predict:
- purchase incidence
- sales lift
- share of wallet

The output is a recall probability distribution over brands, conditional on an occasion. That framing applies throughout this document.

---

## Scope

This document formalises the four mathematical components of the CEP memory simulator:

1. How memory weights are initialised
2. How recall is scored and converted to probabilities
3. How ad exposure updates memory
4. How the model is validated against observed behaviour

Each section states the current implementation, its assumptions, its known limitations, and what a more rigorous version would require.

---

## 1. Memory Initialisation

### 1.1 What the survey gives us

The survey records a binary recall indicator for each (respondent, brand, CEP) triple:

```
m(r, b, c) ∈ {0, 1}
```

where `m = 1` means respondent `r` mentioned brand `b` when prompted with purchase occasion `c`.

Each respondent answers all 11 CEPs × 21 brands = 231 questions. We observe a sparse binary matrix.

### 1.2 Theoretical grounding

Treating survey recall mentions as a proxy for associative memory strength is consistent with **spreading activation theory** (Collins & Loftus, 1975), which models memory as a network of nodes (brands) connected by weighted associations (CEP links). Brand-CEP associations have been used as the operational basis of brand equity since Keller (1993), where brand knowledge is defined as the set of associations held in consumer memory. Top-of-mind recall in a CEP-cued survey is a standard EBI measurement of mental availability (Sharp, 2010).

### 1.3 Current initialisation rule

An association edge is created only when `m(r, b, c) = 1`:

```
w₀(r, b, c) = α · m(r, b, c)
```

where `α = assoc_strength_if_mentioned` (default: 1.0). Edges where `m = 0` are not stored; absent edges are treated as zero weight at scoring time.

**What this means in practice:** every mention is worth the same regardless of how many CEPs a brand was mentioned in, and non-mentions carry no information.

### 1.4 Assumptions

| Assumption | Status |
|---|---|
| Binary mention = binary association | Strong simplification. Survey recall measures top-of-mind accessibility, not association strength. |
| All mentions carry equal weight | Incorrect in principle. A brand mentioned in 9/11 CEPs has deeper penetration than one mentioned in 1/11. |
| Zero weight = no association | Reasonable for MVP — unmeasured edges are treated as no evidence rather than negative evidence. |
| α = 1.0 is the right scale | Arbitrary. Scale matters only relative to β (base prior) and λ (learning rate). Internally consistent but not anchored to any external probability. |

### 1.5 A stronger initialisation

A defensible alternative scales initial weights by the respondent's own CEP breadth for that brand. Let `k(r,b) = Σ_c m(r,b,c)` be the number of CEPs where respondent r mentioned brand b. Then:

```
         ⎧  [Σ_c m(r,b,c) / |C|] / k(r,b)   if m(r,b,c) = 1
w₀(r,b,c) = ⎨
         ⎩  0                                  otherwise
```

This preserves the respondent's total brand breadth score `Σ_c m(r,b,c) / |C|` — a value in [0, 1] interpretable as CEP coverage fraction — and allocates it equally across the CEPs they mentioned. A brand mentioned in 9/11 CEPs gets a higher total weight than one mentioned in 1/11, but each per-CEP edge is proportionally smaller.

This preserves the respondent's total brand breadth score as a CEP coverage fraction in [0,1] — consistent with EBI's concept of mental availability breadth — and allocates it proportionally across mentioned CEPs. The breadth-weighted version is **currently implemented** in `respondent_builder.py: build_respondent_brand_cep()`.

---

## 2. Recall Scoring

### 2.1 Theoretical grounding

The scoring structure has strong grounding in two independent traditions:

**ACT-R cognitive architecture (Anderson, 1983; Anderson & Lebiere, 1998):**
ACT-R models memory retrieval as an activation competition. The activation of a memory chunk i is:

```
A_i = B_i + Σ_j W_j · S_ji
```

where `B_i` is base-level activation (frequency/recency of prior retrieval) and `S_ji` is the associative strength from cue j. This maps directly onto the scoring model: `β` is the base-level activation, and `Σ w(r,b,c)` is the cue-weighted associative sum. ACT-R also predicts the fan effect — more items in memory linked to a cue means lower individual activation — which motivates the competition term.

**Luce's Choice Axiom / Multinomial Logit (Luce, 1959; McFadden, 1973):**
The softmax probability function is a direct implementation of Luce's Choice Axiom: `P(i | S) = u(i) / Σ_{j∈S} u(j)`. In log-linear form with utility `u(b) = exp(score(b)/τ)` this is exactly the Multinomial Logit model, for which McFadden received the Nobel Prize in Economics (2000). Temperature `τ` maps to the precision parameter in Random Utility Models.

### 2.2 Raw score

For respondent `r`, brand `b`, and a set of active CEPs `S` (the purchase occasion being simulated):

```
score(r, b, S) = semantic(r, b, S) + β + episodic(r, b, S) − γ · other_semantic(r, b, S)
```

Where:

```
semantic(r, b, S)       = Σ_{c ∈ S} w(r, b, c)
other_semantic(r, b, S) = Σ_{b′ ≠ b} semantic(r, b′, S)   [total respondent semantic − semantic(b)]
```

- `β` = `base_usage_default` (default: 0.2) — a uniform baseline prior for every brand
- `γ` = `competition_penalty_weight` (default: 0.05) — weight on competing semantic strength
- `episodic` = 0 unless episodic events are passed explicitly (see §3.2)

**Competition term (updated — cosine-similarity based):**
The original flat competition term `γ · (total_semantic − semantic(b))` was mathematically inert under softmax: it reduced to a constant shift that cancelled during normalisation, leaving recall probabilities unchanged regardless of γ.

The competition term is now brand-specific, using pairwise CEP-profile cosine similarity:

```
other_semantic(r, b, S) = Σ_{b′ ≠ b} sim(b, b′) · semantic(r, b′, S)
```

where `sim(b, b′)` is the cosine similarity between brand b and brand b′ computed from their population-level CEP mention vectors (`build_brand_similarity` in `recall_engine.py`). Brands with similar CEP profiles compete more strongly with each other than brands with orthogonal profiles. This term is brand-specific and does **not** cancel in softmax — γ now has a genuine effect on recall probabilities.

### 2.3 Recall probability

Scores are converted to probabilities via softmax with temperature `τ`:

```
P(r recalls b | S) = exp(score(r, b, S) / τ) / Σ_{b' ∈ B_S} exp(score(r, b', S) / τ)
```

Default `τ = 1.0`. Lower τ sharpens the distribution toward the leading brand; higher τ flattens it.

### 2.4 Assumptions and limitations

**Base prior (implemented — awareness-weighted):**
Brand priors are now computed from the survey rather than set flat. `compute_brand_priors(long_df)` derives `β(b)` as the population-level CEP mention rate for each brand, scaled by `base_prior_weight` (fitted). A brand mentioned in more CEPs by more respondents receives a larger prior, consistent with EBI evidence that mental availability is correlated with brand penetration. The `base_usage_default` config value acts as a minimum floor for brands with very low survey coverage.

**Additive combination:**
The additive structure (semantic + base + episodic) is an assumption shared with ACT-R, not a derivation. The components could interact — high semantic strength may reduce the marginal value of an episodic boost. A multiplicative or attention-weighted model would capture this at the cost of more parameters.

**Softmax and the IIA problem:**
Softmax satisfies Independence of Irrelevant Alternatives — adding a new brand dilutes all others equally. In practice, a niche craft beer entering the market does not equally dilute Heineken and another niche brand. Nested Logit or Mixed Logit models address this; they are out of scope for this version.

### 2.4 What the score is not

The recall probability `P(r recalls b | S)` is a relative accessibility rank, not an absolute probability of purchase. It should be interpreted as: *given that this respondent is at this purchase occasion, how likely are they to think of brand b before brand b'?* It does not model whether the respondent buys, nor whether the occasion actually occurs.

---

## 3. Ad Exposure Update

### 3.1 Theoretical grounding

The update rule is structurally equivalent to the **Rescorla-Wagner model** (Rescorla & Wagner, 1972), the most replicated model in associative learning:

```
ΔV_CS = α · β · (λ − V_CS)                          [Rescorla-Wagner]
Δw    = λ_lr · e · δ · φ(c) · (1 − w_old / w_max)   [this model]
```

The correspondence: `φ(c)` is stimulus salience (α); `δ` (branding clarity) is reinforcement salience (β); `w_max` is the asymptote (λ); and `(1 − w/w_max)` is the prediction error `(λ − V)`. The saturation factor produces exactly the diminishing-returns curve that Rescorla-Wagner predicts from first principles.

The new-edge friction parameter (`new_edge_weight = 0.3`) reduces the update for forming a completely new association vs. reinforcing an existing one. This is supported by Ebbinghaus (1885) on memory consolidation and a substantial advertising literature distinguishing between new-brand awareness building (requires more exposures) and reminder advertising for established associations.

### 3.2 Update rule

When respondent `r` is exposed to an ad for brand `b`:

```
w_new(r, b, c) = w_old(r, b, c) + Δ(b, c)
```

where:

```
Δ(b, c) = λ · e · δ · φ(c) · (1 − w_old / w_max)
```

- `λ` = `learning_rate` (default: 0.1)
- `e` = `exposure_strength` (default: 1.0; could encode channel reach or frequency)
- `δ` = `branding_clarity` (ad-level; 0–1 scale; how clearly the ad links brand to occasion)
- `φ(c)` = CEP fit weight: 1.0 for focal CEPs, 0.5 for secondary CEPs, 0.0 otherwise
- `w_max` = saturation ceiling (default: 5.0)

If no edge exists for `(r, b, c)`, a new edge is created with an additional friction factor `new_edge_weight` (default: 0.3) applied to `Δ`, reflecting the greater cognitive cost of forming a new association vs. reinforcing an existing one.

### 3.3 Episodic events

In addition to updating `w`, each ad application creates an `EpisodicEvent` record archived to `episodic_events.csv`. These are an audit trail only.

**Double-count resolution (implemented):** The `episodic_events` parameter has been removed from `run_scenario_recall`. The updated weight table `rbc_post` returned by `apply_ad_to_population` is the authoritative source of ad effect — all downstream scoring reads from it. Episodic events are not passed to the scorer, eliminating the latent double-count risk.

### 3.4 Assumptions and limitations

**Saturation (implemented):**
The update rule includes a saturation factor to prevent unbounded weight growth:

```
w_new = w_old + Δ · (1 − w_old / w_max)
```

This keeps weights in [0, w_max] and models diminishing returns on repeated exposures. `w_max` is set in config (default: 5.0). As `w_old` approaches `w_max`, additional ad exposures have progressively less effect — consistent with the empirical wear-in/wear-out pattern observed in advertising research.

**No decay:**
The model has no time dimension. An ad seen today has the same effect as one seen six months ago. For a longitudinal simulator, a decay factor `ρ ∈ (0, 1)` should be applied at each time step:

```
w(t+1) = ρ · w(t) + Δ_new(t)
```

**New edges vs. reinforcement:**
Creating a new edge (where `w_old = 0`) and reinforcing an existing strong edge are treated identically. In cognitive models, new association formation is qualitatively different from strength reinforcement and typically requires more exposures.

**Population homogeneity (partially addressed — responsiveness implemented):**
A per-respondent learning rate multiplier is now computed from survey data via `compute_respondent_responsiveness(rbc_df)` and wired into `apply_ad_to_population`. Respondents who mentioned more brands across more CEPs (higher baseline engagement) receive higher multipliers, reflecting greater openness to brand associations. The multiplier is applied to λ before the update: `λ_r = λ · responsiveness(r)`. Attention, prior brand attitude, and media context remain unmodelled.

---

## 4. Validation

### 4.1 What we can validate against

The survey gives us ground truth at the individual level: for each respondent × CEP, we know which brands they mentioned. This is a held-out test set if we split respondents before fitting.

There are two levels of validation possible:

**Population level:** Do the model's predicted recall probabilities, averaged across respondents, match the observed mention rates?

**Individual level:** For a given respondent and CEP, does the model's brand ranking correlate with that respondent's pattern of mentions across other CEPs (cross-CEP generalisation)?

### 4.2 Current sanity checks (implemented)

| Check | What it tests |
|---|---|
| `cue_relevance` | At least one brand has recall_prob > 0.001 for every scenario |
| `respondent_heterogeneity` | Brand recall prob std > 0 across respondents for at least one scenario |
| `ad_lift` | Mean delta > 0 for at least one brand |
| `competition_realism` | More than one brand appears in rankings |

These are existence checks. They confirm the pipeline produces non-degenerate output. They do not assess whether the output is accurate.

### 4.3 Calibration check (implemented — `validator.py: run_calibration_check`)

For each CEP `c`, the model predicts `P(r recalls b | c)` for all respondents `r` and brands `b`. The observed mention rate for brand `b` at CEP `c` is:

```
p̂(b, c) = (1/|R|) Σ_r m(r, b, c)
```

A calibrated model should satisfy:

```
(1/|R|) Σ_r P(r recalls b | c) ≈ p̂(b, c)
```

`run_calibration_check(scenario_recall_df, long_df)` computes mean predicted `recall_prob` per `(scenario_name, brand_name)` and compares to the raw mention rates from `long_df`. Returns a DataFrame with per-cell absolute error and stores overall MAE in `cal_df.attrs["mae"]`.

The calibration scatter (predicted vs. observed) and per-scenario Spearman ρ bars are displayed in the UI Calibration tab.

### 4.4 Construct-validity check (implemented — `validator.py: run_spearman_validity`)

For each scenario, brands are ranked by their predicted population-average recall probability and compared to their observed survey mention rate ranking.

Compute Spearman rank correlation between the two rankings:

```
ρ_s(c) = Spearman( rank_by_predicted_prob(c), rank_by_observed_rate(c) )
```

`run_spearman_validity(scenario_recall_df, long_df)` returns a DataFrame of per-scenario ρ values sorted descending. Scenarios with ρ < 0.2 are flagged as candidates for ontology review. The median ρ across scenarios is reported as the **Median Spearman ρ** badge in the UI.

**Important caveat:** this is a construct-validity check, not a ground-truth accuracy test. The observed mention rate is itself a survey-derived measure, not observed purchase behavior. A high correlation confirms that the model's predicted brand ordering is internally consistent with the survey data — it does not confirm that the model correctly predicts what a consumer would think of at the moment of purchase.

Label it accordingly when presenting results.

### 4.5 Hold-out test (implemented — `validator.py: make_holdout_split, run_holdout_validation`)

Respondents are split 80/20 by `respondent_id` hash (deterministic, reproducible). Parameters (τ, γ, prior_weight) are optimised on the train split only. Held-out respondents are scored using the train-derived parameters and their own individual memory edges.

`run_holdout_validation` returns `holdout_mae` and `holdout_rho` — both are stored in the session and surfaced in the UI alongside the in-sample metrics. A large gap between in-sample and hold-out MAE would indicate overfit to the training population, which is a useful signal for ontology or parameter review.

Note: since each respondent's memory edges `w(r,b,c)` come directly from their own survey responses and are not estimated from the population, there is no classical overfitting risk on the edges themselves — the hold-out test primarily validates the shared parameters (τ, γ, prior_weight) and brand priors.

### 4.6 What good looks like

| Metric | Minimum bar | Strong result |
|---|---|---|
| Calibration (mean abs. error, predicted vs. observed mention rate) | < 5 pp per brand-CEP cell | < 2 pp |
| Spearman ρ (predicted rank vs. cross-CEP breadth) | > 0.15 median | > 0.35 median |
| Ad lift for focal brand (trendy bar, Heineken ad) | > 0 | Statistically distinguishable from noise across respondents |

---

## 5. Parameter Summary

| Parameter | Symbol | Default | How set | What it controls |
|---|---|---|---|---|
| `assoc_strength_if_mentioned` | α | 1.0 | config | Initial edge weight |
| `base_usage_default` | β(b) | survey-derived | fitted from data | Per-brand awareness prior (population CEP mention rate × prior_weight) |
| `base_prior_weight` | — | fitted | grid search | Scalar multiplier on brand awareness priors |
| `learning_rate` | λ | 0.1 | config | Base ad update step size |
| `competition_penalty_weight` | γ | fitted | grid search | Weight on cosine-similarity competition term |
| `softmax_temperature` | τ | fitted | grid search | Probability sharpness (lower = sharper) |
| `w_max` | — | 5.0 | config | Saturation ceiling for ad updates |
| `new_edge_weight` | — | 0.3 | config | Friction for forming new brand–CEP associations |
| `responsiveness(r)` | — | survey-derived | per respondent | Per-respondent learning rate multiplier |
| `branding_clarity` | δ | 0.8 (ad-level) | UI / Ad object | Ad-brand linkage quality |
| `exposure_strength` | e | 1.0 (ad-level) | Ad object | Reach / frequency multiplier |
| `attention_weight` | — | 1.0 (ad-level) | Ad object | Encoding efficiency (archived in episodic events) |
| CEP fit (focal) | φ | 1.0 | hardcoded | Weight on focal CEP update |
| CEP fit (secondary) | φ | 0.5 | hardcoded | Weight on secondary CEP update |

**Parameter fitting (implemented):** τ, γ, and `base_prior_weight` are jointly optimised via grid search at setup time (`fit_parameters` in `calibration.py`). The grid covers τ ∈ {0.3, 0.7, 1.0, 2.0}, γ ∈ {0.0, 0.05, 0.1, 0.2}, prior_weight ∈ {0.5, 1.0, 2.0}. The combination that minimises in-sample calibration MAE is selected and stored in the session. Config values serve as defaults only if fitting is skipped.

**Performance (fast numpy grid search):** The grid evaluation avoids repeated pandas operations. `_precompute_scenario_frames()` builds `semantic`, `competition`, `prior`, and `observed` numpy arrays once per scenario before the grid loop. `_fast_grid_mae()` then applies numerically-stable softmax and computes MAE in pure numpy for each (τ, γ, prior_weight) combination — no DataFrames in the inner loop. This produces a 5–10× speedup over the original pandas path, particularly for larger markets.

**γ correctness fix:** The original implementation passed a flat competition term to the grid search, which is constant under softmax and cancels during normalisation — meaning γ had zero effect on the fitted probabilities. The fix: `brand_similarity` is now passed into `_precompute_scenario_frames()` so that the cosine-weighted competition term is used during fitting, giving γ a genuine effect on MAE and ensuring the optimised γ reflects true competitive structure.

---

## 6. Roadmap

The specification above identifies concrete gaps. In priority order:

1. ~~**Implement calibration check (§4.3)**~~ — done (`run_calibration_check` in `validator.py`).
2. ~~**Implement construct-validity check (§4.4)**~~ — done (`run_spearman_validity` in `validator.py`).
3. ~~**Fix double-count risk (§3.3)**~~ — done (`episodic_events` removed from `run_scenario_recall`; `rbc_post` is authoritative).
4. ~~**Scale initial weights by CEP breadth (§1.5)**~~ — done (breadth-weighting implemented in `respondent_builder.py`).
5. ~~**Fit τ, γ, and prior_weight to observed mention rates (§5)**~~ — done (joint grid search in `fit_parameters`; runs at setup).
6. ~~**Add saturation to update rule (§3.4)**~~ — done (implemented in `ad_engine.py`).
7. ~~**Hold-out validation (§4.5)**~~ — done (`make_holdout_split` + `run_holdout_validation`; metrics surfaced in UI).
8. ~~**Awareness-weighted brand priors**~~ — done (`compute_brand_priors` replaces flat β).
9. ~~**Per-respondent responsiveness**~~ — done (`compute_respondent_responsiveness` wired into ad update).
10. ~~**Cosine-similarity competition term**~~ — done (`build_brand_similarity` + similarity-weighted competition replaces inert flat term).
11. ~~**Run A vs B comparison**~~ — done (`POST /api/compare`; Compare A vs B tab in UI with grouped displacement chart and side-by-side flight tables).
12. ~~**Export**~~ — done (`GET /api/export/{session_id}`; streams a ZIP containing `market_baseline.csv`, `model_params.json`, and simulation CSVs; download button in UI tab bar).
13. **Multiple ad flights with decay** — sequential exposures with time-decay between flights.

---

## 7. References

Anderson, J. R. (1983). *The Architecture of Cognition*. Harvard University Press.

Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought*. Lawrence Erlbaum Associates.

Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82(6), 407–428.

Ebbinghaus, H. (1885). *Über das Gedächtnis*. Duncker & Humblot. [Translation: *Memory: A Contribution to Experimental Psychology*, 1913.]

Keller, K. L. (1993). Conceptualizing, measuring, and managing customer-based brand equity. *Journal of Marketing*, 57(1), 1–22.

Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*. Wiley.

McFadden, D. (1973). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105–142). Academic Press.

Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical Conditioning II* (pp. 64–99). Appleton-Century-Crofts.

Sharp, B. (2010). *How Brands Grow: What Marketers Don't Know*. Oxford University Press.

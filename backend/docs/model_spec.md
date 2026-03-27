# CEP Simulator — Model Specification

**Version:** 0.1 (MVP)
**Status:** Methodological checkpoint — specification precedes further development

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

### 1.2 Current initialisation rule

An association edge is created only when `m(r, b, c) = 1`:

```
w₀(r, b, c) = α · m(r, b, c)
```

where `α = assoc_strength_if_mentioned` (default: 1.0). Edges where `m = 0` are not stored; absent edges are treated as zero weight at scoring time.

**What this means in practice:** every mention is worth the same regardless of how many CEPs a brand was mentioned in, and non-mentions carry no information.

### 1.3 Assumptions

| Assumption | Status |
|---|---|
| Binary mention = binary association | Strong simplification. Survey recall measures top-of-mind accessibility, not association strength. |
| All mentions carry equal weight | Incorrect in principle. A brand mentioned in 9/11 CEPs has deeper penetration than one mentioned in 1/11. |
| Zero weight = no association | Reasonable for MVP — unmeasured edges are treated as no evidence rather than negative evidence. |
| α = 1.0 is the right scale | Arbitrary. Scale matters only relative to β (base prior) and λ (learning rate). Internally consistent but not anchored to any external probability. |

### 1.4 A stronger initialisation

A defensible alternative scales initial weights by the respondent's own CEP breadth for that brand. Let `k(r,b) = Σ_c m(r,b,c)` be the number of CEPs where respondent r mentioned brand b. Then:

```
         ⎧  [Σ_c m(r,b,c) / |C|] / k(r,b)   if m(r,b,c) = 1
w₀(r,b,c) = ⎨
         ⎩  0                                  otherwise
```

This preserves the respondent's total brand breadth score `Σ_c m(r,b,c) / |C|` — a value in [0, 1] interpretable as CEP coverage fraction — and allocates it equally across the CEPs they mentioned. A brand mentioned in 9/11 CEPs gets a higher total weight than one mentioned in 1/11, but each per-CEP edge is proportionally smaller.

In the current MVP, k(r,b) = 1 for most edges (binary, not breadth-weighted), so the numerator and denominator cancel and w₀ = α for all edges. The formula above is the upgrade path, not the current state.

Implementing this requires updating `respondent_builder.py: build_respondent_brand_cep()` to compute k(r,b) per respondent-brand pair before assigning per-CEP weights.

---

## 2. Recall Scoring

### 2.1 Raw score

For respondent `r`, brand `b`, and a set of active CEPs `S` (the purchase occasion being simulated):

```
score(r, b, S) = semantic(r, b, S) + β − γ · (|B_S| − 1)
```

Where:

```
semantic(r, b, S) = Σ_{c ∈ S} w(r, b, c)
```

- `β` = `base_usage_default` (default: 0.2) — a uniform baseline prior applied to every brand
- `γ` = `competition_penalty_weight` (default: 0.05) — a per-competitor deduction
- `|B_S|` = number of distinct brands with any association to any CEP in `S` across the full population

An episodic term is added when ad exposure events are present (see §3):

```
score(r, b, S) = semantic(r, b, S) + episodic(r, b, S) + β − γ · (|B_S| − 1)
```

### 2.2 Recall probability

Scores are converted to probabilities via softmax with temperature `τ`:

```
P(r recalls b | S) = exp(score(r, b, S) / τ) / Σ_{b' ∈ B_S} exp(score(r, b', S) / τ)
```

Default `τ = 1.0`. Lower τ sharpens the distribution toward the leading brand; higher τ flattens it.

### 2.3 Assumptions and limitations

**Competition term:**
The penalty `γ · (|B_S| − 1)` is a flat constant tax — it does not depend on which brand is being scored or how strong competitors are. Every brand pays the same deduction regardless of whether competitors are dominant or marginal. This is the weakest part of the scoring model.

The next version should replace it with a brand-specific competition term based on CEP profile overlap:

```
competition(r, b, S) = Σ_{b′ ≠ b} sim(b, b′ | S) · availability(r, b′)
```

where `sim(b, b′ | S)` measures how much b and b′ overlap in their CEP associations (cosine similarity of their population-level CEP vectors restricted to S), and `availability(r, b′)` is b′'s semantic score for respondent r. This makes the competitive pressure on Heineken different from the pressure on a niche brand even under the same scenario.

The current term also uses population-level brand density (`|B_S|` across all respondents), not respondent-level density. This means the competitive landscape is identical for a heavy Heineken user and a non-buyer — only their semantic weight differs.

**Base prior:**
`β = 0.2` is a uniform flat prior. In reality, brand awareness (Q8 in the survey) is heterogeneous and correlated with CEP recall. A stronger base would be `β(b) ∝ market_share(b)` or `β(r, b) ∝ awareness(r, b)`.

**Additive combination:**
The score is a linear sum of components. This is an assumption, not a derivation. The components could interact — for example, high semantic strength may reduce the marginal value of an episodic boost. A multiplicative or attention-weighted model would capture this, at the cost of more parameters.

**Softmax and the IIA problem:**
Softmax satisfies Independence of Irrelevant Alternatives — adding a new brand equally dilutes all others. In beer, new alternatives do not uniformly dilute market leaders. This is a known limitation of softmax-based choice models; Dirichlet-logistic or nested logit are stronger but out of scope for this MVP.

### 2.4 What the score is not

The recall probability `P(r recalls b | S)` is a relative accessibility rank, not an absolute probability of purchase. It should be interpreted as: *given that this respondent is at this purchase occasion, how likely are they to think of brand b before brand b'?* It does not model whether the respondent buys, nor whether the occasion actually occurs.

---

## 3. Ad Exposure Update

### 3.1 Update rule

When respondent `r` is exposed to an ad for brand `b`:

```
w_new(r, b, c) = w_old(r, b, c) + Δ(b, c)
```

where:

```
Δ(b, c) = λ · e · δ · φ(c)
```

- `λ` = `learning_rate` (default: 0.1)
- `e` = `exposure_strength` (default: 1.0; could encode channel reach or frequency)
- `δ` = `branding_clarity` (ad-level; 0–1 scale; how clearly the ad links brand to occasion)
- `φ(c)` = CEP fit weight: 1.0 for focal CEPs, 0.5 for secondary CEPs, 0.0 otherwise

If no edge exists for `(r, b, c)`, a new edge is created with weight `Δ(b, c)`.

### 3.2 Episodic events

In addition to updating `w`, each ad application creates an `EpisodicEvent` record. When `episodic_events` is passed to the scoring function, the episodic boost is:

```
episodic(r, b, S) = Σ_{events e: e.respondent=r, e.brand=b, e.cep∈S} e.strength
```

where `e.strength = exposure_strength × branding_clarity × attention_weight`.

This means the model currently double-counts ad impact: once through the updated `w` values and again through the episodic events. In the notebook, `apply_ad_to_population` returns `rbc_post` (updated weights), and `run_ad_impact` uses `rbc_post` without passing episodic events — so the double-count is avoided in practice, but it is a latent design risk.

### 3.3 Assumptions and limitations

**No upper bound:**
Weights grow without limit under repeated exposures. There is no saturation. In practice, mental association strength has diminishing returns and an effective ceiling. A saturating update would be:

```
w_new = w_old + Δ · (1 − w_old / w_max)
```

This keeps weights in [0, w_max] and naturally models wear-in/wear-out.

**No decay:**
The model has no time dimension. An ad seen today has the same effect as one seen six months ago. For a longitudinal simulator, a decay factor `ρ ∈ (0, 1)` should be applied at each time step:

```
w(t+1) = ρ · w(t) + Δ_new(t)
```

**New edges vs. reinforcement:**
Creating a new edge (where `w_old = 0`) and reinforcing an existing strong edge are treated identically. In cognitive models, new association formation is qualitatively different from strength reinforcement and typically requires more exposures.

**Population homogeneity:**
Every respondent receives the same Δ. In reality, attention, prior brand attitude, and media context modulate encoding. The `attention_weight` field exists on the `Ad` object but is currently only factored into the episodic strength, not the weight update.

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

### 4.5 Hold-out test (not yet implemented — future work)

Split respondents 80/20 by `respondent_id` hash (deterministic, reproducible).

What is derived from the train set only:
- The **shared CEP ontology** (deduplication, family labels)
- The **competition denominator** `|B_S|` (population-level brand density)
- Any future **parameter tuning** of β and γ against calibration MAE

What is not fitted and therefore not at risk of overfitting:
- Each respondent's own memory edges `w(r,b,c)` — these come directly from their individual survey responses

Held-out respondents are scored using the train-derived ontology and parameters, but their own observed edges. Calibration (§4.3) and construct-validity (§4.4) are then measured on the held-out set.

If population-level calibration holds on held-out respondents but degrades on CEP-level cells, it indicates that the ontology or competition structure is sensitive to population composition — a useful signal for future data collection design.

### 4.6 What good looks like

| Metric | Minimum bar | Strong result |
|---|---|---|
| Calibration (mean abs. error, predicted vs. observed mention rate) | < 5 pp per brand-CEP cell | < 2 pp |
| Spearman ρ (predicted rank vs. cross-CEP breadth) | > 0.15 median | > 0.35 median |
| Ad lift for focal brand (trendy bar, Heineken ad) | > 0 | Statistically distinguishable from noise across respondents |

---

## 5. Parameter Summary

| Parameter | Symbol | Default | Where set | What it controls |
|---|---|---|---|---|
| `assoc_strength_if_mentioned` | α | 1.0 | config | Initial edge weight |
| `base_usage_default` | β | 0.2 | config | Uniform brand prior |
| `learning_rate` | λ | 0.1 | config | Ad update step size |
| `competition_penalty_weight` | γ | 0.05 | config | Per-competitor score deduction |
| `softmax_temperature` | τ | 1.0 | config | Probability sharpness |
| `branding_clarity` | δ | 0.8 (ad-level) | Ad object | Ad-brand linkage quality |
| `exposure_strength` | e | 1.0 (ad-level) | Ad object | Reach / frequency multiplier |
| `attention_weight` | — | 1.0 (ad-level) | Ad object | Encoding efficiency |
| CEP fit (focal) | φ | 1.0 | hardcoded | Weight on focal CEP update |
| CEP fit (secondary) | φ | 0.5 | hardcoded | Weight on secondary CEP update |

None of these parameters are estimated from data. They are set by assumption. Before claiming predictive validity, at least β and γ should be fit to minimise the calibration error in §4.3.

---

## 6. Roadmap

The specification above identifies concrete gaps. In priority order:

1. ~~**Implement calibration check (§4.3)**~~ — done (`run_calibration_check` in `validator.py`).
2. ~~**Implement construct-validity check (§4.4)**~~ — done (`run_spearman_validity` in `validator.py`).
3. **Fix double-count risk (§3.2)** — decide whether episodic events are additive to weight updates or a replacement. Document the choice.
4. **Scale initial weights by CEP breadth (§1.4)** — stronger initialisation, one change in `respondent_builder.py`.
5. **Fit β and γ to observed mention rates (§5)** — simple grid search on calibration MAE. Makes the model defensible.
6. **Add saturation to update rule (§3.3)** — one-line change, prevents unbounded weight growth.
7. **Hold-out validation (§4.5)** — 80/20 respondent split for held-out calibration and construct-validity.

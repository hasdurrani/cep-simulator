"""
Recall engine — core of the CEP simulator.

Scoring model (additive):
    score = semantic + base + episodic - competition

semantic    : sum of respondent brand–CEP association strengths across active CEPs
base        : small fixed prior representing baseline brand awareness
episodic    : boost from recent ad exposures or purchase events
competition : penalty proportional to the total semantic strength of competing brands
"""
import logging
import pandas as pd

from backend.schemas.config import CepSimConfig
from backend.service.utils import softmax

logger = logging.getLogger(__name__)


def get_recall_scores(
    respondent_id: str,
    active_ceps: list[str],
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    config: CepSimConfig,
    episodic_events: pd.DataFrame | None = None,
    brand_priors: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Return a raw recall score per brand for a given respondent and set of active CEPs.

    active_ceps: list of cep_labels or cep_ids to activate.
    """
    defaults = config.defaults

    # Resolve cep labels / ids to cep_ids
    active_cep_ids = _resolve_cep_ids(active_ceps, cep_master_df)
    if not active_cep_ids:
        logger.warning("No matching CEP IDs found for: %s", active_ceps)
        return {}

    # Guard: warn if both updated rbc (with ad_exposure edges) and episodic_events are passed —
    # this would double-count the ad effect.
    if episodic_events is not None and len(episodic_events) > 0:
        ad_edges = rbc_df[
            (rbc_df["respondent_id"] == respondent_id)
            & (rbc_df.get("source", pd.Series(dtype=str)) == "ad_exposure")
            & (rbc_df["cep_id"].isin(active_cep_ids))
        ]
        if len(ad_edges) > 0:
            ep_overlap = episodic_events[
                (episodic_events["respondent_id"] == respondent_id)
                & (episodic_events["cep_id"].isin(active_cep_ids))
            ]
            if len(ep_overlap) > 0:
                logger.warning(
                    "Double-count risk for respondent %s: rbc_df contains ad_exposure edges "
                    "AND episodic_events cover the same CEPs. Pass either updated rbc OR "
                    "episodic_events — not both.",
                    respondent_id,
                )

    # Filter respondent's memory edges to active CEPs
    mask = (rbc_df["respondent_id"] == respondent_id) & (rbc_df["cep_id"].isin(active_cep_ids))
    relevant = rbc_df[mask]

    # All brands that appear for ANY respondent under active CEPs (to compute competition)
    all_brands_active = rbc_df[rbc_df["cep_id"].isin(active_cep_ids)]["brand_id"].unique()

    # Pre-compute total semantic strength for this respondent (used in competition term)
    total_respondent_semantic = relevant["assoc_strength"].sum()

    scores: dict[str, float] = {}

    for brand_id in all_brands_active:
        # --- semantic ---
        brand_edges = relevant[relevant["brand_id"] == brand_id]
        semantic = brand_edges["assoc_strength"].sum()

        # --- base ---
        if brand_priors is not None:
            base = brand_priors.get(brand_id, defaults.base_usage_default) * defaults.base_prior_weight
        else:
            base = defaults.base_usage_default

        # --- episodic ---
        episodic = 0.0
        if episodic_events is not None and len(episodic_events) > 0:
            ep_mask = (
                (episodic_events["respondent_id"] == respondent_id)
                & (episodic_events["brand_id"] == brand_id)
                & (episodic_events["cep_id"].isin(active_cep_ids))
            )
            episodic = episodic_events[ep_mask]["strength"].sum()

        # --- competition ---
        # Sum of semantic strengths of all *other* brands for this respondent at active CEPs.
        # Note: this term is (total - semantic(b)), so it's brand-specific but still linear in
        # semantic(b). Under softmax this cancels to softmax(semantic); the value of this
        # formulation is that raw scores become more interpretable (brands in crowded CEP
        # contexts have lower raw scores). Probability differentiation comes from temperature.
        other_semantic_sum = total_respondent_semantic - semantic
        competition = defaults.competition_penalty_weight * max(0.0, other_semantic_sum)

        scores[brand_id] = semantic + base + episodic - competition

    return scores


def get_recall_probs(
    respondent_id: str,
    active_ceps: list[str],
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    config: CepSimConfig,
    episodic_events: pd.DataFrame | None = None,
    brand_priors: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return softmax-normalised recall probabilities per brand."""
    scores = get_recall_scores(
        respondent_id, active_ceps, rbc_df, cep_master_df, config, episodic_events,
        brand_priors=brand_priors,
    )
    return softmax(scores, temperature=config.defaults.softmax_temperature)


def rank_brands(probs: dict[str, float]) -> list[tuple[str, float]]:
    """Return brands sorted by recall probability descending."""
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def _resolve_cep_ids(active_ceps: list[str], cep_master_df: pd.DataFrame) -> list[str]:
    """
    Accept a mix of cep_ids (cep_001), cep_labels, or normalized text fragments
    and return the matching cep_ids.
    """
    resolved = []
    for cep in active_ceps:
        cep_lower = cep.strip().lower()
        # Exact cep_id match
        if cep_lower in cep_master_df["cep_id"].values:
            resolved.append(cep_lower)
            continue
        # Substring match against cep_label or cep_description
        matches = cep_master_df[
            cep_master_df["cep_label"].str.lower().str.contains(cep_lower, na=False)
            | cep_master_df["cep_description"].str.lower().str.contains(cep_lower, na=False)
        ]
        resolved.extend(matches["cep_id"].tolist())

    return list(set(resolved))


# ---------------------------------------------------------------------------
# Scenario libraries — one per market, keyed to actual survey CEP stems
# ---------------------------------------------------------------------------

BRAZIL_SCENARIOS: list[dict] = [
    {"scenario_name": "trendy_bar",        "active_ceps": ["bar da moda"],                        "context": {"setting": "trendy_bar"}},
    {"scenario_name": "outdoor_hot_day",   "active_ceps": ["ao ar livre em um dia quente"],        "context": {"setting": "outdoor"}},
    {"scenario_name": "bbq",               "active_ceps": ["churrasco com amigos"],                "context": {"setting": "bbq"}},
    {"scenario_name": "neighbourhood_bar", "active_ceps": ["bar de bairro"],                       "context": {"setting": "local_bar"}},
    {"scenario_name": "sharing_large_bottle","active_ceps":["garrafa grande com amigos"],          "context": {"format": "garrafão"}},
    {"scenario_name": "outdoor_festival",  "active_ceps": ["festivais ao ar livre"],               "context": {"setting": "festival"}},
    {"scenario_name": "watching_football", "active_ceps": ["futebol ou outros esportes"],          "context": {"activity": "sports"}},
    {"scenario_name": "special_occasion",  "active_ceps": ["celebrando uma ocasião especial"],     "context": {"occasion": "celebration"}},
    {"scenario_name": "social_dinner",     "active_ceps": ["jantar social"],                       "context": {"setting": "dinner"}},
    {"scenario_name": "after_work",        "active_ceps": ["encontrando amigos depois do trabalho"],"context": {"setting": "after_work"}},
    {"scenario_name": "on_holiday",        "active_ceps": ["férias"],                              "context": {"setting": "holiday"}},
]

UK_SCENARIOS: list[dict] = [
    {"scenario_name": "bbq",                     "active_ceps": ["BBQ"],                                       "context": {"setting": "bbq"}},
    {"scenario_name": "hosting_at_home",          "active_ceps": ["hosting friends at home"],                   "context": {"setting": "home"}},
    {"scenario_name": "watching_sport",           "active_ceps": ["watching sport"],                            "context": {"activity": "sport"}},
    {"scenario_name": "outdoor_hot_day",          "active_ceps": ["outdoors on a hot or sunny day"],            "context": {"setting": "outdoor"}},
    {"scenario_name": "sociable_dinner",          "active_ceps": ["sociable dinner"],                           "context": {"setting": "dinner"}},
    {"scenario_name": "house_party_club",         "active_ceps": ["lively house party or club"],                "context": {"setting": "party"}},
    {"scenario_name": "trendy_bar",               "active_ceps": ["trendy bar and want to make"],               "context": {"setting": "trendy_bar"}},
    {"scenario_name": "out_with_coworkers",       "active_ceps": ["out with co-workers"],                       "context": {"setting": "after_work"}},
    {"scenario_name": "going_out_for_meal",       "active_ceps": ["going out for a meal"],                      "context": {"setting": "meal"}},
    {"scenario_name": "pre_drinks",               "active_ceps": ["heading out for the night"],                 "context": {"setting": "pre_drinks"}},
    {"scenario_name": "quick_drink_home",         "active_ceps": ["quick drink on your way home"],              "context": {"setting": "quick_drink"}},
    {"scenario_name": "pub_with_friends",         "active_ceps": ["meeting your friends at the pub"],           "context": {"setting": "pub"}},
    {"scenario_name": "live_music_gig",           "active_ceps": ["live music event or gig"],                   "context": {"setting": "gig"}},
    {"scenario_name": "special_occasion",         "active_ceps": ["celebrating a special occasion"],            "context": {"occasion": "celebration"}},
    {"scenario_name": "out_to_have_fun",          "active_ceps": ["out to have fun"],                          "context": {"mood": "fun"}},
    {"scenario_name": "easy_to_drink",            "active_ceps": ["something easy to drink"],                   "context": {"preference": "easy"}},
    {"scenario_name": "pacing_yourself",          "active_ceps": ["pace yourself"],                             "context": {"preference": "pacing"}},
    {"scenario_name": "after_long_day",           "active_ceps": ["after a long day"],                          "context": {"mood": "relaxation"}},
    {"scenario_name": "buying_for_group",         "active_ceps": ["buying drinks for a big group"],             "context": {"format": "group"}},
    {"scenario_name": "holiday_discovery",        "active_ceps": ["discovered and enjoyed while on holiday"],   "context": {"occasion": "holiday"}},
    {"scenario_name": "trying_something_different","active_ceps": ["trying something a little different"],      "context": {"preference": "discovery"}},
]

# Default (backwards-compatible alias)
SCENARIOS = BRAZIL_SCENARIOS


def get_scenarios(country: str) -> list[dict]:
    """Return the scenario list for the given country ('Brazil' or 'UK')."""
    if country.upper() == "UK":
        return UK_SCENARIOS
    return BRAZIL_SCENARIOS

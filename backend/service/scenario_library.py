"""
Demo scenario library — Brazil market.

Scenarios are keyed to the actual CEP stems from the Dynata survey
(Q10–Q20, excluding Q12=BBQ as per demo spec).

Each entry:
    scenario_name  : unique slug used throughout the demo notebook
    active_ceps    : list of substrings that _resolve_cep_ids() will match
                     against cep_label or cep_description in cep_master_df
    context        : free-form metadata for display / annotation purposes
"""

DEMO_BRAZIL_SCENARIOS: list[dict] = [
    {
        "scenario_name": "trendy_bar_good_impression",
        "active_ceps":   ["bar da moda"],
        "context":       {"setting": "trendy_bar", "motivation": "impress"},
    },
    {
        "scenario_name": "hot_sunny_outdoors",
        "active_ceps":   ["ao ar livre em um dia quente"],
        "context":       {"setting": "outdoor", "weather": "hot"},
    },
    {
        "scenario_name": "neighborhood_bar",
        "active_ceps":   ["bar de bairro"],
        "context":       {"setting": "local_bar"},
    },
    {
        "scenario_name": "sharing_large_bottle_with_friends",
        "active_ceps":   ["garrafa grande com amigos"],
        "context":       {"format": "garrafão", "social": True},
    },
    {
        "scenario_name": "outdoor_festival_or_street_event",
        "active_ceps":   ["festivais ao ar livre"],
        "context":       {"setting": "festival"},
    },
    {
        "scenario_name": "watching_sports_with_friends",
        "active_ceps":   ["futebol ou outros esportes"],
        "context":       {"activity": "sports"},
    },
    {
        "scenario_name": "special_occasion",
        "active_ceps":   ["celebrando uma ocasião especial"],
        "context":       {"occasion": "celebration"},
    },
    {
        "scenario_name": "social_dinner",
        "active_ceps":   ["jantar social"],
        "context":       {"setting": "dinner"},
    },
    {
        "scenario_name": "meeting_friends_after_work",
        "active_ceps":   ["encontrando amigos depois do trabalho"],
        "context":       {"setting": "after_work"},
    },
    {
        "scenario_name": "vacation_drinking_more_often",
        "active_ceps":   ["férias"],
        "context":       {"setting": "holiday"},
    },
]

DEMO_UK_SCENARIOS: list[dict] = [
    {
        "scenario_name": "pub_with_friends",
        "active_ceps":   ["meeting your friends at the pub"],
        "context":       {"setting": "pub"},
    },
    {
        "scenario_name": "watching_sport",
        "active_ceps":   ["watching sport"],
        "context":       {"activity": "sport"},
    },
    {
        "scenario_name": "outdoor_hot_day",
        "active_ceps":   ["outdoors on a hot or sunny day"],
        "context":       {"setting": "outdoor", "weather": "hot"},
    },
    {
        "scenario_name": "trendy_bar",
        "active_ceps":   ["trendy bar and want to make"],
        "context":       {"setting": "trendy_bar"},
    },
    {
        "scenario_name": "hosting_at_home",
        "active_ceps":   ["hosting friends at home"],
        "context":       {"setting": "home"},
    },
    {
        "scenario_name": "sociable_dinner",
        "active_ceps":   ["sociable dinner"],
        "context":       {"setting": "dinner"},
    },
    {
        "scenario_name": "out_with_coworkers",
        "active_ceps":   ["out with co-workers"],
        "context":       {"setting": "after_work"},
    },
    {
        "scenario_name": "pre_drinks",
        "active_ceps":   ["heading out for the night"],
        "context":       {"setting": "pre_drinks"},
    },
    {
        "scenario_name": "special_occasion",
        "active_ceps":   ["celebrating a special occasion"],
        "context":       {"occasion": "celebration"},
    },
    {
        "scenario_name": "after_long_day",
        "active_ceps":   ["after a long day"],
        "context":       {"mood": "relaxation"},
    },
]

# ── Ad definitions for the demo ──────────────────────────────────────────────
# These are NOT Ad objects (that would create a circular import with ad_engine).
# The demo notebook imports Ad from ad_engine and uses these dicts to construct them.

DEMO_BRAZIL_ADS: list[dict] = [
    {
        "ad_id":           "heineken_after_work_sports",
        "brand_name":      "Heineken",
        "focal_scenarios": ["meeting_friends_after_work", "watching_sports_with_friends"],
        "secondary_scenarios": ["social_dinner"],
        "branding_clarity":  0.9,
        "channel":         "digital_video",
        "emotion":         "social_warmth",
    },
    {
        "ad_id":           "brahma_neighborhood_sharing",
        "brand_name":      "Brahma",
        "focal_scenarios": ["neighborhood_bar", "sharing_large_bottle_with_friends"],
        "secondary_scenarios": ["hot_sunny_outdoors"],
        "branding_clarity":  0.85,
        "channel":         "tv",
        "emotion":         "pride",
    },
]

DEMO_UK_ADS: list[dict] = [
    {
        "ad_id":           "heineken_pub_sport",
        "brand_name":      "Heineken",
        "focal_scenarios": ["pub_with_friends", "watching_sport"],
        "secondary_scenarios": ["trendy_bar"],
        "branding_clarity":  0.9,
        "channel":         "digital_video",
        "emotion":         "social_warmth",
    },
    {
        "ad_id":           "guinness_evening_occasion",
        "brand_name":      "Guinness",
        "focal_scenarios": ["after_long_day", "special_occasion"],
        "secondary_scenarios": ["sociable_dinner"],
        "branding_clarity":  0.88,
        "channel":         "tv",
        "emotion":         "depth",
    },
]

# Backwards-compat alias
DEMO_ADS = DEMO_BRAZIL_ADS

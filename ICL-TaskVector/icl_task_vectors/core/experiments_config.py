TASKS_TO_EVALUATE = [
    # Knowledge
    "knowledge_country_capital",
    "knowledge_person_language",
    "knowledge_location_continent",
    "knowledge_location_religion",
    # Algorithmic
    "algorithmic_next_letter",
    "algorithmic_prev_letter",
    "algorithmic_list_first",
    "algorithmic_list_last",
    # Bijection
    # "algorithmic_copy",
    "algorithmic_to_upper",
    "algorithmic_to_lower",
    "translation_en_fr",
    "translation_fr_en",
    "translation_en_it",
    "translation_it_en",
    "translation_en_es",
    "translation_es_en",
    "linguistic_present_simple_gerund",
    "linguistic_gerund_present_simple",
    "linguistic_present_simple_past_simple",
    "linguistic_past_simple_present_simple",
    "linguistic_present_simple_past_perfect",
    "linguistic_past_perfect_present_simple",
    "linguistic_singular_plural",
    "linguistic_plural_singular",
    "linguistic_antonyms",
    "bijection_algorithmic_lower_upper",
    "bijection_translation_fr_en",
    "bijection_translation_it_en",
    "bijection_translation_es_en",
    "bijection_linguistic_present_simple_gerund",
    "bijection_linguistic_present_simple_past_simple",
    "bijection_linguistic_present_simple_past_perfect",
    "bijection_linguistic_plural_singular",
    "bijection_linguistic_antonyms",
]

MODELS_TO_EVALUATE = [
    # ("gpt-2", "1.5B"),
    ("pythia", "2.8B"),
    ("llama", "7B"),
    ("gpt-j", "6B"),
    ("pythia", "6.9B"),
    ("llama", "13B"),
    ("pythia", "12B"),
    # ("llama", "30B"),
    # ("mpt", "7B"), # error in ForwardTracer
    # ("falcon", "7B"), # error in past_key_values
]

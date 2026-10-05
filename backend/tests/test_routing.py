"""Deterministic query routing: every mode must be selected without LLM calls."""

from app import routing


def test_normal_fast_path_question_stays_normal():
    plan = routing.plan("What does the evidence guard do before answering?")
    assert plan.mode == routing.NORMAL
    assert plan.uncertain is False


def test_doc_scoped_contribution_routes_to_exhaustive():
    # Top-k retrieval demonstrably misses the contributions paragraph on real papers;
    # a document-scoped list noun must trigger a whole-document sweep instead.
    plan = routing.plan("What is the main contribution of this paper?")
    assert plan.mode == routing.EXHAUSTIVE
    assert plan.exhaustive is True
    assert routing.plan("contribution of this paper").mode == routing.EXHAUSTIVE


def test_summary_of_paper_routes_to_global():
    assert routing.plan("summary of this paper").mode == routing.GLOBAL
    assert routing.plan("summarize this paper").mode == routing.GLOBAL
    assert routing.plan("give me a brief summary of the paper").mode == routing.GLOBAL


def test_typo_summary_routes_to_global():
    # Real users type: "samary of thi paper"
    assert routing.plan("samary of thi paper").mode == routing.GLOBAL


def test_topic_summary_stays_out_of_global():
    assert routing.plan("summarize the caching mechanism").mode == routing.NORMAL


def test_where_mentioned_routes_to_location_not_presence():
    assert routing.plan("Where in the document is caching discussed?").mode == routing.LOCATION
    assert routing.plan("Where is pgvector mentioned?").mode == routing.LOCATION


def test_count_question_routes_to_count():
    plan = routing.plan("How many times does the word graph appear?")
    assert plan.mode == routing.COUNT
    assert plan.uncertain is False


def test_how_many_references_routes_to_reference_not_count():
    plan = routing.plan("How many references are in the bibliography?")
    assert plan.mode == routing.REFERENCE


def test_list_references_routes_to_reference():
    plan = routing.plan("List all references in the bibliography")
    assert plan.mode == routing.REFERENCE


def test_complete_overview_routes_to_global():
    plan = routing.plan("Give a complete overview of this document")
    assert plan.mode == routing.GLOBAL


def test_chapter_question_routes_to_chapter():
    plan = routing.plan("Summarize chapter 2")
    assert plan.mode == routing.CHAPTER
    assert plan.chapter == ("chapter", "2")


def test_comparison_parses_entities():
    plan = routing.plan("Compare alpha-engine and beta-engine")
    assert plan.mode == routing.COMPARISON
    assert plan.entities == ("alpha-engine", "beta-engine")


def test_presence_question_routes_to_presence():
    plan = routing.plan("Does the document mention zebras?")
    assert plan.mode == routing.PRESENCE


def test_location_question_routes_to_location():
    plan = routing.plan("Which pages mention caching?")
    assert plan.mode == routing.LOCATION


def test_quoted_phrase_routes_to_exact():
    plan = routing.plan('Find the exact phrase "cosine similarity"')
    assert plan.mode == routing.EXACT
    assert plan.phrase == "cosine similarity"


def test_metadata_question_routes_to_metadata():
    plan = routing.plan("What is the title of this paper?")
    assert plan.mode == routing.METADATA


def test_quantified_list_question_routes_to_exhaustive():
    plan = routing.plan("List all contributions of this paper")
    assert plan.mode == routing.EXHAUSTIVE
    assert plan.exhaustive is True


def test_definition_question_routes_to_definition():
    plan = routing.plan("What is semantic caching?")
    assert plan.mode == routing.DEFINITION


def test_search_terms_drop_question_and_quantifier_noise():
    assert routing.search_terms("Which pages mention caching?") == ["caching"]
    assert routing.search_terms("List all components of the paper") == ["component"]


def test_search_terms_keep_content_words():
    assert routing.search_terms("How does the evidence checker work?") == [
        "evidence",
        "checker",
        "work",
    ]


def test_stem_like_normalizes_plurals():
    assert routing.stem_like("contributions") == "contribution"
    assert routing.stem_like("findings") == "finding"
    assert routing.stem_like("analysis") == "analysis"


def test_plan_never_raises_on_odd_input():
    for question in ("", "   ", "?", "12345", "мир PDF", "????"):
        assert routing.plan(question).mode in routing.MODES


def test_format_instructions_exist_for_every_mode():
    for mode in routing.MODES:
        assert routing.format_instructions(mode)


def test_uncertain_plan_needs_llm_but_falls_back(monkeypatch):
    plan = routing.plan("Something complete about several things")
    if not plan.uncertain:
        return
    # LLM unavailable → NORMAL, never an exception.
    from app import llm

    def _boom(*args, **kwargs):
        raise llm.LLMError("no api")

    monkeypatch.setattr(llm, "chat", _boom)
    assert routing.classify_uncertain("Something complete about several things") == routing.NORMAL

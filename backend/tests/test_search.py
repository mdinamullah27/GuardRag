"""Pure-function tests for exact search, counting, and document structure helpers."""

from app import search


def test_count_in_text_uses_word_boundaries():
    assert search.count_in_text("A cat catalog", "cat") == 1
    assert search.count_in_text("Cat cat CAT", "cat") == 3


def test_count_in_text_substring_for_phrases():
    assert search.count_in_text("cosine similarity and cosine distance", "cosine similarity") == 1


def test_merge_overlapping_removes_duplicated_overlap():
    merged = search.merge_overlapping(["alpha beta gamma", "beta gamma delta"])
    assert merged == "alpha beta gamma delta"


def test_merge_overlapping_without_overlap_joins():
    assert search.merge_overlapping(["one", "two"]) == "one\ntwo"


def test_find_labeled_sections_and_range():
    pages = {
        1: "Chapter 1\nIntro material here.",
        2: "More of chapter one.",
        3: "Chapter 2\nSecond chapter body.",
        4: "Second chapter continued.",
        5: "Chapter 3\nThird chapter body.",
    }
    sections = search.find_labeled_sections(pages)
    assert sections == [("chapter", "1", 1), ("chapter", "2", 3), ("chapter", "3", 5)]
    assert search.page_range_for(pages, "chapter", "2") == (3, 4)
    assert search.page_range_for(pages, "chapter", "3") == (5, 5)
    assert search.page_range_for(pages, "chapter", "9") is None


def test_bibliography_range_finds_heading_page():
    pages = {1: "Body text.", 2: "More body.", 3: "References\n[1] Entry one."}
    assert search.bibliography_range(pages) == (3, 3)
    assert search.bibliography_range({1: "Body only."}) is None


def test_bib_entries_parses_numbered_and_wrapped_entries():
    text = (
        "References\n"
        "[1] Smith, J. (2020). Graph neural networks for retrieval. Journal of AI, 12(3), 45-67.\n"
        "[2] Lee, K. (2021). Dense passage retrieval improvements and analysis. ACL, 2021, 1-12.\n"
        "[3] Chen, W. (2022). Evidence grounded generation with guards. NeurIPS 2022, 3344-3355.\n"
    )
    entries = search.bib_entries(text)
    assert len(entries) == 3
    assert entries[0].startswith("[1] Smith")
    assert "NeurIPS 2022" in entries[2]


def test_bib_entries_joins_indented_continuation_lines():
    text = (
        "[1] Smith, J. (2020). Graph neural networks for retrieval.\n"
        "    Journal of AI, 12(3), 45-67.\n"
        "[2] Lee, K. (2021). Dense passage retrieval improvements. ACL, 2021, 1-12.\n"
    )
    entries = search.bib_entries(text)
    assert len(entries) == 2
    assert "Journal of AI" in entries[0]


def test_count_bibliography_entries_ignores_short_lines():
    text = "References\n[1] A very long bibliography entry that easily exceeds forty characters.\n"
    assert search.count_bibliography_entries(text) == 1


def test_escape_escapes_like_wildcards():
    assert search._escape("100%_x") == r"100\%\_x"

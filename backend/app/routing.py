"""Deterministic query routing.

Maps a user question to a retrieval strategy class without spending LLM calls in the
common case. An LLM fallback is used only for low-confidence ("uncertain") inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# --- modes ------------------------------------------------------------------

NORMAL = "NORMAL"
EXHAUSTIVE = "EXHAUSTIVE"
COUNT = "COUNT"
EXACT = "EXACT"
PRESENCE = "PRESENCE"
LOCATION = "LOCATION"
GLOBAL = "GLOBAL"
CHAPTER = "CHAPTER"
COMPARISON = "COMPARISON"
MULTI_HOP = "MULTI_HOP"
METADATA = "METADATA"
REFERENCE = "REFERENCE"
DEFINITION = "DEFINITION"

MODES = (
    NORMAL, EXHAUSTIVE, COUNT, EXACT, PRESENCE, LOCATION, GLOBAL,
    CHAPTER, COMPARISON, MULTI_HOP, METADATA, REFERENCE, DEFINITION,
)

_LIST_NOUNS = (
    "contribution", "contributions", "limitation", "limitations", "objective",
    "objectives", "dataset", "datasets", "algorithm", "algorithms", "experiment",
    "experiments", "conclusion", "conclusions", "recommendation", "recommendations",
    "chapter", "chapters", "section", "sections", "finding", "findings", "theme",
    "themes", "result", "results", "method", "methods", "example", "examples",
    "type", "types", "reason", "reasons", "benefit", "benefits", "issue", "issues",
    "problem", "problems", "question", "questions", "step", "steps", "option",
    "options", "criterion", "criteria", "assumption", "assumptions", "author",
    "authors", "reference", "references", "citation", "citations", "figure",
    "figures", "table", "tables", "definition", "definitions", "term", "terms",
)

_QUANTIFIERS = re.compile(
    r"\b(all|every|each|entire|whole|enumerate|complete(?:\s+list)?|full\s+list|"
    r"exhaustive|throughout|across the)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"“']([^\"“”'’]{2,120})[\"”']")
_QUOTED_STRICT = re.compile(r"[\"“]([^\"“”]{2,120})[\"”]")
_CHAPTER_RE = re.compile(r"\b(chapter|section|part|lesson|lecture)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

# Whole-document summary detection: strict patterns plus a typo-tolerant fallback
# ("samary of thi paper" must still route to GLOBAL).
_SUMMARY_WORDS = (
    "summary", "summaries", "summarize", "summarized", "summarise", "summarised",
    "overview", "synopsis",
)
_DOC_SCOPE_RE = re.compile(
    r"\b(?:of|in|from|across)\s+(?:the\s+)?(?:this|these|that|the|above|foregoing|given)?\s*"
    r"(?:paper|document|book|report|article|pdf|study|thesis|manuscript|publication)\b",
    re.IGNORECASE,
)
# Curated nouns that, when scoped to the whole document, deserve a full-text sweep
# rather than top-k retrieval (authors/titles/DOIs are handled by METADATA first).
_EXHAUSTIVE_DOC_NOUNS = (
    "contribution", "limitation", "objective", "conclusion", "finding",
    "recommendation", "result", "method", "theme", "example", "experiment",
    "dataset", "issue", "problem", "benefit", "assumption", "approach",
)

_COMPARISON_RES = (
    re.compile(r"\bcompare\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bdifference\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\b(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
)

_DOC_WORDS = frozenset(
    """the a an of in on at to for from by with and or is are was were be been being this
    that these those it its as if then than so such not no do does did can could will
    would should paper document book report pdf file text content question answer answers
    author authors write written make makes made main key important various several
    which what where when why how many times pages page find found show give list name
    every each all any entire whole enumerate complete full exhaustive throughout across
    mention mentions mentioned discuss discusses discussed contain contains contained
    include includes included cover covers covered refer refers referred appear appears
    word words number numbers often
    """.split()
)


@dataclass
class QueryPlan:
    mode: str = NORMAL
    exhaustive: bool = False
    uncertain: bool = False
    phrase: str | None = None
    entities: tuple[str, str] | None = None
    chapter: tuple[str, str] | None = None  # (kind, number)
    terms: list[str] = field(default_factory=list)


def _has_quantifier(text: str) -> bool:
    return bool(_QUANTIFIERS.search(text))


def _mentions_list_context(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(list|enumerate|identify|name|give|show|find|what are|which are|what)\b", lowered):
        return True
    return any(re.search(rf"\b{re.escape(noun)}s?\b", lowered) for noun in _LIST_NOUNS)


def stem_like(word: str) -> str:
    """Very small plural normalizer so '%contribution%' matches 'contributions'."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("es") and w[:-2].endswith(("ch", "sh", "x", "z", "s")):
        return w[:-2]
    if w.endswith("is"):
        return w  # singulars like "analysis", "basis"
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def search_terms(question: str, extra_stop: set[str] | None = None) -> list[str]:
    """Content terms for exact/term search (used by exhaustive/location/presence paths)."""
    stop = _DOC_WORDS | (extra_stop or set())
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']{2,}", question.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in stop:
            continue
        stem = stem_like(word)
        if stem in stop or len(stem) < 3:
            continue
        if stem not in seen:
            seen.add(stem)
            terms.append(stem)
    return terms


def _parse_comparison(question: str) -> tuple[str, str] | None:
    for pattern in _COMPARISON_RES:
        match = pattern.search(question)
        if not match:
            continue
        left = match.group(1).strip(" ,;:")
        right = match.group(2).strip(" ,;:")
        filler = re.compile(
            r"\b(?:which\s+is\s+\w+|in\s+this\s+\w+|of\s+the\s+\w+)\s*$", re.IGNORECASE
        )
        left = filler.sub("", left).strip(" ,;:")
        right = filler.sub("", right).strip(" ,;:")
        if left and right and left.lower() != right.lower() and len(left) < 80 and len(right) < 80:
            return left, right
    return None


def _is_definition_like(question: str) -> bool:
    lowered = question.lower().strip()
    if re.search(r"\b(define|definition of|is defined as|how does \w+ define|how is \w+ defined)\b", lowered):
        return True
    words = lowered.split()
    if len(words) > 7:
        return False
    if not re.match(r"^(what is|what's)\s+(a |an |the )?[\w\- ]{2,40}\??$", lowered):
        return False
    return not re.search(
        r"\b(main|key|purpose|goal|problem|contribution|difference|role|best|highest)\b", lowered
    )


def _is_global_summary(lowered: str) -> bool:
    """True for whole-document summary/overview questions (typo-tolerant)."""
    if re.search(r"\bchapter\s+\d|section\s+\d", lowered):
        return False
    if re.search(
        r"\bsummar(?:y|ies|ize|ise|ized|ised)\b[^.?!]*\b(?:entire|whole|full|complete|book|document|report|paper|article)\b"
        r"|\boverview of\b|\bwhat is this (?:paper|book|document|report|article) about\b"
        r"|\bmajor themes\b|\bmain conclusions of the\b|\bcomplete overview\b"
        r"|\bsummar(?:y|ies|ize|ise) (?:every|all) (?:chapter|section)s?\b"
        r"|\bwhat changed throughout\b|\bstructure of (?:the|this) (?:book|document|paper)\b"
        r"|\bgive (?:me )?(?:a |an |the )?(?:short |brief |quick )?summar(?:y|ies|ize|ise)\b"
        r"|\b(?:short|brief|quick|overall|general) (?:summary|overview)\b",
        lowered,
    ):
        return True
    # Typo-tolerant fallback: fuzzy match on summary vocabulary, but only when the
    # question is actually document-scoped (so "summarize the caching section" stays out).
    if not re.search(r"\b(?:paper|document|book|report|article|pdf|thesis|everything|it)\b", lowered):
        return False
    return any(
        SequenceMatcher(None, token, word).ratio() >= 0.75
        for token in re.findall(r"[a-z]{5,}", lowered)
        for word in _SUMMARY_WORDS
    )


def _doc_scoped_list_question(lowered: str) -> bool:
    """'contribution(s) of this paper' → whole-document sweep, not top-k."""
    if not _DOC_SCOPE_RE.search(lowered):
        return False
    return any(
        re.search(rf"\b{re.escape(noun)}s?\b", lowered)
        for noun in _EXHAUSTIVE_DOC_NOUNS
    )


def plan(question: str) -> QueryPlan:
    """Classify a question. Never raises; unknown inputs default to NORMAL."""
    q = question.strip()
    lowered = q.lower()

    quoted_match = _QUOTED_STRICT.search(q) or _QUOTED.search(q)
    phrase = quoted_match.group(1).strip() if quoted_match else None
    quantified = _has_quantifier(q)

    result = QueryPlan(
        phrase=phrase,
        exhaustive=quantified and _mentions_list_context(q),
        terms=search_terms(q),
    )

    # --- high-precision signals, most specific first ------------------------
    if re.search(r"\b(compare|contrast)\b|\bdifference between\b|\bversus\b|\bvs\.?\b", lowered) or re.search(
        r"\bwhich\b.*\b(better|best|higher|faster|more accurate)\b", lowered
    ):
        entities = _parse_comparison(q)
        if entities is not None:
            result.mode = COMPARISON
            result.entities = entities
            return result
        # fall through to other signals when entities cannot be parsed

    if re.search(
        r"\bthat\s+(?:achieved|achieves|achieved the|won|got|has the|recorded)\b"
        r"|\b(?:author|method|model|paper|approach|system|technique)\s+that\b"
        r"|\bon which dataset\b|\bwhich dataset",
        lowered,
    ):
        result.mode = MULTI_HOP
        return result

    if re.search(
        r"\b(?:what is the (?:title|doi|isbn)|who (?:is|are) the (?:author|authors|writer)"
        r"|what year (?:was|is|were)|which (?:journal|conference|publisher|university|institution)"
        r"|affiliated with\b|affiliation\b|publication (?:year|date)|\bdoi\b|\bisbn\b"
        r"|when (?:was|is) (?:it|this|the paper|the document) published\b)",
        lowered,
    ):
        result.mode = METADATA
        return result

    if _is_global_summary(lowered):
        result.mode = GLOBAL
        return result

    chapter_match = _CHAPTER_RE.search(q)
    if chapter_match:
        result.mode = CHAPTER
        result.chapter = (chapter_match.group(1).lower(), chapter_match.group(2))
        return result

    if re.search(r"\b(references|bibliography|works cited|citations)\b", lowered):
        if (
            quantified
            or re.search(r"\bhow many\b", lowered)
            or re.search(r"\b(list|give|which|relevant|related|what|who|where)\b", lowered)
        ):
            result.mode = REFERENCE
            return result

    if re.search(
        r"\bhow many times\b|\bhow often\b|\bcount (?:the )?(?:occurrences|mentions|instances|times)\b"
        r"|\bnumber of times\b|\bhow many (?:mentions|occurrences|times|instances)\b"
        r"|\bhow many\b",
        lowered,
    ):
        result.mode = COUNT
        return result

    if re.match(
        r"^(?:does|do|did|is|are|was|were|can|has|have)\b[^?]{0,80}\b"
        r"(?:mention|mentions|mentioned|discuss|discusses|discussed|contain|contains|contained"
        r"|include|includes|included|cover|covers|covered|address|addresses|refer to|references"
        r"|use|uses|used|appear|appears|talk about)\b",
        lowered,
    ) or re.search(r"\bis\b[^?]{0,40}\b(?:mentioned|discussed|included|covered|present)\b", lowered):
        # A "where/which pages" question wants locations, not a yes/no.
        if not re.search(
            r"^(?:where|when|which|on which)\b|\b(?:which|what|on which|how many) pages?\b",
            lowered,
        ):
            result.mode = PRESENCE
            return result

    if re.search(
        r"\bwhere (?:in|throughout|within)\b|\bwhich pages?\b|\bon which pages?\b"
        r"|\blist (?:the |all )?pages\b|\bevery page\b|\bfind all occurrences\b"
        r"|\ball occurrences\b|\bwhere throughout\b|\bwhich chapters?\b[^?]{0,40}\b(?:discuss|mention|cover)"
        r"|\bwhere\b[^?]{0,80}\b(?:mention(?:s|ed)?|discuss(?:ed|es)?|cover(?:s|ed)?|appear(?:s|ed)?|found|located|can i find)\b",
        lowered,
    ):
        result.mode = LOCATION
        return result

    if result.exhaustive and _mentions_list_context(q) and _has_quantifier(q):
        result.mode = EXHAUSTIVE
        return result

    if phrase:
        result.mode = EXACT
        return result

    if _doc_scoped_list_question(lowered):
        result.mode = EXHAUSTIVE
        result.exhaustive = True
        return result

    if _is_definition_like(q):
        result.mode = DEFINITION
        result.exhaustive = False
        return result

    if quantified and _mentions_list_context(q):
        # Quantifier but weak context: worth one LLM classification.
        result.uncertain = True
        return result

    return result


_LLM_CLASSIFY_PROMPT = """\
Classify a PDF question into exactly one mode. Reply with JSON only:
{"mode": "..."}

Modes:
- EXHAUSTIVE: the user wants a COMPLETE list ("all", "every", "enumerate").
- COUNT: wants an exact number of occurrences.
- GLOBAL: wants a whole-document summary/overview.
- CHAPTER: targets a specific chapter/section/part.
- COMPARISON: compares two entities.
- MULTI_HOP: needs facts from different places connected ("which method achieved X on which dataset").
- METADATA: bibliographic facts (title, authors, year, DOI, venue).
- PRESENCE: asks whether something is mentioned at all.
- LOCATION: asks where/on which pages something appears.
- REFERENCE: bibliography/citation related.
- DEFINITION: asks what a term means.
- NORMAL: a normal factual question about content.

Question: {question}
"""


def classify_uncertain(question: str) -> str:
    """LLM fallback for low-confidence inputs. Returns NORMAL on any failure."""
    from . import llm
    from .schemas import ModeChoice

    try:
        verdict = llm.chat_json(
            [{"role": "user", "content": _LLM_CLASSIFY_PROMPT.format(question=question)}],
            ModeChoice,
        )
    except Exception:  # noqa: BLE001 - routing must never fail the request
        return NORMAL
    mode = getattr(verdict, "mode", NORMAL)
    return mode if mode in MODES else NORMAL


def format_instructions(mode: str) -> str:
    """Answer-format guidance injected into the final answer prompt."""
    return _FORMAT_INSTRUCTIONS.get(mode, _FORMAT_INSTRUCTIONS[NORMAL])


_FORMAT_INSTRUCTIONS: dict[str, str] = {
    NORMAL: "Answer concisely and directly. Cite the page numbers you used as [p. N].",
    DEFINITION: (
        "Give the document's explicit definition first (quote or closely paraphrase it), "
        "then any broader explanation. Cite pages as [p. N]."
    ),
    EXHAUSTIVE: (
        "Return a COMPLETE numbered list. Every item must be supported by the provided "
        "evidence and carry its own [p. N] citation. Do not silently drop items. State up "
        "front how much of the document was searched; if coverage is partial, say so explicitly."
    ),
    COUNT: (
        "State the exact number first. Then explain the counting methodology (what was "
        "searched, case sensitivity) and list the pages. Never estimate from partial context."
    ),
    EXACT: (
        "Quote the exact matching text snippets with their page numbers [p. N]. "
        "If nothing matches, say the phrase was not found in the extracted text."
    ),
    PRESENCE: (
        "Answer yes or no explicitly, quoting the matching text when present with [p. N]. "
        "When nothing matches, say 'I could not find evidence of that in the extracted "
        "document' — do not claim the document definitely does not contain it."
    ),
    LOCATION: (
        "List every matching location as page numbers with a one-line snippet each, "
        "then the total count of locations."
    ),
    GLOBAL: (
        "Return a structured summary with short sections (scope, key points, conclusions). "
        "Base it strictly on the provided document summary/evidence."
    ),
    CHAPTER: "Summarize only the requested section, preserving its structure. Cite [p. N].",
    COMPARISON: (
        "Compare BOTH entities side by side; a markdown table is welcome. Cover each side "
        "even when one side has less evidence, and say so when evidence is missing for a side."
    ),
    MULTI_HOP: (
        "Connect the retrieved facts step by step and give the final answer. "
        "Cite each supporting fact with [p. N]."
    ),
    METADATA: (
        "Report each fact with its source (PDF metadata or page). If a fact is absent, "
        "say it could not be found — do not guess."
    ),
    REFERENCE: (
        "Return a numbered reference list. Include every match found; cite the pages where "
        "each reference or citation appears [p. N]."
    ),
}

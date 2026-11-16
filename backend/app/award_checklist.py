"""Find explicit selection criteria and omission statements in award evidence."""

import re
from typing import Literal

from pydantic import BaseModel

from .claims import Citation

CRITERIA_LABEL = r"(?:selection|judging|evaluation)[ \t]+criteria"
OMISSION_VALUE = r"(?:not[ \t]+(?:provided|included|disclosed|specified|stated|available)|unavailable|undisclosed|none|n/a)"
OMISSION = re.compile(
    rf"\b(?:this[ \t]+(?:letter|document|notice)[ \t]+)?"
    rf"(?:does[ \t]+not[ \t]+(?:include|state|describe|disclose)|omits)[ \t]+"
    rf"(?:the[ \t]+)?{CRITERIA_LABEL}\b"
    rf"|\b{CRITERIA_LABEL}(?:[ \t]*:[ \t]*|[ \t]+(?:are|were)[ \t]+){OMISSION_VALUE}\b",
    re.IGNORECASE,
)
CRITERIA_DESCRIPTION = re.compile(
    rf"^[ \t]*(?:{CRITERIA_LABEL}[ \t]*:[ \t]*"
    r"|judges[ \t]+(?:scored|evaluated)[ \t]+(?:entries|candidates)[ \t]+(?:on|based on)[ \t]+)"
    r"(?P<details>[^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)


class AwardCriteriaCheck(BaseModel):
    kind: Literal["award_selection_criteria"] = "award_selection_criteria"
    award_name: str
    status: Literal["found", "not_found", "needs_review"]
    criteria_citations: list[Citation]
    limitation_citations: list[Citation]
    unreadable_page_count: int


def award_criteria_passages(page_text: str, award_name: str) -> tuple[list[str], list[str]]:
    page = " ".join(page_text.casefold().split())
    award = " ".join(award_name.casefold().split())
    if not re.search(rf"(?<!\w){re.escape(award)}(?!\w)", page):
        return [], []

    criteria = [
        match.group().strip()
        for match in CRITERIA_DESCRIPTION.finditer(page_text)
        if match["details"].strip() and not OMISSION.search(match.group())
    ]
    limitations = [match.group() for match in OMISSION.finditer(page_text)]
    return criteria, limitations


def summarize_award_check(
    award_name: str,
    criteria_citations: list[Citation],
    limitation_citations: list[Citation],
    unreadable_page_count: int,
) -> AwardCriteriaCheck:
    status: Literal["found", "not_found", "needs_review"] = "not_found"
    if criteria_citations:
        status = "found"
    elif unreadable_page_count:
        status = "needs_review"
    return AwardCriteriaCheck(
        award_name=award_name,
        status=status,
        criteria_citations=criteria_citations,
        limitation_citations=limitation_citations,
        unreadable_page_count=unreadable_page_count,
    )

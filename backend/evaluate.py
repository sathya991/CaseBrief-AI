"""Run labeled synthetic evidence cases against the running local app."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
import pymupdf
from pydantic import ValidationError

from app.claims import Citation
from app.main import CaseReviewResponse, ClaimExtractionResponse

SAMPLES = Path(__file__).resolve().parents[1] / "sample_documents"
RESUME = "01_alex_morgan_resume.pdf"
EMPLOYMENT_LETTER = "02_employment_confirmation.pdf"
AWARD_LETTER = "03_award_letter.pdf"
AWARD = "National Technology Innovation Award"
RECIPIENT = "Alex Morgan"
EMPLOYER = "Lumina Labs"

AWARD_FACT = f"{RECIPIENT} was selected for the 2023 {AWARD}"
ALLOWED_AWARD_STATEMENTS = {
    f"{AWARD_FACT}.",
    f"{AWARD_FACT} for work in reliable ML infrastructure.",
    f"The letter states that {AWARD_FACT}.",
    f"The letter states that {AWARD_FACT} for work in reliable ML infrastructure.",
    f"The Technology Research Council congratulated {RECIPIENT} on being selected "
    f"for the 2023 {AWARD} for work in reliable ML infrastructure.",
}
ALLOWED_DATE_STATEMENTS = {
    "The award letter is dated October 9, 2023.",
    "The award notification is dated October 9, 2023.",
    "The letter is dated October 9, 2023.",
}
EXPECTED_STARTS = {
    "2022-03": (RESUME, "March 2022"),
    "2022-06": (EMPLOYMENT_LETTER, "June 2022"),
}


@dataclass
class CaseResult:
    name: str
    finding_errors: list[str] = field(default_factory=list)
    citation_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.finding_errors or self.citation_errors)


@lru_cache(maxsize=3)
def first_page(filename: str) -> str:
    with pymupdf.open(SAMPLES / filename) as pdf:
        return pdf[0].get_text("text", sort=True).strip()


def check_citation(
    result: CaseResult,
    citation: Optional[Citation],
    documents_by_id: dict[str, str],
    expected_file: str,
    required_text: tuple[str, ...],
    label: str,
) -> None:
    if citation is None:
        result.citation_errors.append(f"{label}: missing citation.")
        return
    if (
        citation.filename != expected_file
        or documents_by_id.get(citation.document_id) != expected_file
        or citation.page_number != 1
    ):
        result.citation_errors.append(f"{label}: expected {expected_file}, page 1.")
        return
    if not citation.quote.strip() or citation.quote not in first_page(expected_file):
        result.citation_errors.append(f"{label}: quote is not verbatim on the original PDF page.")
        return
    if any(fragment not in citation.quote for fragment in required_text):
        result.citation_errors.append(f"{label}: quote omits part of the labeled evidence.")


def score_award(response: ClaimExtractionResponse) -> CaseResult:
    result = CaseResult("Award extraction")
    documents = {response.document.document_id: response.document.filename}
    if sorted(claim.category for claim in response.claims) != ["award", "date"]:
        result.finding_errors.append("Expected one award claim and one letter-date claim.")
    if not {"award", "date"}.issubset(claim.category for claim in response.claims):
        result.citation_errors.append("The award and letter date each need cited evidence.")

    for claim in response.claims:
        if claim.category == "award":
            allowed = ALLOWED_AWARD_STATEMENTS
            fragments = (RECIPIENT, "being selected", "2023 National\nTechnology Innovation Award")
        elif claim.category == "date":
            allowed = ALLOWED_DATE_STATEMENTS
            fragments = ("Date\nOctober 9, 2023",)
        else:
            continue
        if " ".join(claim.statement.split()) not in allowed:
            result.finding_errors.append(f"Unexpected {claim.category} statement: {claim.statement}")
        citation = claim.citation if claim.quote_status == "matched" and claim.source_page == 1 else None
        check_citation(result, citation, documents, AWARD_LETTER, fragments, claim.category)
    return result


def score_employment(response: CaseReviewResponse) -> CaseResult:
    result = CaseResult("Employment start-month discrepancy")
    documents = {document.document_id: document.filename for document in response.documents}
    flags = response.possible_discrepancies
    if len(flags) != 1:
        result.finding_errors.append(f"Expected exactly one discrepancy, got {len(flags)}.")
    if not flags:
        result.citation_errors.append("No discrepancy citations to verify.")

    for flag in flags:
        months = sorted(item.start_month for item in flag.observations)
        if (flag.person, flag.employer) != (RECIPIENT, EMPLOYER) or months != sorted(EXPECTED_STARTS):
            result.finding_errors.append(f"Unexpected discrepancy for {flag.person} at {flag.employer}: {months}.")
        if len(flag.observations) != 2:
            result.citation_errors.append("Expected both observations to have citations.")
        for observation in flag.observations:
            if (observation.person, observation.employer) != (flag.person, flag.employer):
                result.finding_errors.append("An observation has a different person or employer from its discrepancy.")
            expected = EXPECTED_STARTS.get(observation.start_month)
            if expected is None:
                result.citation_errors.append(f"No labeled source for month {observation.start_month}.")
                continue
            filename, date_text = expected
            check_citation(
                result, observation.citation, documents, filename,
                (RECIPIENT, EMPLOYER, date_text), observation.start_month,
            )
    return result


def score_checklist(response: CaseReviewResponse) -> CaseResult:
    result = CaseResult("Award selection criteria checklist")
    documents = {document.document_id: document.filename for document in response.documents}
    checks = response.evidence_checks
    if len(checks) != 1 or (checks[0].award_name, checks[0].status) != (AWARD, "not_found"):
        result.finding_errors.append("Expected one not_found result for the named award.")
    citations = [citation for check in checks for citation in check.limitation_citations]
    if len(citations) != 1 or any(check.criteria_citations for check in checks):
        result.citation_errors.append("Expected one omission citation and no criteria citations.")
    for citation in citations:
        check_citation(
            result, citation, documents, AWARD_LETTER,
            ("This letter does not include the selection criteria",), "omission",
        )
    return result


def pdf_part(field_name: str, filename: str) -> tuple[str, tuple[str, bytes, str]]:
    return field_name, (filename, (SAMPLES / filename).read_bytes(), "application/pdf")


def require_success(response: httpx.Response) -> None:
    if response.is_success:
        return
    raise RuntimeError(f"HTTP {response.status_code} from {response.request.url.path}: {response.text[:400]}")


def run_evaluation(client: httpx.Client) -> list[CaseResult]:
    try:
        response = client.post("/api/documents/claims", files=[pdf_part("file", AWARD_LETTER)])
        require_success(response)
        award = score_award(ClaimExtractionResponse.model_validate_json(response.content))
    except (httpx.HTTPError, OSError, RuntimeError, ValidationError) as error:
        award = CaseResult("Award extraction", [str(error)], ["No response to score."])

    try:
        response = client.post(
            "/api/documents/review",
            files=[pdf_part("files", name) for name in (RESUME, EMPLOYMENT_LETTER, AWARD_LETTER)],
            data={"award_selection_criteria_for": AWARD},
        )
        require_success(response)
        review = CaseReviewResponse.model_validate_json(response.content)
    except (httpx.HTTPError, OSError, RuntimeError, ValidationError) as error:
        employment = CaseResult("Employment start-month discrepancy", [str(error)], ["No response to score."])
        checklist = CaseResult("Award selection criteria checklist", [str(error)], ["No response to score."])
    else:
        employment = score_employment(review)
        checklist = score_checklist(review)
    return [award, employment, checklist]


def main(port: int = 8000) -> int:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=240, trust_env=False) as client:
        results = run_evaluation(client)
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'}  {result.name}")
        for label, errors in (("Findings", result.finding_errors), ("Citations", result.citation_errors)):
            print(f"  {'FAIL' if errors else 'PASS'}  {label}")
            for error in errors:
                print(f"    {error}")
    passed = sum(result.passed for result in results)
    print(f"\n{passed}/{len(results)} cases passed.")
    return int(passed != len(results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Local FastAPI port (default: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    raise SystemExit(main(args.port))

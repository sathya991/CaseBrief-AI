import pytest

import evaluate
from app.claims import CheckedClaim, Citation
from app.main import ClaimExtractionResponse, extract_pdf


@pytest.fixture
def award_response():
    filename = evaluate.AWARD_LETTER
    document = extract_pdf((evaluate.SAMPLES / filename).read_bytes(), filename)
    page = document.pages[0]
    award_quote = next(
        paragraph for paragraph in page.text.split("\n\n")
        if paragraph.startswith("Statement\n")
    ).removeprefix("Statement\n")
    claims = []
    for category, statement, quote in (
        ("award", f"The letter states that {evaluate.AWARD_FACT}.", award_quote),
        ("date", "The award notification is dated October 9, 2023.", "Date\nOctober 9, 2023"),
    ):
        claims.append(
            CheckedClaim(
                category=category,
                statement=statement,
                source_page=1,
                quote_status="matched",
                citation=Citation(
                    document_id=document.document_id,
                    filename=filename,
                    page_number=1,
                    quote=quote,
                ),
            )
        )
    return ClaimExtractionResponse(model="mock", document=document, claims=claims)


def test_supported_award_and_exact_citations_pass(award_response):
    assert evaluate.score_award(award_response).passed


def test_exact_award_citation_does_not_pass_an_overstated_finding(award_response):
    award_response.claims[0].statement = (
        "The Technology Research Council selected Alex Morgan for the 2023 "
        "National Technology Innovation Award."
    )

    result = evaluate.score_award(award_response)
    assert result.finding_errors
    assert result.citation_errors == []
    assert not result.passed


def test_edited_quote_in_the_api_response_does_not_pass_citation_check(award_response):
    citation = award_response.claims[0].citation
    citation.quote += " Selection was unanimous."
    award_response.document.pages[0].text += "\n" + citation.quote

    result = evaluate.score_award(award_response)
    assert result.finding_errors == []
    assert "not verbatim" in result.citation_errors[0]
    assert not result.passed


@pytest.mark.parametrize("failed", [False, True])
def test_exit_status_tracks_evaluation_result(monkeypatch, capsys, failed):
    result = evaluate.CaseResult("Synthetic fixture")
    if failed:
        result.finding_errors.append("An unsupported detail")
    monkeypatch.setattr(evaluate, "run_evaluation", lambda client: [result])

    assert evaluate.main() == int(failed)
    output = capsys.readouterr().out
    assert ("FAIL  Findings" if failed else "PASS  Findings") in output
    assert "PASS  Citations" in output

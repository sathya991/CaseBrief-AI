import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[2] / "sample_documents"
RESUME = "01_alex_morgan_resume.pdf"
LETTER = "02_employment_confirmation.pdf"


def mock_ollama(monkeypatch, page_results):
    responses = iter(page_results)

    async def fake_post(self, url, *, json):
        assert url == "http://127.0.0.1:11434/api/generate"
        assert "employment_starts" in json["format"]["properties"]
        return httpx.Response(
            200,
            json={"done": True, "response": next(responses)},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


def employment_proposal(*, person="Alex Morgan", employer="Lumina Labs", date="March 2022"):
    return {"person": person, "employer": employer, "start_date_text": date}


def model_result(*proposals):
    return json.dumps({"employment_starts": proposals})


def review(*documents):
    return client.post(
        "/api/documents/review",
        files=[("files", (name, pdf, "application/pdf")) for name, pdf in documents],
    )


def employment_pdf(person, employer, date):
    quote = f"Employee: {person}\nEmployer: {employer}\nStart Date: {date}"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 100), quote)
        return pdf.tobytes()


def test_review_flags_the_sample_discrepancy_with_both_citations(monkeypatch):
    resume = (SAMPLES / RESUME).read_bytes()
    letter = (SAMPLES / LETTER).read_bytes()
    with pymupdf.open(stream=resume, filetype="pdf") as pdf:
        text = pdf[0].get_text("text", sort=True)
        resume_quote = text[text.index("Alex Morgan (fictional)"):text.index(" - Present")]
    letter_quote = "Alex Morgan\n\nEmployer\nLumina Labs (fictional company)\n\nStart Date\nJune 2022"
    resume_proposal = employment_proposal()
    letter_proposal = employment_proposal(
        employer="Lumina Labs (fictional company)",
        date="June 2022",
    )
    mock_ollama(
        monkeypatch,
        [
            model_result(resume_proposal, resume_proposal),
            model_result(letter_proposal),
        ],
    )

    response = review((RESUME, resume), (LETTER, letter))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["evidence_checks"] == []
    [flag] = payload["possible_discrepancies"]
    assert flag["kind"] == "employment_start_month_mismatch"
    assert flag["person"] == "Alex Morgan"
    assert flag["employer"] == "Lumina Labs"
    assert [start["start_month"] for start in flag["observations"]] == ["2022-03", "2022-06"]
    for observation, document, expected_quote in zip(
        flag["observations"], payload["documents"], [resume_quote, letter_quote]
    ):
        citation = observation["citation"]
        assert citation == {
            "document_id": document["document_id"],
            "filename": document["filename"],
            "page_number": 1,
            "quote": expected_quote,
        }
        assert citation["quote"] in document["pages"][0]["text"]


@pytest.mark.parametrize(
    ("person", "employer", "date", "proposal_overrides", "expected_count"),
    [
        ("Alex Morgan", "Lumina Labs", "March 2022", {}, 2),
        ("Alex Morgan", "Halcyon Systems", "June 2022", {}, 2),
        ("Sam Patel", "Lumina Labs", "June 2022", {}, 2),
        ("Alex Morgan", "Lumina Labs", "June 2022", {"date": "July 2022"}, 1),
        ("Alex Morgan", "Lumina Labs", "June 2022", {"employer": "umina"}, 1),
        ("Alex Morgan", "Lumina Labs", "June 2022", {"person": "Sam Patel"}, 1),
        ("Alex Morgan", "Lumina Labs", "2022", {}, 1),
    ],
)
def test_review_ignores_matches_unrelated_jobs_and_unanchored_values(
    monkeypatch, person, employer, date, proposal_overrides, expected_count
):
    first_pdf = employment_pdf("Alex Morgan", "Lumina Labs", "March 2022")
    second_pdf = employment_pdf(person, employer, date)
    fields = {"person": person, "employer": employer, "date": date}
    fields.update(proposal_overrides)
    mock_ollama(
        monkeypatch,
        [model_result(employment_proposal()), model_result(employment_proposal(**fields))],
    )

    response = review(("resume.pdf", first_pdf), ("letter.pdf", second_pdf))
    assert response.status_code == 200, response.text
    assert len(response.json()["employment_starts"]) == expected_count
    assert response.json()["possible_discrepancies"] == []

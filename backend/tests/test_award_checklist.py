from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES = Path(__file__).resolve().parents[2] / "sample_documents"
AWARD_LETTER = "03_award_letter.pdf"
AWARD = "National Technology Innovation Award"
OMISSION = "This letter does not include the selection criteria"
CRITERIA = "Judges scored entries on originality and reliability."
client = TestClient(app)


def make_pdf(text):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 100), text)
        return pdf.tobytes()


@pytest.mark.parametrize(
    ("supplement_text", "expected_criteria", "status", "unreadable", "limitation_count"),
    [
        ("Regional Science Prize\n" + CRITERIA, None, "not_found", 0, 1),
        (AWARD + "\n" + CRITERIA, CRITERIA, "found", 0, 1),
        (AWARD + "\nSelection criteria: originality", "Selection criteria: originality", "found", 0, 1),
        (AWARD + "\nSelection criteria: not disclosed.", None, "not_found", 0, 2),
        (AWARD + "\nSelection criteria:   ", None, "not_found", 0, 1),
        (AWARD + "\nAlex Morgan was selected for work in reliable ML infrastructure.", None, "not_found", 0, 1),
        ("", None, "needs_review", 1, 1),
    ],
)
def test_award_checklist_requires_literal_evidence(
    monkeypatch, supplement_text, expected_criteria, status, unreadable, limitation_count
):
    async def fake_post(self, url, *, json):
        assert "employment_starts" in json["format"]["properties"]
        return httpx.Response(
            200,
            json={"done": True, "response": '{"employment_starts":[]}'},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = client.post(
        "/api/documents/review",
        files=[
            ("files", (AWARD_LETTER, (SAMPLES / AWARD_LETTER).read_bytes(), "application/pdf")),
            ("files", ("supplement.pdf", make_pdf(supplement_text), "application/pdf")),
        ],
        data={"award_selection_criteria_for": f"  {AWARD.lower()}  "},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    [check] = payload["evidence_checks"]
    assert check["kind"] == "award_selection_criteria"
    assert check["award_name"] == AWARD.lower()
    assert check["status"] == status
    assert check["unreadable_page_count"] == unreadable
    assert len(check["limitation_citations"]) == limitation_count
    limitation = check["limitation_citations"][0]
    assert limitation["quote"] == OMISSION
    assert limitation["filename"] == AWARD_LETTER
    if expected_criteria:
        [criteria] = check["criteria_citations"]
        assert criteria["quote"] == expected_criteria
        assert criteria["filename"] == "supplement.pdf"
    else:
        assert check["criteria_citations"] == []

    documents = {document["document_id"]: document for document in payload["documents"]}
    for citation in check["criteria_citations"] + check["limitation_citations"]:
        document = documents[citation["document_id"]]
        assert citation["page_number"] == 1
        assert citation["quote"] in document["pages"][0]["text"]

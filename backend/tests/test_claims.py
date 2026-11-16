import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
AWARD_PDF = Path(__file__).resolve().parents[2] / "sample_documents" / "03_award_letter.pdf"


def upload_for_claims(pdf: bytes):
    return client.post(
        "/api/documents/claims",
        files={"file": (AWARD_PDF.name, pdf, "application/pdf")},
    )


def test_claims_use_exact_quotes_on_the_current_page(monkeypatch):
    quote = "the 2023 National\nTechnology Innovation Award"
    other_page_quote = "Issued in Fictional City"
    with pymupdf.open(AWARD_PDF) as pdf:
        pdf.new_page()
        pdf.new_page().insert_text((72, 100), other_page_quote)
        content = pdf.tobytes()

    prompts = []

    async def fake_post(self, url, *, json):
        assert url == "http://127.0.0.1:11434/api/generate"
        assert json["model"] == "qwen3:4b"
        assert json["format"]["properties"]["claims"]["type"] == "array"
        assert json["stream"] is False
        prompts.append(json["prompt"])
        proposals = (
            [
                {"category": "award", "statement": "Alex Morgan was selected.", "quote": quote},
                {"category": "award", "statement": "Changed quote.", "quote": quote.replace("\n", " ")},
                {"category": "award", "statement": "Wrong page.", "quote": other_page_quote},
            ]
            if len(prompts) == 1
            else [{"category": "achievement", "statement": "Model claim.", "quote": other_page_quote}]
        )
        return generation_response(proposals)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = upload_for_claims(content)
    assert response.status_code == 200, response.text
    result = response.json()
    document = result["document"]
    assert prompts == [document["pages"][0]["text"], document["pages"][2]["text"]]
    matched, changed_quote, wrong_page, third_page = result["claims"]
    assert matched["quote_status"] == "matched"
    assert matched["citation"] == {
        "filename": AWARD_PDF.name,
        "document_id": document["document_id"],
        "page_number": 1,
        "quote": quote,
    }
    for claim in (changed_quote, wrong_page):
        assert claim["source_page"] == 1
        assert claim["quote_status"] == "unmatched"
        assert claim["citation"] is None
    assert third_page["citation"]["page_number"] == 3


def test_claims_reject_long_pages_before_calling_ollama(monkeypatch):
    async def unexpected_post(*args, **kwargs):
        pytest.fail("Ollama should not receive a partial document")

    monkeypatch.setattr(httpx.AsyncClient, "post", unexpected_post)
    with pymupdf.open(AWARD_PDF) as pdf:
        pdf.new_page().insert_text((72, 72), ("Source text " * 7 + "\n") * 55, fontsize=8)
        content = pdf.tobytes()

    response = upload_for_claims(content)
    assert response.status_code == 422
    assert "Page 2 exceeds" in response.json()["detail"]


def ollama_response(status_code, data):
    request = httpx.Request("POST", "http://127.0.0.1:11434/api/generate")
    return httpx.Response(status_code, json=data, request=request)


def generation_response(proposals):
    return ollama_response(200, {"done": True, "response": json.dumps({"claims": proposals})})


@pytest.mark.parametrize(
    ("model_response", "status", "message"),
    [
        (ollama_response(404, {"error": "model not found"}), 503, "ollama pull qwen3:4b"),
        (ollama_response(500, {"error": "server error"}), 502, "Ollama returned HTTP 500"),
        (ollama_response(200, {"unexpected": "response"}), 502, "invalid generation response"),
        (
            ollama_response(200, {"done": True, "done_reason": "length", "response": ""}),
            502,
            "output limit",
        ),
        (ollama_response(200, {"done": True, "response": "{broken"}), 502, "invalid claim fields"),
        (
            generation_response([{"category": "award", "statement": "A claim", "quote": "  "}]),
            502,
            "invalid claim fields",
        ),
    ],
)
def test_claim_model_response_errors(monkeypatch, model_response, status, message):
    async def fake_post(*args, **kwargs):
        return model_response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = upload_for_claims(AWARD_PDF.read_bytes())
    assert response.status_code == status
    assert message in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (httpx.ConnectError("connection refused"), 503, "Could not reach local Ollama"),
        (httpx.ReadTimeout("inference timed out"), 504, "Local Ollama timed out"),
    ],
)
def test_claim_model_transport_errors(monkeypatch, error, status, message):
    async def fake_post(*args, **kwargs):
        raise error

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = upload_for_claims(AWARD_PDF.read_bytes())
    assert response.status_code == status
    assert message in response.json()["detail"]

from io import BytesIO

import pymupdf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_pdf(page_texts: list[str]) -> bytes:
    pdf = pymupdf.open()
    for content in page_texts:
        page = pdf.new_page()
        if content:
            page.insert_text((72, 100), content)
    output = pdf.tobytes()
    pdf.close()
    return output


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_extracts_text_with_page_references():
    content = make_pdf(["Applicant: Alex Morgan", "Employment begins: March 2022"])
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("resume.pdf", BytesIO(content), "application/pdf"))],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["document_count"] == 1
    assert payload["total_pages"] == 2
    doc = payload["documents"][0]
    assert doc["filename"] == "resume.pdf"
    assert doc["pages"][0]["page_number"] == 1
    assert "Alex Morgan" in doc["pages"][0]["text"]
    assert doc["pages"][1]["page_number"] == 2
    assert "March 2022" in doc["pages"][1]["text"]


def test_blank_page_is_flagged_not_hallucinated():
    content = make_pdf([""])
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("blank.pdf", BytesIO(content), "application/pdf"))],
    )
    assert response.status_code == 200
    doc = response.json()["documents"][0]
    assert doc["pages"][0]["text"] == ""
    assert doc["pages"][0]["status"] == "no_text_detected"
    assert doc["warnings"]


def test_rejects_non_pdf_upload():
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("notes.txt", BytesIO(b"hello"), "text/plain"))],
    )
    assert response.status_code == 415


def test_rejects_fake_pdf():
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("fake.pdf", BytesIO(b"not really a pdf"), "application/pdf"))],
    )
    assert response.status_code == 415


def test_uploads_multiple_documents():
    files = [
        ("files", ("one.pdf", BytesIO(make_pdf(["Document one"])), "application/pdf")),
        ("files", ("two.pdf", BytesIO(make_pdf(["Document two"])), "application/pdf")),
    ]
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    assert response.json()["document_count"] == 2

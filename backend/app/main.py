"""PDF ingestion and local evidence claim extraction for synthetic documents."""

from pathlib import Path
from typing import Optional
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .award_checklist import (
    AwardCriteriaCheck,
    award_criteria_passages,
    summarize_award_check,
)
from .claims import (
    MAX_CLAIM_PAGE_BYTES,
    OLLAMA_MODEL,
    CheckedClaim,
    Citation,
    OllamaError,
    check_claim,
    generate_structured,
    ollama_client,
    propose_claims,
)
from .employment import (
    EMPLOYMENT_PROMPT,
    EmploymentDiscrepancy,
    EmploymentStart,
    ProposedEmploymentStarts,
    check_employment_start,
    find_employment_discrepancies,
)

MAX_FILES = 10
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB per file
MAX_PAGES = 50
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="CaseBrief AI",
    description="PDF ingestion and local claim extraction. Synthetic data only.",
    version="0.2.0",
)


class ExtractedPage(BaseModel):
    page_number: int  # Human-readable PDF page number, starting at 1
    text: str
    character_count: int
    status: str  # text_found | no_text_detected


class ExtractedDocument(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages: list[ExtractedPage]
    warnings: list[str]


class UploadResponse(BaseModel):
    documents: list[ExtractedDocument]
    document_count: int
    total_pages: int


class ClaimExtractionResponse(BaseModel):
    model: str
    document: ExtractedDocument
    claims: list[CheckedClaim]


class CaseReviewResponse(BaseModel):
    model: str
    documents: list[ExtractedDocument]
    employment_starts: list[EmploymentStart]
    possible_discrepancies: list[EmploymentDiscrepancy]
    evidence_checks: list[AwardCriteriaCheck]


@app.get("/", include_in_schema=False)
def upload_page():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "step": "document_ingestion"}


def extract_pdf(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """Extract plain text from each page, retaining its original page number."""
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise HTTPException(422, f"{filename}: password-protected PDFs aren't supported")
            if pdf.page_count < 1:
                raise HTTPException(422, f"{filename}: this PDF has no pages")
            if pdf.page_count > MAX_PAGES:
                raise HTTPException(413, f"{filename}: maximum {MAX_PAGES} pages per PDF")

            pages = []
            warnings = []
            for index, page in enumerate(pdf):
                text = page.get_text("text", sort=True).strip()
                status = "text_found" if text else "no_text_detected"
                pages.append(
                    ExtractedPage(
                        page_number=index + 1,
                        text=text,
                        character_count=len(text),
                        status=status,
                    )
                )
                if not text:
                    warnings.append(
                        f"Page {index + 1} has no extractable text. "
                        "It may be scanned; OCR isn't included in Step 1."
                    )
            return ExtractedDocument(
                document_id=str(uuid4()),
                filename=filename,
                page_count=pdf.page_count,
                pages=pages,
                warnings=warnings,
            )
    except HTTPException:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"{filename}: unable to read PDF") from exc


async def read_uploaded_pdf(upload: UploadFile) -> ExtractedDocument:
    # Normalize both POSIX and Windows separators in supplied filenames.
    filename = (upload.filename or "unnamed").replace("\\", "/").split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, f"{filename}: only PDF files are supported")

    # Read at most MAX_FILE_SIZE+1 to avoid unbounded memory allocation.
    try:
        content = await upload.read(MAX_FILE_SIZE + 1)
    finally:
        await upload.close()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"{filename}: maximum size is 10 MiB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, f"{filename}: content is not a valid PDF header")
    return extract_pdf(content, filename)


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, f"Upload between 1 and {MAX_FILES} PDF files")

    extracted = [await read_uploaded_pdf(upload) for upload in files]
    return UploadResponse(
        documents=extracted,
        document_count=len(extracted),
        total_pages=sum(doc.page_count for doc in extracted),
    )


def claim_pages(document: ExtractedDocument) -> list[ExtractedPage]:
    pages = [page for page in document.pages if page.text]
    for page in pages:
        if len(page.text.encode("utf-8")) > MAX_CLAIM_PAGE_BYTES:
            raise HTTPException(
                422,
                f"{document.filename}: Page {page.page_number} exceeds "
                f"{MAX_CLAIM_PAGE_BYTES} bytes of extracted text "
                "for claim extraction.",
            )
    return pages


@app.post("/api/documents/claims", response_model=ClaimExtractionResponse)
async def extract_document_claims(file: UploadFile = File(...)) -> ClaimExtractionResponse:
    document = await read_uploaded_pdf(file)
    pages = claim_pages(document)
    claims = []
    async with ollama_client() as client:
        for page in pages:
            try:
                proposals = await propose_claims(client, page.text)
            except OllamaError as exc:
                raise HTTPException(exc.status_code, f"Page {page.page_number}: {exc}") from exc
            claims.extend(
                check_claim(
                    proposal, document.document_id, document.filename, page.page_number, page.text
                )
                for proposal in proposals
            )
    return ClaimExtractionResponse(model=OLLAMA_MODEL, document=document, claims=claims)


def review_award_criteria(
    sources: list[tuple[ExtractedDocument, list[ExtractedPage]]],
    award_name: str,
) -> AwardCriteriaCheck:
    criteria_citations = []
    limitation_citations = []
    for document, pages in sources:
        for page in pages:
            criteria, limitations = award_criteria_passages(page.text, award_name)
            for passages, citations in (
                (criteria, criteria_citations),
                (limitations, limitation_citations),
            ):
                citations.extend(
                    Citation(
                        document_id=document.document_id,
                        filename=document.filename,
                        page_number=page.page_number,
                        quote=passage,
                    )
                    for passage in passages
                )

    return summarize_award_check(
        award_name,
        criteria_citations,
        limitation_citations,
        sum(document.page_count - len(pages) for document, pages in sources),
    )


@app.post("/api/documents/review", response_model=CaseReviewResponse)
async def review_documents(
    files: list[UploadFile] = File(...),
    award_selection_criteria_for: Optional[str] = Form(None, max_length=200),
) -> CaseReviewResponse:
    if not 2 <= len(files) <= MAX_FILES:
        raise HTTPException(400, f"Upload between 2 and {MAX_FILES} PDF files for review")
    if award_selection_criteria_for is not None:
        award_selection_criteria_for = award_selection_criteria_for.strip()
        if not award_selection_criteria_for:
            raise HTTPException(422, "Provide an award name for the selection criteria check")

    documents = [await read_uploaded_pdf(upload) for upload in files]
    sources = [(document, claim_pages(document)) for document in documents]
    starts = []
    evidence_checks = []
    async with ollama_client() as client:
        for document, pages in sources:
            for page in pages:
                try:
                    result = await generate_structured(
                        client, page.text, EMPLOYMENT_PROMPT, ProposedEmploymentStarts
                    )
                except OllamaError as exc:
                    raise HTTPException(
                        exc.status_code, f"{document.filename}: Page {page.page_number}: {exc}"
                    ) from exc
                for proposal in result.employment_starts:
                    start = check_employment_start(
                        proposal, document.document_id, document.filename, page.page_number, page.text
                    )
                    if start is not None:
                        starts.append(start)
    if award_selection_criteria_for:
        evidence_checks.append(review_award_criteria(sources, award_selection_criteria_for))

    return CaseReviewResponse(
        model=OLLAMA_MODEL,
        documents=documents,
        employment_starts=starts,
        possible_discrepancies=find_employment_discrepancies(starts),
        evidence_checks=evidence_checks,
    )

# CaseBrief AI

A local evidence-preparation demo for a fictional immigration case. Upload PDFs, inspect cited claims, compare conflicting evidence, record review decisions, and save a report as PDF.

## Run the demo

You need Python 3.11+, Ollama, and a browser with PDF printing. Run these commands from the project root on macOS or Linux. [Install Ollama](https://ollama.com/download)

1. Start Ollama. If the desktop app or a local service is already running, use it. With a CLI-only install, leave `ollama serve` running in its own terminal.
2. In another terminal, download the model, set up the virtual environment, and start FastAPI.

   ```bash
   ollama pull qwen3:4b
   python3 -m venv backend/.venv
   backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt -c backend/constraints-demo.txt
   backend/.venv/bin/python -m uvicorn --app-dir backend app.main:app --host 127.0.0.1 --port 8000
   ```

3. Open [CaseBrief AI](http://127.0.0.1:8000). Keep FastAPI and Ollama running while you use **Review evidence**.

If Ollama cannot find the model, check `ollama list` in the same setup. If FastAPI's port is occupied, choose another port with `--port` and use that port in the browser and evaluation command.

## Walk through the fictional case

1. Select these three files together from `sample_documents/`:
   - `01_alex_morgan_resume.pdf`
   - `02_employment_confirmation.pdf`
   - `03_award_letter.pdf`
2. Enter `National Technology Innovation Award` in **Award name**, then click **Review evidence**.
3. Check the two findings. Alex Morgan's résumé lists **March 2022** for Lumina Labs, and the confirmation letter lists **June 2022**. The award letter says it omits the selection criteria. Expand **Read full page text** to inspect a quote in context.
4. Mark the employment finding as reviewed and add a note such as “Confirm the start month with the employer.” Leave the award finding pending.
5. Click **Print / Save as PDF** and choose **Save as PDF** in the browser's print dialog. The report should list three source files, one reviewed finding, one pending finding, the note, and three page 1 citations.

**Extract page text** works without Ollama and accepts as few as one PDF. Review marks and notes stay in the tab until a successful new review, a new file selection, or a reload.

## Verify the build

Run the test suite from the project root:

```bash
(cd backend && .venv/bin/python -m pytest -q)
```

Run the live evaluation with FastAPI and Ollama running:

```bash
backend/.venv/bin/python backend/evaluate.py
```

Expect `3/3 cases passed.` The three cases cover the award and letter date, the employment start-month discrepancy, and the award letter's stated omission. Each reports separate **Findings** and **Citations** results. The evaluator reopens the original PDFs to verify quotations and uses labeled expectations for the synthetic findings. It exits with status 1 if a case fails. Use `--port` if FastAPI is listening on a port other than 8000.

The clean-install baseline was verified on Apple Silicon macOS with Python 3.14.3, Ollama 0.15.6, and `qwen3:4b` model ID `31d9857b5705`. Check the model ID with `ollama list` when comparing evaluation results.

## How it works

| Stage | Implementation |
| --- | --- |
| Read PDFs | FastAPI receives multipart uploads. PyMuPDF extracts text with its original PDF page number and flags empty pages. |
| Propose claims | Ollama runs `qwen3:4b` on loopback. The API sends one page at a time with a Pydantic-derived JSON Schema and validates the response. |
| Check evidence | The claim endpoint checks model quotes verbatim. The review endpoint checks proposed employment values against the page, slices original text for the citation, and compares normalized start months in Python. |
| Check criteria | An optional award checklist scans pages naming the award for explicit criteria or omission wording. It uses Python rules and quotes the matched source text. |
| Review and export | A static browser page renders findings and review decisions. Its print layout produces the PDF using the browser. |

A matched quote establishes that a passage occurs on the cited page. A reviewer decides whether the passage supports the statement and how much weight it deserves. Employment flags compare explicit start months for the same name and employer after normalization. The award check covers supported wording on pages naming the award and returns `needs_review` when a page has no extractable text and no criteria are found.

## API examples

[Interactive API docs](http://127.0.0.1:8000/docs) · [Health endpoint](http://127.0.0.1:8000/health)

- `POST /api/documents/upload` accepts the repeatable multipart field `files` and returns page-level text.
- `POST /api/documents/claims` accepts one PDF in the multipart field `file`. The award letter should yield a selection claim and an October 9, 2023 letter date. An unmatched quote has a null citation.

  ```bash
  curl -sS -F 'file=@sample_documents/03_award_letter.pdf' \
    http://127.0.0.1:8000/api/documents/claims | python3 -m json.tool
  ```

- `POST /api/documents/review` accepts 2–10 PDFs in `files`. Supply `award_selection_criteria_for` to request the optional award check.

  ```bash
  curl -sS \
    -F 'files=@sample_documents/01_alex_morgan_resume.pdf' \
    -F 'files=@sample_documents/02_employment_confirmation.pdf' \
    -F 'files=@sample_documents/03_award_letter.pdf' \
    --form-string 'award_selection_criteria_for=National Technology Innovation Award' \
    http://127.0.0.1:8000/api/documents/review | python3 -m json.tool
  ```

## Demo boundaries

- Use the bundled fictional samples. This prototype is for evidence preparation and attorney review.
- Run on loopback. The app has no authentication or production-grade document isolation. Ollama receives page text locally and the app does not persist uploads. The web framework may use temporary files.
- Uploads are limited to 10 PDFs, 10 MiB per file, and 50 pages per file. Model endpoints limit extracted text to 4,096 UTF-8 bytes per page. Scanned pages require OCR, which is outside the demo.

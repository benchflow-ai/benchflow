---
name: office-files
description: Read, edit and write spreadsheets, Word documents, slide decks and PDFs from Python, and check the result before finishing.
---

# Office files

Use Python for every office file; do not hand-edit binary formats.

- Spreadsheets (.xlsx): `openpyxl` to keep formatting and formulas, `pandas` for analysis. Write values, not formulas, unless the task asks for formulas.
- Word (.docx): `python-docx`. Edit runs and paragraphs in place so styles survive.
- Slides (.pptx): `python-pptx`.
- PDF: `pypdf` or `pdfplumber` to read; to change a PDF, write a new file rather than editing bytes.

Before you finish, open every output file again with the same library and check it holds what the task asked for, at the exact path and name the task gives.

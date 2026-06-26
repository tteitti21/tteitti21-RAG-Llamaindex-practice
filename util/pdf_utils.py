import os

from llama_index.core import Document
from pypdf import PdfReader


def load_pdf_documents(pdf_path):
    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    documents = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_path": pdf_path,
                    "file_name": file_name,
                    "page": page_index,
                    "page_label": str(page_index),
                },
            )
        )

    return documents

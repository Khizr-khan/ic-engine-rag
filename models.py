# ============================================================
# models.py — Think of this as FORMS used in a library system
# Just like a library has different forms (borrow form, 
# return form, receipt) — this file defines the exact format
# of every request and response in our API
# ============================================================

from pydantic import BaseModel
# Pydantic is like a FORM VALIDATOR at a government office
# It checks that every field is filled correctly before
# processing — wrong data type = rejected immediately

from typing import List, Optional
# List = "multiple items" — like a shopping list
# Optional = "this field can be left blank" — like a middle
# name field on a form, not everyone has one

# ============================================================
# FORM 1 — What a student fills in to ASK a question
# Like a "Question Slip" you hand to a librarian
# ============================================================
class AskRequest(BaseModel):

    question: str
    # The actual question written on the slip
    # str = must be text, cannot be a number or blank
    # Example: "What is the compression ratio in diesel engine?"

    top_k: Optional[int] = 4
    # How many reference books the librarian should check
    # Optional = student doesn't have to fill this in
    # Default is 4 — librarian checks 4 sources by default
    # Student can increase it: top_k=8 means check 8 sources

# ============================================================
# FORM 2 — One single source/reference the librarian found
# Like one entry in a bibliography
# "I found this answer in THIS book, on THIS page"
# ============================================================
class SourceDoc(BaseModel):

    filename: str
    # Name of the book/PDF where the answer was found
    # Example: "112103262.pdf" or "heywood_ic_engines.pdf"

    page: int
    # The exact page number in that book
    # int = must be a whole number like 43, not 43.5
    # Example: 43

    excerpt: str
    # A small quote from that page — the exact lines used
    # Like highlighting the relevant paragraph in a textbook
    # Limited to 250 characters to keep responses clean

# ============================================================
# FORM 3 — The ANSWER SHEET the librarian hands back
# Contains both the answer AND where it came from
# Like a research report with footnotes/citations
# ============================================================
class AskResponse(BaseModel):

    answer: str
    # The full answer written by the AI
    # Like the librarian summarising what they found
    # Example: "Compression ratio is 1 + Vc/Vs where..."

    sources: List[SourceDoc]
    # A LIST of SourceDoc objects — the bibliography
    # List means multiple items — one per source used
    # Example: [page 43 of file A, page 45 of file A]
    # Student can go back and verify every claim made

# ============================================================
# FORM 4 — The RECEIPT after uploading new documents
# Like a deposit slip at a bank — confirms what went in
# "We received your documents, here is what we processed"
# ============================================================
class IngestResponse(BaseModel):

    message: str
    # A human-readable confirmation message
    # Like the "Transaction Successful" on an ATM receipt
    # Example: "Documents ingested successfully"

    chunks_added: int
    # How many pieces the PDF was cut into and stored
    # Like counting how many index cards were filed
    # A 100-page book might produce 300-400 chunks
    # If this is 0 — something went wrong, nothing was stored

    files_processed: int
    # How many PDF files were successfully read
    # If you upload 3 PDFs but get files_processed=1
    # — two of them failed silently, go check them
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    history: Optional[List[ChatMessage]] = []

class SourceDoc(BaseModel):
    filename: str
    page: int
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    files_processed: int
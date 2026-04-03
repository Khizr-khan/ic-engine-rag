import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import AskRequest, AskResponse, SourceDoc, IngestResponse
from rag_engine import rag
from ingest import ingest_documents

load_dotenv()

app = FastAPI(
    title="IC Engine RAG API",
    description="AI study assistant for IC engine engineering students",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "IC Engine RAG API is running",
        "docs": "Visit /docs for interactive API documentation",
        "health": "Visit /health to check server status"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_indexed": rag.get_doc_count(),
        "message": "IC Engine RAG API is running"
    }

@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(body.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question is too short — please ask a complete question")
    if len(body.question.strip()) > 500:
        raise HTTPException(status_code=400, detail="Question is too long — maximum 500 characters allowed")
    try:
        history = [{"role": m.role, "content": m.content}
                   for m in (body.history or [])]
        result = rag.ask(body.question, top_k=body.top_k, history=history)
        sources = [SourceDoc(**s) for s in result["sources"]]
        return AskResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    

@app.post("/generate-quiz")
def generate_quiz(body: dict):
    topic = body.get("topic", "")
    num_questions = body.get("num_questions", 5)
    
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    try:
        questions = rag.generate_quiz(topic, num_questions)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")
    
@app.get("/token-stats")
def get_token_stats():
    return rag.get_token_stats()

@app.post("/switch-model")
def switch_model(body: dict):
    model = body.get("model", "")
    if model not in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    rag.switch_model(model)
    return {"message": f"Switched to {model}", "model": model}



@app.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    admin_key: Optional[str] = Header(None)
):
    expected_key = os.getenv("ADMIN_KEY", "")
    if not expected_key:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not set in .env file")
    if admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized — invalid or missing admin key")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
    tmp_dir = tempfile.mkdtemp()
    try:
        for file in files:
            dest = os.path.join(tmp_dir, file.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
        chunks, file_count = ingest_documents(tmp_dir)
        return IngestResponse(
            message="Documents ingested successfully",
            chunks_added=chunks,
            files_processed=file_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir)

@app.get("/docs-list")
def list_docs():
    try:
        collection = rag.vectorstore._collection
        results = collection.get(include=["metadatas"])
        sources = list({
            os.path.basename(m.get("source", "unknown"))
            for m in results["metadatas"]
        })
        return {
            "indexed_files": sources,
            "total_files": len(sources),
            "total_chunks": rag.get_doc_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve document list: {str(e)}")
    
from fastapi.responses import StreamingResponse

@app.post("/ask-stream")
def ask_stream(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(body.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question is too short")
    if len(body.question.strip()) > 500:
        raise HTTPException(status_code=400, detail="Question is too long")

    history = [{"role": m.role, "content": m.content}
               for m in (body.history or [])]

    def generate():
        for chunk in rag.ask_stream(body.question, top_k=10, history=history):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
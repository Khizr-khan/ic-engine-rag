# IC Engine RAG Study Assistant — Technical Documentation

**Version:** 2.0  
**Author:** Khizr Khan  
**Backend:** https://khizr72-ic-engine-rag-v2.hf.space  
**Repository:** https://github.com/Khizr-khan/ic-engine-rag  
**Dataset:** https://huggingface.co/datasets/Khizr72/ic-engine-chromadb  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Reference](#2-file-reference)
3. [RAG Engine](#3-rag-engine)
4. [Prompt System](#4-prompt-system)
5. [Model Routing](#5-model-routing)
6. [Smart RAG Routing](#6-smart-rag-routing)
7. [API Reference](#7-api-reference)
8. [Frontend](#8-frontend)
9. [Database](#9-database)
10. [Deployment](#10-deployment)
11. [Known Issues](#11-known-issues)
12. [Verified Reference Answers](#12-verified-reference-answers)
13. [Push Commands](#13-push-commands)

---

## 1. Architecture Overview

```
Student (Browser)
      ↓
Streamlit Frontend (app.py)   ← Streamlit Cloud
      ↓  HTTP streaming
FastAPI Backend (main.py)     ← HuggingFace Spaces (Docker)
      ↓
RAG Engine (rag_engine.py)
      ↓                    ↓
ChromaDB Vector DB      Groq LLM API
(HuggingFace Dataset)   (llama-3.3-70b / llama-4-scout / llama-3.1-8b)
      ↑
Ganesan PDF (ingested via ingest.py)
```

### Key Design Decisions

- **Streaming responses** — FastAPI streams chunks, Streamlit renders them in real time
- **RAG with MMR** — Maximum Marginal Relevance retrieval avoids redundant chunks
- **Smart routing** — numerical questions skip RAG for performance questions, keep RAG for cycle theory
- **Model fallback chain** — automatic switching on rate limits, no user intervention needed
- **HuggingFace dataset** — ChromaDB stored as dataset, downloaded on Space startup

---

## 2. File Reference

### `app.py` — Streamlit Frontend
- Dark theme UI with IBM Plex fonts
- Streaming chat interface with thinking indicator (pulsing dots)
- Token usage progress bar (color-coded: green → yellow → red)
- Model switcher dropdown (3 models)
- Quiz mode with answer checking
- Dynamic auto-grow textarea input
- `format_subscripts()` — strips LaTeX, converts to HTML subscripts
- `is_quiz_request()` — detects 11 quiz trigger patterns
- `st.form` with `clear_on_submit=True` for input clearing

### `main.py` — FastAPI Backend
- Runs on port 7860 (HuggingFace Spaces)
- CORS enabled for all origins
- Imports `rag` singleton from `rag_engine.py`
- All endpoints documented at `/docs`

### `rag_engine.py` — Core Logic
- `RAGEngine` class with all methods
- `PROMPT_TEMPLATE` — 9 response types with formatting rules
- `NUMERICAL_SYSTEM_PROMPT` — dedicated prompt for calculations
- `MODELS` dict — model names, limits, labels
- `token_stats` dict — usage tracking
- ChromaDB downloaded from HuggingFace on startup

### `ingest.py` — Document Processing
- Loads PDFs with `PyPDFDirectoryLoader`
- Filters noise pages (blank, TOC, index)
- Splits with `RecursiveCharacterTextSplitter`
- Creates embeddings with `all-MiniLM-L6-v2`
- Saves to `./chroma_ic_db`

### `models.py` — Pydantic Models
```python
AskRequest:     question, top_k, history
AskResponse:    answer, sources
ChatMessage:    role, content
SourceDoc:      filename, page, excerpt
IngestResponse: message, chunks_added, files_processed
```

---

## 3. RAG Engine

### Initialization
```python
rag = RAGEngine()
# Downloads ChromaDB from HuggingFace on first run
# Loads all-MiniLM-L6-v2 embeddings
# Sets default model to llama-3.3-70b-versatile
```

### Key Methods

**`enhance_question(question, history)`**
Improves vague questions using conversation history:
- "formula" → "formula for [last topic]"
- bare topics → "Explain [topic] in detail"
- short keywords preserved as-is

**`is_numerical_question(question)`**
Returns True if question has 3+ numerical keywords OR numbers with units:
```python
numerical_keywords = ["calculate", "find", "bore", "stroke", "rpm",
                      "kpa", "kw", "efficiency", "pressure", "temperature",
                      "power", "volume", "cylinder", "piston", "ratio"]
```

**`needs_rag(question)`**
Smart routing — determines whether to retrieve from ChromaDB:
- Engine performance (bp, ip, fp, bmep, swept volume) → skip RAG
- Thermodynamic cycles (otto, diesel, brayton, T1/T2/T3/T4) → use RAG
- Default → use RAG

**`ask_stream(question, top_k, history)`**
Main streaming method:
1. Enhance question
2. If numerical → retrieve 6 RAG chunks → try compound-mini (Python executor on Groq) → local Python subprocess executor → Scout fallback
3. If conceptual → check needs_rag → retrieve if needed → send to 70b
4. Auto-switches model on 429 / quota rate limit errors

**`generate_quiz(topic, num_questions)`**
- Retrieves 20 chunks for topic
- Uses `@@@` separator between questions
- Temperature 0.4 for variety

---

## 4. Prompt System

### Response Types

| Type | Trigger | Response Style |
|------|---------|----------------|
| TYPE 1 | "only", "just", "briefly" | 1-3 lines max |
| TYPE 2 | "what is", "define" | Definition + formula + typical values |
| TYPE 3 | "why", "how does", "effect of" | Chain of thought reasoning |
| TYPE 4 | "difference", "compare", "vs" | Structured comparison |
| TYPE 5 | "explain", "describe", "in detail" | Full comprehensive answer |
| TYPE 6 | "draw", "diagram", "sketch" | ASCII art diagram |
| TYPE 7 | Numbers + "calculate"/"find" | Step-by-step calculation |
| TYPE 8 | "explain more", "elaborate" | Expand on previous topic |
| TYPE 9 | "regenerate", "try again" | Completely different approach |

### Critical Reminders in Prompt
```
• /60 mandatory in power formulas
• Keep imep/bmep in kPa — do NOT convert to Pa
• Diesel Q_in uses Cp = γ × Cv (NOT Cv)
• Diesel T_4 = T_3 × (rc/r)^(γ-1)
• Brayton exponent = (γ-1)/γ = 0.2857 (NOT 0.4)
• 16^0.4 = 3.031 and 18^0.4 = 3.178 are DIFFERENT
• Verification step after every numerical solution
• NEVER write TYPE labels in response
```

### Exponent Tables in Prompt
```
Otto/Diesel (γ-1 = 0.4):    8^0.4=2.297, 9^0.4=2.408, 16^0.4=3.031, 18^0.4=3.178
Diesel cutoff (rc^γ):        2.5^1.4=3.607, 2.2^1.4=3.016
Brayton ((γ-1)/γ = 0.2857): 8^0.2857=1.811, 10^0.2857=1.931
```

---

## 5. Model Routing

### Fallback Chain

```
Question received
      ↓
is_numerical_question()?
      │
      ├── YES → Retrieve RAG context (6 chunks)
      │              ↓
      │         TIER 1: compound-mini (Python on Groq servers)
      │              ├── Success → clean LaTeX → stream answer
      │              └── Fails → TIER 2: Local Python executor
      │                              ├── Scout writes Python code
      │                              ├── HF Space runs subprocess
      │                              ├── Success → Scout formats → stream answer
      │                              └── Fails → TIER 3: Scout pattern matching
      │                                              └── 429 → 8B fallback
      │
      └── NO  → needs_rag()?
                    ├── YES → retrieve from ChromaDB
                    └── NO  → use prompt formulas only
                         ↓
                    send to llama-3.3-70b
                         ├── 429/quota → switch to Scout
                         │                   └── 429 → switch to 8B
                         └── success → stream response
```

### Model Configuration
```python
MODELS = {
    "llama-3.3-70b-versatile":                   {"limit": 100000, "label": "70B — High Quality", "provider": "groq"},
    "llama-3.1-8b-instant":                      {"limit": 500000, "label": "8B — Fast",          "provider": "groq"},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"limit": 100000, "label": "Llama 4 Scout",      "provider": "groq"},
}
```

### Numerical Pipeline — 3 Tiers
```
Tier 1: compound-mini    → Python executes on Groq servers  → exact
Tier 2: Local executor   → Scout writes code, HF Space runs → exact  
Tier 3: Scout fallback   → pattern matching                 → ~80% accurate
```

### Token Reset
- 70B and Scout: 100,000 tokens/day → resets midnight UTC (5am PKT)
- 8B: 500,000 tokens/day
- compound-mini: shares 70B pool

### Rate Limit Detection
All of these are caught and trigger model switching:
```python
["429", "rate_limit", "quota", "resource_exhausted", "resourceexhausted"]
```

---

## 6. Smart RAG Routing

### `needs_rag()` Logic

**Skip RAG — engine performance questions:**
```
brake power, indicated power, friction power,
bp, ip, fp, bmep, imep, fmep,
swept volume, displacement, volumetric efficiency,
bsfc, mechanical efficiency
```
These skip retrieval because formulas are already in the prompt and RAG adds noise.

**Keep RAG — thermodynamic cycle questions:**
```
otto, diesel, brayton, carnot, cycle,
thermal efficiency, heat addition, compression ratio,
cutoff ratio, pressure ratio, t1, t2, t3, t4
```
These use retrieval because Ganesan textbook provides worked examples and theory.

**Default:** use RAG for all other conceptual questions.

---

## 7. API Reference

### POST `/ask-stream`
```json
Request:
{
  "question": "What is compression ratio?",
  "top_k": 10,
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}

Response: text/plain streaming
```

### POST `/generate-quiz`
```json
Request:
{
  "topic": "Otto cycle",
  "num_questions": 5
}

Response:
{
  "questions": [
    {
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A",
      "explanation": "..."
    }
  ]
}
```

### POST `/switch-model`
```json
Request:  {"model": "llama-3.1-8b-instant"}
Response: {"message": "Switched to llama-3.1-8b-instant", "model": "..."}
```

### GET `/token-stats`
```json
{
  "used": 45231,
  "limit": 100000,
  "remaining": 54769,
  "model": "llama-3.3-70b-versatile",
  "label": "70B — High Quality",
  "percent_used": 45.2
}
```

### GET `/health`
```json
{
  "status": "ok",
  "docs_indexed": 4821,
  "message": "IC Engine RAG API is running"
}
```

---

## 8. Frontend

### Key Components

**Thinking Indicator**
Shown immediately when request is sent, before first chunk arrives:
- Pulsing green dots animation matching app theme
- Replaced by actual text as soon as first chunk arrives

**Token Tracker**
- Shows used/remaining tokens
- Color coded: green (<60%) → yellow (<85%) → red (>85%)
- Fetches from `/token-stats` on each page load

**Model Switcher**
- Dropdown with 3 models
- Calls `/switch-model` on change
- Shows current model in header

**Chat Display**
- User messages: right-aligned green bubble
- AI messages: left-aligned dark bubble with streaming
- `format_subscripts()` applied to all AI responses

**Dynamic Textarea**
- Starts at 52px (2 lines)
- Auto-grows as user types via JavaScript
- Max height 300px, scrolls after that
- `resize: vertical` allows manual drag

**`format_subscripts()` Processing**
```
1. Remove **bold** and *italic* markdown
2. Strip LaTeX delimiters ($, $$, \[, \])
3. Handle \frac{}{} → (a/b)
4. Convert \times → ×, \eta → η, \gamma → γ
5. Remove remaining LaTeX commands
6. Convert V_x → V<sub>x</sub>, T_x, P_x, η_x, W_x, Q_x
```

**Quiz Mode**
Triggered by patterns like "ask me N questions", "quiz me on", "test me":
- Parses `@@@` separated questions from backend
- Shows one question at a time
- Checks answers and shows explanation

---

## 9. Database

### ChromaDB Settings
```python
chunk_size    = 1500   # characters per chunk
chunk_overlap = 200    # overlap between chunks
embeddings    = "all-MiniLM-L6-v2"
collection    = "langchain"
```

### Retrieval Settings
```python
search_type = "mmr"    # Maximum Marginal Relevance
k           = 10       # chunks retrieved per query
fetch_k     = 20       # candidates before MMR filtering
```

### Noise Filter
Pages are skipped if:
- Length < 100 characters
- Contains INDEX_KEYWORDS (INDEX, S.NO, PAGE.NO, etc.)
- More than 95% of lines are shorter than 60 characters

### Re-ingestion Steps
```bash
# 1. Backup old database
rename chroma_ic_db chroma_ic_db_backup

# 2. Place PDFs in ic_engine_docs/
# 3. Run ingest
python ingest.py

# 4. Verify
python view_db.py
python test.py

# 5. Upload to HuggingFace
python upload_db.py
```

---

## 10. Deployment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM calls |
| `HF_TOKEN` | Yes | HuggingFace token for dataset download |
| `ADMIN_KEY` | Yes | Secret key for `/ingest` endpoint |

Set these in:
- **Local:** `.env` file in project root
- **HuggingFace Space:** Settings → Variables and secrets
- **Streamlit Cloud:** Settings → Secrets

### HuggingFace Space Settings
```
SDK:      Docker
Hardware: CPU Basic (free)
Port:     7860
```

### Dockerfile Notes
- Python 3.10 base image
- uvicorn runs on port 7860
- ChromaDB downloaded from HuggingFace on first startup (~30 seconds)

---

## 11. Known Issues

| Issue | Root Cause | Status |
|-------|-----------|--------|
| compound-mini intermittently returns empty | Non-deterministic code executor | Handled — falls to local Python executor |
| Arithmetic errors on unfamiliar exponents (e.g. 6^0.4) | LLM limitation in Tier 3 Scout fallback only | Mitigated — Python executor handles Tiers 1 and 2 exactly |
| 8B model leaks TYPE labels in response | Weak instruction following | Partially fixed via prompt |
| Diesel T4 slight inconsistency across attempts | Rounding differences | Acceptable for study assistant |
| Google/Gemma 4 model disabled | Model name not confirmed | Disabled — re-enable when correct model string confirmed |

---

## 12. Verified Reference Answers

### Otto Cycle (CR=8, T1=300K, Qin=1800, γ=1.4, Cv=0.718, R=0.287, P1=100kPa)
```
T2  = 300 × 8^0.4 = 300 × 2.297 = 689.1 K
T3  = 689.1 + 1800/0.718 = 3196 K
T4  = 3196 / 2.297 = 1391 K
η_th = 1 - 1/2.297 = 56.5%
Wnet = 0.565 × 1800 = 1017 kJ/kg
MEP  = 1017 / 0.7534 = 1350 kPa
```

### Diesel Cycle (CR=16, rc=2.5, T1=300K, γ=1.4, Cv=0.718, R=0.287, P1=100kPa)
```
T2  = 300 × 16^0.4 = 300 × 3.031 = 909.3 K
T3  = 909.3 × 2.5 = 2273 K
T4  = 2273 × (2.5/16)^0.4 = 1082 K
η_th = 59.1%
Qin  = 1.005 × (2273-909) = 1371 kJ/kg   [Cp = γ×Cv]
Wnet = 810 kJ/kg
MEP  = 1003 kPa
```

### Brayton Cycle (rp=8, T1=300K, T3=1200K, γ=1.4, Cp=1.005)
```
8^0.2857 = 1.8114
T2  = 300 × 1.8114 = 543.4 K
T4  = 1200 / 1.8114 = 662.5 K
Wc  = 1.005 × (543.4-300) = 244.7 kJ/kg
Wt  = 1.005 × (1200-662.5) = 540.2 kJ/kg
Wnet = 540.2 - 244.7 = 295.6 kJ/kg
η_th = 1 - 1/1.8114 = 44.8%
```

### Engine Performance (bore=90mm, stroke=110mm, N=2500rpm, imep=900kPa, K=6, η_mech=85%)
```
A   = π/4 × 0.09² = 0.006362 m²
ip  = 900 × 0.11 × 0.006362 × 1250 × 6 / 60 = 78.73 kW
bp  = 0.85 × 78.73 = 66.92 kW
fp  = 78.73 - 66.92 = 11.81 kW
```

---

## 13. Push Commands

### Frontend only (Streamlit auto-redeploys)
```bash
git add app.py
git commit -m "msg"
git push origin main
```

### Backend changes (HF Space restarts)
```bash
git add rag_engine.py main.py
git commit -m "msg"
git push origin main
git push space main --force
```

### Both
```bash
git add .
git commit -m "msg"
git push origin main
git push space main --force
```
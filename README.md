---
title: Ic Engine Rag
emoji: ⚙️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# ⚙️ IC Engine RAG Study Assistant

An AI-powered study assistant for IC Engine engineering students, built with Retrieval-Augmented Generation (RAG). Answers conceptual and numerical questions directly from the **Ganesan IC Engines textbook**.

---

## 🔗 Live Demo

| Service | URL |
|---------|-----|
| Frontend (Streamlit) | https://ic-engine-rag-9gay6meunapdazol96crzg.streamlit.app |
| Backend (FastAPI) | https://khizr72-ic-engine-rag-v2.hf.space |
| API Docs | https://khizr72-ic-engine-rag-v2.hf.space/docs |
| Dataset | https://huggingface.co/datasets/Khizr72/ic-engine-chromadb |

---

## ✨ Features

- **RAG-powered answers** grounded in Ganesan IC Engines textbook
- **Numerical problem solving** — Otto, Diesel, Brayton cycles and engine performance
- **9 response types** — short answers, explanations, comparisons, diagrams, calculations
- **Auto quiz generation** from course material with answer checking
- **Streaming responses** with real-time typing effect and thinking indicator
- **Smart model routing** — compound-mini → Llama 4 Scout → 8B fallback for numericals
- **Auto model switching** when daily token limits are hit
- **Token usage tracker** with color-coded progress bar
- **3 model options** — 70B, 8B, Llama 4 Scout
- **Smart RAG routing** — skips retrieval for performance questions, uses it for cycle theory

---

## 🏗️ Architecture

```
Student (Browser)
      ↓
Streamlit Frontend (app.py)
      ↓  HTTP streaming
FastAPI Backend (main.py)
      ↓
RAG Engine (rag_engine.py)
      ↓                    ↓
ChromaDB Vector DB      Groq / Google LLM API
(HuggingFace Dataset)   (70B / Scout / 8B)
      ↑
Ganesan PDF (ingested via ingest.py)
```

---

## 🗂️ Project Structure

```
ic_engine_rag_deploy/
├── app.py              # Streamlit frontend UI
├── main.py             # FastAPI backend endpoints
├── rag_engine.py       # RAG pipeline, prompts, model routing
├── ingest.py           # PDF processing and ChromaDB creation
├── models.py           # Pydantic request/response models
├── upload_db.py        # Upload ChromaDB to HuggingFace
├── view_db.py          # Inspect local ChromaDB chunks
├── test.py             # Test retrieval quality
├── requirements.txt    # Python dependencies
├── Dockerfile          # HuggingFace Spaces deployment
└── ic_engine_docs/     # Place PDF textbooks here for ingestion
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Khizr-khan/ic-engine-rag
cd ic-engine-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
ADMIN_KEY=your_admin_key
GOOGLE_API_KEY=your_google_ai_studio_key
```

### 5. Add textbook PDF
```bash
mkdir ic_engine_docs
# Copy Ganesan IC Engines PDF into ic_engine_docs/
```

### 6. Ingest documents
```bash
python ingest.py
```

### 7. Upload database to HuggingFace
```bash
python upload_db.py
```

### 8. Run backend locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 9. Run frontend locally
```bash
streamlit run app.py
```

---

## 🤖 Models

| Model | Provider | Token Limit | Use Case |
|-------|----------|------------|----------|
| llama-3.3-70b-versatile | Groq | 100k/day | Primary — conceptual questions |
| meta-llama/llama-4-scout-17b-16e-instruct | Groq | 100k/day | Numerical problems |
| llama-3.1-8b-instant | Groq | 500k/day | Fallback when 70B exhausted |
| groq/compound-mini | Groq | Shared pool | Numerical with code execution |

Token limits reset at **midnight UTC (5am Pakistan time)**.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check + doc count |
| POST | `/ask` | Non-streaming question |
| POST | `/ask-stream` | Streaming question (used by UI) |
| POST | `/generate-quiz` | Generate MCQ quiz |
| GET | `/token-stats` | Current token usage |
| POST | `/switch-model` | Switch active model |
| POST | `/ingest` | Upload new PDFs (admin only) |
| GET | `/docs-list` | List indexed documents |

---

## 🗃️ Database

| Setting | Value |
|---------|-------|
| Type | ChromaDB vector database |
| Embeddings | all-MiniLM-L6-v2 |
| Chunk size | 1500 characters |
| Chunk overlap | 200 characters |
| Search type | MMR (Maximum Marginal Relevance) |
| Chunks retrieved | 10 per query |
| Hosted on | HuggingFace Datasets |

---

## 🔄 Deployment

### Backend (HuggingFace Spaces)
```bash
git add rag_engine.py main.py
git commit -m "your message"
git push origin main
git push space main --force
```

### Frontend (Streamlit Cloud)
```bash
git add app.py
git commit -m "your message"
git push origin main
```

### Both
```bash
git add .
git commit -m "your message"
git push origin main
git push space main --force
```

---

## 📊 Verified Reference Answers

### Otto Cycle (CR=8, T1=300K, Qin=1800, γ=1.4, Cv=0.718, P1=100kPa)
```
T2 = 689K  |  T3 = 3196K  |  T4 = 1391K
η  = 56.5% |  Wnet = 1017 kJ/kg  |  MEP = 1350 kPa
```

### Diesel Cycle (CR=16, rc=2.5, T1=300K, P1=100kPa)
```
T2 = 909K  |  T3 = 2273K  |  T4 = 1082K
η  = 59.1% |  Qin = 1371 kJ/kg  |  Wnet = 810 kJ/kg
```

### Brayton Cycle (rp=8, T1=300K, T3=1200K, γ=1.4, Cp=1.005)
```
T2 = 543.4K  |  T4 = 662.5K
Wc = 244.7   |  Wt = 540.2  |  Wnet = 295.6 kJ/kg  |  η = 44.8%
```

### Engine Performance (bore=90mm, stroke=110mm, N=2500rpm, imep=900kPa, K=6, η_mech=85%)
```
ip = 78.73 kW  |  bp = 66.92 kW  |  fp = 11.81 kW
```

---

## ⚠️ Disclaimer

AI can make mistakes — always verify numerical answers independently. Token limits reset daily at midnight UTC.
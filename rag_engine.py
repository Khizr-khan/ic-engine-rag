import os
from huggingface_hub import snapshot_download
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_ic_db"
HF_REPO_ID = "khizr72/ic-engine-chromadb"

def download_database():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("Database not found locally — downloading from Hugging Face...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=CHROMA_DIR,
            token=os.getenv("HF_TOKEN")
        )
        print("Database downloaded successfully!")
    else:
        print("Database found locally — skipping download")

download_database()

PROMPT_TEMPLATE = """You are an IC Engine professor. Answer the question using ONLY the context below.

STRICT RULES:
- If question contains "only", "just", "formula only", "definition only" → respond in 1-3 lines MAXIMUM. Nothing else.
- If question asks to "explain", "describe", "elaborate", "detail" → give full explanation
- If question asks for "diagram" → draw ASCII art
- NEVER add definitions, examples, or extra info when student asks for something specific
- NEVER mention slide times, page numbers, or document names
- If not in context → say: 'This topic is not covered in the course material.'
- If not about IC engines → say: 'Please ask questions related to IC engines only.'
- NEVER make up information

Context:
{context}

Question:
{question}

Answer (follow the STRICT RULES above):"""



class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    def enhance_question(self, question: str) -> str:
        question = question.strip()

        # Short answer keywords — don't expand these
        short_keywords = [
            "only", "just", "formula", "definition only",
            "give me formula", "only formula", "briefly"
        ]
        if any(kw in question.lower() for kw in short_keywords):
            return question  # return as-is, don't add "Explain...in detail"

        question_words = [
            "what", "how", "why", "when", "where",
            "which", "explain", "describe", "define",
            "draw", "show", "sketch", "diagram"
        ]
        is_proper_question = any(
            question.lower().startswith(w) for w in question_words
        ) or question.endswith("?")

        if not is_proper_question:
            return f"Explain {question} in detail"
        return question

    def ask(self, question: str, top_k: int = 6) -> dict:
        question = self.enhance_question(question)

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "No documents found. Please make sure ingest.py has been run.",
                "sources": []
            }

        context = "\n\n".join(d.page_content for d in docs)
        prompt_value = self.prompt.format(
            context=context,
            question=question
        )
        answer = self.llm.invoke(prompt_value).content

        seen = set()
        sources = []
        for doc in docs:
            filename = os.path.basename(
                doc.metadata.get("source", "unknown")
            )
            page = doc.metadata.get("page", 0)
            key = f"{filename}_{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": filename,
                    "page": page,
                    "excerpt": doc.page_content[:250]
                })

        return {
            "answer": answer,
            "sources": sources
        }

    def get_doc_count(self) -> int:
        return self.vectorstore._collection.count()


rag = RAGEngine()
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

PPROMPT_TEMPLATE = """You are an expert IC Engine professor teaching engineering students.
Use the context below from course documents to give a DETAILED, thorough explanation.

Your answer must include:
1. A clear definition of the concept
2. The underlying principle or theory behind it
3. The mathematical formula or expression if applicable (explain each term)
4. A practical real-world example or application
5. Why it matters in engine design or performance
6. Any related concepts the student should know

If the student asks for a diagram, flowchart, cycle, or process,
draw it using ASCII art using box drawing characters like:
┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ → ↓ ↑ ←

IMPORTANT: Do NOT mention slide times, refer slide time, page numbers,
document names, or any source references in your answer.
Write as a professor explaining directly to a student.
Use simple language an engineering student can understand.
Never make up information. Never answer from general knowledge.
If the answer is not in the context, say: 'This topic is not covered in the course material.'
If the question is not related to IC engines, say: 'Please ask questions related to IC engines only.'

Context:
{context}

Question:
{question}

Detailed Answer:"""


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
            model="llama-3.1-8b-instant",
            temperature=0
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def enhance_question(self, question: str) -> str:
        question = question.strip()
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
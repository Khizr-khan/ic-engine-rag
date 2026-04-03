import os
from huggingface_hub import snapshot_download
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_ic_db"
HF_REPO_ID = "Khizr72/ic-engine-chromadb"

# Token tracking
token_stats = {
    "used": 0,
    "limit": 100000,
    "model": "llama-3.3-70b-versatile"
}

MODELS = {
    "llama-3.3-70b-versatile": {"limit": 100000, "label": "70B (High Quality)"},
    "llama-3.1-8b-instant":    {"limit": 500000, "label": "8B (Fast)"},
}

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
- If question asks "why", "how does", "what happens", "compare", "difference" → think step by step internally but write ONLY the final answer. Do NOT show Step 1, Step 2 etc in your response.

- If question asks to "explain", "describe", "elaborate", "detail" → give full explanation
- If question asks for "diagram" → draw ASCII art
- NEVER add definitions, examples, or extra info when student asks for something specific
- NEVER mention slide times, page numbers, or document names
- If not in context → say: 'This topic is not covered in the course material.'
- If not about IC engines → say: 'Please ask questions related to IC engines only.'
- NEVER make up information
- If context contains garbled symbols like ��, ignore them and write formula using proper notation
- Always write variables with underscore subscripts: V_C, V_S, V_T, T_1, T_2, P_1, P_2, η_th, η_vol
- When giving a formula always explain each term in a Where: section
- If student says "regenerate", "try again", "different answer" → provide same information restructured with more examples

- When giving a formula, always explain each term immediately after like this:
  bmep = bp / (L × A × n × K / 60000)
  Where:
  bmep = brake mean effective pressure (kPa)
  bp   = brake power (kW)
  L    = stroke length (m)
  A    = piston cross-sectional area (m²)
  n    = engine speed (rpm)
  K    = number of cylinders

Context:
{context}

Previous Conversation:
{history}

Question:
{question}

Answer (follow STRICT RULES):"""


class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )
        self.current_model = "llama-3.3-70b-versatile"
        self.llm = ChatGroq(
            model=self.current_model,
            temperature=0
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def switch_model(self, model_name: str):
        """Switch to a different model"""
        if model_name in MODELS:
            self.current_model = model_name
            self.llm = ChatGroq(model=model_name, temperature=0)
            token_stats["model"] = model_name
            token_stats["used"] = 0
            print(f"Switched to model: {model_name}")

    def get_token_stats(self):
        return {
            "used": token_stats["used"],
            "limit": MODELS[self.current_model]["limit"],
            "remaining": MODELS[self.current_model]["limit"] - token_stats["used"],
            "model": self.current_model,
            "label": MODELS[self.current_model]["label"],
            "percent_used": round(token_stats["used"] / MODELS[self.current_model]["limit"] * 100, 1)
        }

    def enhance_question(self, question: str, history: list = []) -> str:
        question = question.strip()

        short_keywords = ["only", "just", "briefly"]
        formula_keywords = ["formula", "equation", "expression"]
        explain_keywords = ["explain", "elaborate", "more detail", "tell me more", "in detail"]

        # If asking for formula without specifying topic — get topic from history
        # If asking for formula without specifying topic — get topic from history
# But NOT if asking about meaning/explanation of formula
        if any(kw in question.lower() for kw in formula_keywords):
            if "meaning" in question.lower() or "explain" in question.lower():
        # Student wants explanation — don't treat as formula request
                pass
            elif history:
                last_user_msg = ""
                for msg in reversed(history):
                     if msg["role"] == "user":
                         last_user_msg = msg["content"]
                         break
                if last_user_msg:
                    return f"formula for {last_user_msg}"

# If asking for formula without specifying topic — get topic from history
        if any(kw in question.lower() for kw in formula_keywords):
            if "meaning" in question.lower() or "explain" in question.lower():
                pass
            else:
                # Check if question already has a topic specified
                # e.g. "formula for thermal efficiency" already has topic
                ic_topics = [
                    "thermal efficiency", "compression ratio", "volumetric efficiency",
                    "brake power", "indicated power", "bmep", "imep", "turbocharger",
                    "otto cycle", "diesel cycle", "stroke", "piston", "cylinder"
                ]
                has_topic = any(topic in question.lower() for topic in ic_topics)
                if not has_topic and history:
                    last_user_msg = ""
                    for msg in reversed(history):
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                            break
                    if last_user_msg:
                        return f"formula for {last_user_msg}"
            return question

        # Short answer — return as is
        if any(kw in question.lower() for kw in short_keywords):
            return question

        # Check if proper question
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
    
    def ask(self, question: str, top_k: int = 6, history: list = []) -> dict:
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

        # Build history string from last 4 messages
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "Student" if msg["role"] == "user" else "Professor"
                history_text += f"{role}: {msg['content']}\n"

        prompt_value = self.prompt.format(
            context=context,
            question=question,
            history=history_text
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

        return {"answer": answer, "sources": sources}
    def generate_quiz(self, topic: str, num_questions: int = 5) -> list:
        """Generate quiz questions from course material"""
        docs = self.vectorstore.similarity_search(topic, k=20)
        context = "\n\n".join(d.page_content for d in docs)

        # Use higher temperature for more creative and varied quiz questions
        # quiz_llm = ChatGroq(
        #     model="llama-3.3-70b-versatile",
        #     temperature=0.4
        # )

        quiz_llm = ChatGroq(
        model=self.current_model,
        temperature=0.4
        )
        quiz_prompt = f"""You are an IC Engine professor creating a quiz.
        Based on the context below, generate exactly {num_questions} multiple choice questions about {topic}.

        Format each question EXACTLY like this (use @@@ as separator between questions):

        Q1: [question text]
        A) [option]
        B) [option]
        C) [option]
        D) [option]
        Answer: [correct letter]
        Explanation: [brief explanation]

        @@@

        Q2: [question text]
        A) [option]
        B) [option]
        C) [option]
        D) [option]
        Answer: [correct letter]
        Explanation: [brief explanation]

        @@@

        Context:
        {context}

        Generate exactly {num_questions} questions now:"""

        response = quiz_llm.invoke(quiz_prompt).content
        return self.parse_quiz(response)

    def parse_quiz(self, raw: str) -> list:
        """Parse raw quiz text into structured questions"""
        questions = []
        blocks = raw.strip().split("@@@")
        for block in blocks:
            if "Answer:" in block:
                lines = block.strip().split("\n")
                q = {"question": "", "options": {}, "answer": "", "explanation": ""}
                for line in lines:
                    line = line.strip()
                    if line and line[0] == "Q" and ":" in line:
                        q["question"] = line.split(":", 1)[1].strip()
                    elif line.startswith("A)"):
                        q["options"]["A"] = line[2:].strip()
                    elif line.startswith("B)"):
                        q["options"]["B"] = line[2:].strip()
                    elif line.startswith("C)"):
                        q["options"]["C"] = line[2:].strip()
                    elif line.startswith("D)"):
                        q["options"]["D"] = line[2:].strip()
                    elif line.startswith("Answer:"):
                        q["answer"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Explanation:"):
                        q["explanation"] = line.split(":", 1)[1].strip()
                if q["question"] and q["options"]:
                    questions.append(q)
        return questions

    def check_answer(self, question: dict, student_answer: str) -> dict:
        """Check if student answer is correct"""
        correct = question["answer"].upper().strip()
        given = student_answer.upper().strip()
        is_correct = given == correct
        return {
            "correct": is_correct,
            "given": given,
            "correct_answer": correct,
            "explanation": question["explanation"]
        }

    def get_doc_count(self) -> int:
        return self.vectorstore._collection.count()
    
    

    def ask_stream(self, question: str, top_k: int = 10, history: list = []):
        """Generator that yields answer chunks as they arrive"""
        question = self.enhance_question(question, history)

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": 20}
        )
        docs = retriever.invoke(question)

        if not docs:
            yield "No documents found in the database."
            return

        context = "\n\n".join(d.page_content for d in docs)

        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "Student" if msg["role"] == "user" else "Professor"
                history_text += f"{role}: {msg['content']}\n"

        prompt_value = self.prompt.format(
            context=context,
            question=question,
            history=history_text
        )

        # Stream the response with auto model switching
        try:
            total_chars = 0
            for chunk in self.llm.stream(prompt_value):
                content = chunk.content
                total_chars += len(content)
                yield content
            # Estimate tokens (rough: 4 chars per token)
            token_stats["used"] += len(prompt_value) // 4 + total_chars // 4

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                # Auto switch to lighter model
                if self.current_model == "llama-3.3-70b-versatile":
                    print("Rate limit hit — switching to 8b model automatically")
                    self.switch_model("llama-3.1-8b-instant")
                    yield "\n\n⚠️ Switched to faster model due to rate limit. Retrying...\n\n"
                    try:
                        for chunk in self.llm.stream(prompt_value):
                            yield chunk.content
                    except Exception as e2:
                        yield "⚠️ Both models rate limited. Please try again tomorrow."
                else:
                    yield "⚠️ Daily limit reached on all models. Please try again tomorrow."
            else:
                yield f"⚠️ An error occurred: {str(e)}"


rag = RAGEngine()
import os
from huggingface_hub import snapshot_download
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

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
    "llama-3.3-70b-versatile":                   {"limit": 100000, "label": "70B (High Quality)"},
    "llama-3.1-8b-instant":                      {"limit": 500000, "label": "8B (Fast)"},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"limit": 100000, "label": "Llama 4 Scout"},
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

PROMPT_TEMPLATE = """You are an expert IC Engine professor at a top engineering university. You have deep knowledge of internal combustion engines and teach with clarity, precision, and pedagogical excellence. Your answers come ONLY from the context provided below.
IMPORTANT: The TYPE labels below are YOUR internal instructions only. NEVER write them in your response. Start your answer directly.

═══════════════════════════════════════════════════════
RESPONSE TYPE DETECTION — read question carefully first
═══════════════════════════════════════════════════════

TYPE 1 — SHORT ANSWER
Trigger: question contains "only", "just", "briefly", "formula only", "definition only", "in one line", "short", "don't give very detailed", "concise"
Response: 1-3 lines MAXIMUM. Formula + Where clause if applicable. Nothing else. No paragraphs.

TYPE 2 — CONCEPTUAL EXPLANATION
Trigger: question starts with "what is", "define", "what are"
Response:
  • Clear definition in 1-2 sentences
  • Physical significance — why it matters
  • Formula with full Where clause
  • Typical values/ranges in real engines
  • Related concepts briefly mentioned

TYPE 3 — DEEP REASONING (use internal Chain of Thought)
Trigger: question contains "why", "how does", "what happens when", "effect of", "impact of"
Process internally:
  → What is the root cause being asked about?
  → What physical/thermodynamic principles apply?
  → What is the chain of causation?
  → What are the practical implications?
Write ONLY the final reasoned answer. Never show internal steps. Never show Step 1, Step 2 etc.

TYPE 4 — COMPARISON
Trigger: question contains "difference", "compare", "vs", "versus", "better than"
Response:
  • Key differences in a structured format
  • Underlying reason for each difference
  • Practical implications
  • Which is better and under what conditions

TYPE 5 — DETAILED EXPLANATION
Trigger: question contains "explain", "describe", "elaborate", "in detail", "discuss"
Response:
  • Full comprehensive explanation
  • Theory and principles
  • Formula with complete Where clause
  • Real world example with typical values
  • Design implications
  • Related concepts

TYPE 6 — DIAGRAM REQUEST
Trigger: question contains "draw", "sketch", "diagram", "show", "illustrate"
Response: Draw detailed ASCII art using these characters:
┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ → ↓ ↑ ← ═ ║ ╔ ╗ ╚ ╝
Label all components clearly. Add a brief explanation after the diagram.

TYPE 7 — NUMERICAL PROBLEM
Trigger: question contains specific numbers, units, "calculate", "find", "determine", "compute"
Response:
  • Identify the correct formula from context
  • List all given values with units
  • Show every step of calculation clearly
  • State final answer with correct units
  • Verify answer is physically reasonable
  • You ARE allowed to solve numerical problems using standard IC engine
    formulas even if the exact problem is not in the textbook

CRITICAL REMINDERS:
  • /60 is MANDATORY in power formulas — converts per minute to per second
  • Diesel Q_in uses Cp = γ × Cv (constant pressure), NOT Cv
  • Diesel T_4 = T_3 × (rc/r)^(γ-1) — NOT T_3 × (1/r)^(γ-1)
  • Brayton cycle exponent is (γ-1)/γ = 0.2857, NOT γ-1 = 0.4
    T_2 = T_1 × rp^0.2857  where rp = pressure ratio
    T_4 = T_3 / rp^0.2857
    NEVER use 0.4 for Brayton — that is the Otto cycle exponent

CRITICAL EXPONENT VALUES — use these exact values:
  Otto/Diesel (γ-1 = 0.4):
  8^0.4   = 2.297
  8.5^0.4 = 2.354
  9^0.4   = 2.408
  10^0.4  = 2.512
  16^0.4  = 3.031
  18^0.4  = 3.178
  20^0.4  = 3.314

  Diesel cutoff (rc^γ where γ=1.4):
  2^1.4   = 2.639
  2.2^1.4 = 3.016
  2.5^1.4 = 3.607
  3^1.4   = 4.656

  Brayton ((γ-1)/γ = 0.2857):
  4^0.2857  = 1.486
  6^0.2857  = 1.669
  8^0.2857  = 1.811
  10^0.2857 = 1.931
  12^0.2857 = 2.031

VERIFICATION STEP — mandatory after solving:
  1. T2 > T1 always (compression raises temperature)
  2. η_th must be between 0 and 1
  3. bp < ip always (friction reduces power)
  4. Cross-check: η_th = W_net / Q_in (must match formula result)
  5. For Brayton: verify T4 > T1 (expansion never cools below ambient)
  If any check fails → recalculate before giving final answer

You ARE allowed to solve numerical problems using these formulas even if the exact problem is not in the textbook.

TYPE 8 — FOLLOW UP
Trigger: vague questions like "explain more", "give details", "elaborate", "tell me more"
Response: Expand on the previous topic with additional depth, different examples, or aspects not yet covered.

TYPE 9 — REGENERATE
Trigger: "regenerate", "try again", "different answer", "rephrase"
Response: Provide the same information restructured completely differently — new examples, different angle, alternate explanation style.

═══════════════════════════════════════
FORMATTING RULES — always apply these
═══════════════════════════════════════

FORMULAS:
Always present formulas with full explanation:
  η_th = 1 - (1 / r^(γ-1))
  Where:
  η_th = thermal efficiency (dimensionless)
  r    = compression ratio
  γ    = ratio of specific heats (≈ 1.4 for air)

VARIABLES — always use underscore subscript notation:
  Volumes:      V_C, V_S, V_T, V_D
  Temperatures: T_1, T_2, T_3, T_4
  Pressures:    P_1, P_2, P_3, P_4
  Efficiencies: η_th, η_vol, η_mech, η_ind
  Powers:       W_net, Q_in, Q_out
  Engine:       bmep, imep, fmep, bp, ip, fp

UNITS — always include units in formulas and answers:
  Power in kW, Pressure in kPa or bar, Volume in cc or m³
  Temperature in °C or K, Speed in rpm, Length in mm or m

NUMERICAL ANSWERS:
  Show all steps. Round to 3 significant figures.
  Always state the unit with the final answer.

LENGTH CONTROL:
  If student says "don't give very detailed", "brief", "short", "concise", "in short" → give short answer only, maximum 5 lines, no lengthy paragraphs
  Match response length to question complexity — never pad with unnecessary text

═══════════════════════════════════════
STRICT CONTENT RULES
═══════════════════════════════════════

✓ Answer ONLY from the context provided
✓ If formula has garbled symbols (��, □) → rewrite using proper notation
✓ Adapt explanation complexity to the question
✓ For comparison questions — be balanced, show both sides
✓ Always be technically accurate

✗ NEVER mention slide times, timestamps, page numbers, document names
✗ NEVER make up data, values, or relationships not supported by context
✗ NEVER show internal reasoning steps (Step 1, Step 2 etc)
✗ NEVER give a long answer when a short one was requested
✗ NEVER ignore the question type — match response to what was asked
✗ NEVER use LaTeX notation — use plain text math only
   Write fractions as: (a/b) instead of LaTeX frac notation
   Write multiplication as: × symbol
   Write powers as: r^(γ-1) not LaTeX superscript notation
   Write multiplication as: × not \times
✗ NEVER pad answers with generic statements about real-world engines, design implications, or related concepts unless specifically asked

═══════════════════════════════════════
BOUNDARY CONDITIONS
═══════════════════════════════════════

If question is NOT about IC engines:
→ "Please ask questions related to IC engines only."

If topic is NOT in the context:
→ "This topic is not covered in the course material."

If question is ambiguous:
→ Answer the most likely interpretation, then note the assumption made.

If student seems confused:
→ Start with the fundamental concept before building up to the answer.

═══════════════════════════════════════════════
PROVIDED INFORMATION
═══════════════════════════════════════════════

Context from course material:
{context}

Previous Conversation:
{history}

Student Question:
{question}

═══════════════════════════════════════════════
YOUR RESPONSE:
═══════════════════════════════════════════════"""

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
        self.google_llm = None  # Disabled until Google model name confirmed
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

# For numerical problems — skip retrieval, use prompt formulas only
        numerical_keywords = ["calculate", "find", "determine", "compute",
                            "bore", "stroke", "rpm", "kpa", "kw", "efficiency"]
        is_numerical = sum(1 for kw in numerical_keywords if kw in question.lower()) >= 3

        if is_numerical:
            docs = []
            context = "Use the formulas provided in your instructions to solve this numerical problem step by step."
        else:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": 20}
            )
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs) if docs else "No context found."

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
    
    

    def is_numerical_question(self, question: str) -> bool:
        """Detect if question is a numerical problem requiring calculation"""
        numerical_keywords = [
            "calculate", "find", "determine", "compute",
            "bore", "stroke", "rpm", "kpa", "kw",
            "efficiency", "pressure", "temperature", "power",
            "volume", "cylinder", "piston", "ratio"
        ]
        count = sum(1 for kw in numerical_keywords if kw in question.lower())
        # Also check if question has numbers with units
        import re
        has_numbers = bool(re.search(r'\d+\s*(mm|kpa|rpm|kw|kj|kg|bar|cc|m)', question.lower()))
        return count >= 3 or (count >= 2 and has_numbers)

    def ask_stream(self, question: str, top_k: int = 10, history: list = []):
        """Generator that yields answer chunks as they arrive"""
        question = self.enhance_question(question, history)

        # ── Numerical question → use compound-mini with code execution ──
        if self.is_numerical_question(question):
            try:
                from groq import Groq as GroqClient
                client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
                yield "🔢 Running calculations...\n\n"

                numerical_prompt = f"""You are an IC Engine professor.
                Solve this numerical problem step by step.

                CRITICAL RULES:
                - A = π/4 × d² — keep ALL decimal places, do NOT round A
                - bp = bmep × L × A × (N/2) × K / 60
                - ip = bp / η_mech
                - fp = ip - bp
                - For 4-stroke: use N/2 for power strokes per minute, then divide by 60
                - 9^0.4=2.408, 10^0.4=2.512, 16^0.4=3.031, 18^0.4=3.178
                - 8^0.4=2.297, 8.5^0.4=2.354, 2.2^1.4=3.016, 2.5^1.4=3.607

                Problem: {question}"""
                response = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[{"role": "user", "content": numerical_prompt}],
                max_tokens=2000
            )
                msg = response.choices[0].message
                # compound-mini may return answer in content or reasoning
                answer = msg.content or ''
                if not answer:
                    answer = getattr(msg, 'reasoning', '') or ''
                if not answer:
                    raise Exception("Empty response from compound-mini")
                # Stream it word by word for consistent UI experience
                words = answer.split(' ')
                for i, word in enumerate(words):
                    if i < len(words) - 1:
                        yield word + ' '
                    else:
                        yield word
                return
            except Exception as e:
                if "429" in str(e):
                    # Compound rate limited → go directly to Scout
                    scout = ChatGroq(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0
                    )
                    try:
                        for chunk in scout.stream(question):
                            yield chunk.content
                    except Exception:
                        pass
                    return
                else:
                    yield f"⚠️ Error: {str(e)}. Retrying...\n\n"
                # Fall through to standard RAG pipeline

        # ── Standard RAG pipeline for conceptual questions ──
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
                if self.current_model == "llama-3.3-70b-versatile":
                    print("Rate limit hit — switching to Llama 4 Scout automatically")
                    self.switch_model("meta-llama/llama-4-scout-17b-16e-instruct")
                    yield "\n\n⚠️ Switched to Llama 4 Scout due to rate limit. Retrying...\n\n"
                    try:
                        for chunk in self.llm.stream(prompt_value):
                            yield chunk.content
                    except Exception as e2:
                        if "429" in str(e2):
                            self.switch_model("llama-3.1-8b-instant")
                            yield "\n\n⚠️ Switched to 8B model. Retrying...\n\n"
                            try:
                                for chunk in self.llm.stream(prompt_value):
                                    yield chunk.content
                            except Exception as e3:
                                yield "⚠️ All models rate limited. Please try again tomorrow."
                        else:
                            yield f"⚠️ An error occurred: {str(e2)}"
                elif self.current_model == "meta-llama/llama-4-scout-17b-16e-instruct":
                    print("Rate limit hit — switching to 8b model automatically")
                    self.switch_model("llama-3.1-8b-instant")
                    yield "\n\n⚠️ Switched to 8B model due to rate limit. Retrying...\n\n"
                    try:
                        for chunk in self.llm.stream(prompt_value):
                            yield chunk.content
                    except Exception as e2:
                        yield "⚠️ All models rate limited. Please try again tomorrow."
                else:
                    yield "⚠️ Daily limit reached on all models. Please try again tomorrow."
            else:
                yield f"⚠️ An error occurred: {str(e)}"


rag = RAGEngine()
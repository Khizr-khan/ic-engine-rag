import streamlit as st
import requests
import re

st.set_page_config(
    page_title="IC Engine Assistant",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

API_URL = "https://khizr72-ic-engine-rag-v2.hf.space"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 860px; }
.stApp { background: #0a0d0e; font-family: 'IBM Plex Sans', sans-serif; }

.blueprint-bg {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none; opacity: 0.05;
}

.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 0;
    background: transparent;
}
.header-icon {
    width: 42px; height: 42px; border-radius: 11px;
    background: rgba(26,58,26,0.9);
    border: 1px solid rgba(74,222,128,0.3);
    display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.header-title { font-size: 17px; font-weight: 600; color: #e8e6df; letter-spacing: -0.2px; }
.header-sub { font-size: 10px; color: #6b7a6b; letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace; }

.user-bubble { display: flex; justify-content: flex-end; margin: 6px 0; }
.user-bubble-inner {
    max-width: 70%;
    background: rgba(20,50,20,0.95);
    border-radius: 16px 16px 3px 16px;
    padding: 12px 16px; color: #d4f7d4;
    font-size: 14px; line-height: 1.65;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.ai-bubble { display: flex; justify-content: flex-start; margin: 6px 0; }
.ai-bubble-inner {
    max-width: 75%;
    background: rgba(16,22,16,0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(74,222,128,0.12);
    border-radius: 16px 16px 16px 3px;
    padding: 14px 17px; color: #e8e6df;
    font-size: 14px; line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}

.empty-state { text-align: center; padding: 60px 20px 40px; }
.empty-icon { font-size: 56px; margin-bottom: 20px; display: block; }
.empty-title { font-size: 22px; font-weight: 600; color: #e8e6df; letter-spacing: -0.3px; margin-bottom: 10px; }
.empty-subtitle { font-size: 13px; color: #6b7a6b; line-height: 1.6; max-width: 360px; margin: 0 auto 32px; }
.sugg-title { font-size: 11px; color: #6b7a6b; letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace; margin-bottom: 12px; text-align: center; }

.error-box {
    background: rgba(40,10,10,0.9); border: 1px solid #5a1a1a;
    border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: #f87171;
    font-family: 'IBM Plex Mono', monospace; margin: 8px 0;
}

.quiz-box {
    background: rgba(16,22,16,0.92);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 12px; padding: 16px;
    margin: 8px 0;
    color: #e8e6df;
    font-size: 14px;
    line-height: 1.7;
}
.quiz-correct {
    color: #4ade80; padding: 10px;
    background: rgba(26,58,26,0.8);
    border-radius: 8px; margin: 8px 0;
}
.quiz-wrong {
    color: #f87171; padding: 10px;
    background: rgba(40,10,10,0.8);
    border-radius: 8px; margin: 8px 0;
}
.quiz-explanation {
    color: #6b7a6b; font-size: 12px; padding: 8px;
    background: rgba(16,22,16,0.8);
    border-radius: 8px; margin: 4px 0;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(14,18,14,0.85) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 12px !important; color: #e8e6df !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important; padding: 12px 15px !important;
    resize: none !important;
    min-height: 80px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
}

.stButton > button {
    background: #4ade80 !important; color: #0a1a0a !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 10px 20px !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: #22c55e !important; }

.sugg-btn > button {
    background: rgba(16,22,16,0.8) !important; color: #6b7a6b !important;
    border: 1px solid rgba(74,222,128,0.15) !important;
    border-radius: 20px !important; font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.sugg-btn > button:hover {
    border-color: #4ade80 !important; color: #4ade80 !important;
    background: rgba(26,58,26,0.8) !important;
}

hr { border-color: rgba(74,222,128,0.1) !important; margin: 16px 0 !important; }

.stSelectbox > div > div {
    background: rgba(14,18,14,0.85) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 10px !important;
    color: #e8e6df !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    cursor: pointer !important;
}
.stSelectbox > div > div > div {
    color: #e8e6df !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stSelectbox input {
    pointer-events: none !important;
    caret-color: transparent !important;
}
</style>

<svg class="blueprint-bg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#4ade80" stroke-width="0.4"/>
    </pattern>
    <pattern id="smallgrid" width="8" height="8" patternUnits="userSpaceOnUse">
      <path d="M 8 0 L 0 0 0 8" fill="none" stroke="#4ade80" stroke-width="0.15"/>
    </pattern>
  </defs>
  <rect width="1200" height="800" fill="url(#smallgrid)"/>
  <rect width="1200" height="800" fill="url(#grid)"/>
  <g transform="translate(580,340)" stroke="#4ade80" fill="none" stroke-width="1">
    <rect x="-80" y="-160" width="160" height="220" rx="4" stroke-width="1.5"/>
    <ellipse cx="0" cy="-160" rx="50" ry="12"/>
    <rect x="-50" y="-160" width="100" height="190"/>
    <rect x="-42" y="-40" width="84" height="50" rx="3" stroke-width="1.5"/>
    <line x1="-30" y1="-40" x2="-30" y2="-30"/>
    <line x1="30" y1="-40" x2="30" y2="-30"/>
    <line x1="-30" y1="-20" x2="30" y2="-20"/>
    <line x1="0" y1="10" x2="0" y2="100" stroke-width="2"/>
    <ellipse cx="0" cy="10" rx="12" ry="8"/>
    <circle cx="0" cy="110" r="40" stroke-width="1.5"/>
    <circle cx="0" cy="110" r="8" stroke-width="2"/>
    <rect x="-70" y="-170" width="12" height="30" rx="2"/>
    <rect x="58" y="-170" width="12" height="30" rx="2"/>
    <circle cx="0" cy="-178" r="6" stroke-width="1.5"/>
    <line x1="0" y1="-172" x2="0" y2="-160"/>
  </g>
  <g transform="translate(140,130)" stroke="#4ade80" fill="none" stroke-width="0.8">
    <text x="-10" y="-60" fill="#4ade80" font-family="monospace" font-size="9" stroke="none">TURBOCHARGER</text>
    <circle cx="0" cy="0" r="50"/>
    <circle cx="0" cy="0" r="15"/>
    <line x1="0" y1="-15" x2="0" y2="-44" stroke-width="1.5"/>
    <line x1="0" y1="15" x2="0" y2="44" stroke-width="1.5"/>
    <line x1="-15" y1="0" x2="-44" y2="0" stroke-width="1.5"/>
    <line x1="15" y1="0" x2="44" y2="0" stroke-width="1.5"/>
    <line x1="60" y1="0" x2="120" y2="0" stroke-width="2"/>
    <circle cx="170" cy="0" r="50"/>
    <circle cx="170" cy="0" r="15"/>
    <line x1="170" y1="-15" x2="170" y2="-44" stroke-width="1.5"/>
    <line x1="170" y1="15" x2="170" y2="44" stroke-width="1.5"/>
    <text x="-70" y="4" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">AIR IN</text>
    <text x="230" y="4" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">HOT GAS</text>
  </g>
  <g transform="translate(80,540)" stroke="#4ade80" fill="none">
    <text x="0" y="-20" fill="#4ade80" font-family="monospace" font-size="9" stroke="none">OTTO CYCLE P-V DIAGRAM</text>
    <line x1="0" y1="0" x2="0" y2="-180" stroke-width="1"/>
    <line x1="0" y1="0" x2="220" y2="0" stroke-width="1"/>
    <text x="100" y="18" fill="#4ade80" font-family="monospace" font-size="8" stroke="none">VOLUME</text>
    <path d="M 180,-10 C 170,-15 150,-20 120,-30 C 90,-45 70,-70 60,-110 L 30,-160 L 30,-20 C 60,-22 100,-18 140,-14 Z" stroke-width="1.5"/>
    <circle cx="30" cy="-20" r="3" fill="#4ade80"/>
    <circle cx="30" cy="-160" r="3" fill="#4ade80"/>
    <circle cx="180" cy="-10" r="3" fill="#4ade80"/>
  </g>
  <g transform="translate(980,160)" stroke="#4ade80" fill="none">
    <text x="-60" y="-30" fill="#4ade80" font-family="monospace" font-size="9" stroke="none">VALVE TIMING</text>
    <circle cx="0" cy="0" r="100" stroke-width="1"/>
    <circle cx="0" cy="0" r="4" fill="#4ade80"/>
    <line x1="0" y1="-105" x2="0" y2="105" stroke-width="0.5" stroke-dasharray="2,4"/>
    <line x1="-105" y1="0" x2="105" y2="0" stroke-width="0.5" stroke-dasharray="2,4"/>
    <text x="2" y="-108" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">TDC</text>
    <text x="2" y="118" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">BDC</text>
    <path d="M 0,-100 A 100,100 0 0,1 87,50" stroke-width="2.5"/>
    <path d="M -87,50 A 100,100 0 0,1 0,-100" stroke-width="2" stroke-dasharray="5,3"/>
    <text x="50" y="-70" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">IVO</text>
    <text x="-105" y="40" fill="#4ade80" font-family="monospace" font-size="7" stroke="none">EVO</text>
  </g>
  <g fill="#4ade80" font-family="monospace" font-size="8" opacity="0.8">
    <text x="50" y="50">CR = Vc/Vs + 1</text>
    <text x="860" y="50">n_th = 1 - 1/r^(y-1)</text>
    <text x="50" y="775">MEP = W_net / V_displacement</text>
    <text x="790" y="775">BSFC = mf / P_brake</text>
    <text x="400" y="50">BORE x STROKE</text>
  </g>
</svg>
""", unsafe_allow_html=True)

# ── Token stats helpers ───────────────────────────────────────────────────────
def get_token_stats():
    try:
        res = requests.get(f"{API_URL}/token-stats", timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

def switch_model(model: str):
    try:
        res = requests.post(
            f"{API_URL}/switch-model",
            json={"model": model},
            timeout=10
        )
        return res.status_code == 200
    except:
        return False

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested" not in st.session_state:
    st.session_state.suggested = None
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "input_counter" not in st.session_state:  
    st.session_state.input_counter = 0
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# ── Header with token tracker and model switcher ──────────────────────────────
stats = get_token_stats()

col_header, col_model, col_tokens = st.columns([3, 2, 2])

with col_header:
    st.markdown("""
    <div class="app-header">
      <div class="header-icon">⚙️</div>
      <div>
        <div class="header-title">IC Engine Assistant</div>
        <div class="header-sub">POWERED BY COURSE MATERIAL</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_model:
    st.markdown("<div style='padding-top:14px'>", unsafe_allow_html=True)
    current_model = stats["model"] if stats else "llama-3.3-70b-versatile"
    selected = st.selectbox(
    "Model",
    options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    format_func=lambda x: {
        "llama-3.3-70b-versatile": "70B — High Quality",
        "llama-3.1-8b-instant": "8B — Fast",
        "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 — Math",
    }[x],
        index=0 if "70b" in current_model else 2 if "scout" in current_model else 1,
        label_visibility="collapsed"
    )
    if selected != current_model:
        if switch_model(selected):
            st.success("Switched!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_tokens:
    st.markdown("<div style='padding-top:14px'>", unsafe_allow_html=True)
    if stats:
        used = stats["used"]
        limit = stats["limit"]
        remaining = stats["remaining"]
        pct = stats["percent_used"]
        color = "#4ade80" if pct < 70 else "#facc15" if pct < 90 else "#f87171"
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{color}">
            TOKENS: {used:,} / {limit:,}<br>
            LEFT: {remaining:,} ({100-pct:.0f}%)<br>
            <div style="background:#1a3a1a;border-radius:4px;height:6px;margin-top:4px">
                <div style="background:{color};width:{min(pct,100)}%;height:6px;border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6b7a6b">
            TOKENS: unavailable
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(74,222,128,0.1);margin:0 0 16px 0'>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_history():
    history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        history.append({"role": role, "content": msg["text"]})
    return history

def format_subscripts(text: str) -> str:
    # Remove markdown bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove LaTeX delimiters
    text = re.sub(r'\\\[|\\\]', '', text)
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1/\2)', text)
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\eta', 'η', text)
    text = re.sub(r'\\gamma', 'γ', text)
    text = re.sub(r'\\pi', 'π', text)
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\\w+', '', text)
    # Subscripts
    text = re.sub(r'V_([A-Za-z0-9]+)', r'V<sub>\1</sub>', text)
    text = re.sub(r'T_([A-Za-z0-9]+)', r'T<sub>\1</sub>', text)
    text = re.sub(r'P_([A-Za-z0-9]+)', r'P<sub>\1</sub>', text)
    text = re.sub(r'η_([A-Za-z0-9]+)', r'η<sub>\1</sub>', text)
    text = re.sub(r'W_([A-Za-z0-9]+)', r'W<sub>\1</sub>', text)
    text = re.sub(r'Q_([A-Za-z0-9]+)', r'Q<sub>\1</sub>', text)
    return text

def call_api(question: str, history: list = []):
    try:
        res = requests.post(
            f"{API_URL}/ask-stream",
            json={"question": question, "top_k": 10, "history": history},
            stream=True,
            timeout=180
        )
        if res.status_code == 200:
            return res, None
        return None, res.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the server."
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except Exception as e:
        return None, str(e)

def call_quiz_api(topic: str, num_questions: int):
    try:
        res = requests.post(
            f"{API_URL}/generate-quiz",
            json={"topic": topic, "num_questions": num_questions},
            timeout=120
        )
        if res.status_code == 200:
            return res.json(), None
        return None, res.json().get("detail", "Unknown error")
    except Exception as e:
        return None, str(e)

def is_quiz_request(question: str):
    patterns = [
        r"ask me (\d+) questions? (?:on|about) (.+)",
        r"quiz me (?:on|about) (.+)",
        r"test me (?:on|about) (.+)",
        r"(\d+) questions? (?:on|about) (.+)",
        r"generate (\d+) questions? (?:on|about) (.+)",
        r"take my quiz (?:on|about) (.+)",
        r"give me a quiz (?:on|about) (.+)",
        r"quiz on (.+)",
        r"take my quiz on (.+)",
        r"take (\d+) questions? quiz (?:on|about) (.+)",
        r"(\d+) questions? quiz (?:on|about) (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            groups = match.groups()
            if len(groups) == 2:
                try:
                    return True, int(groups[0]), groups[1].strip()
                except:
                    return True, 5, groups[1].strip()
            else:
                return True, 5, groups[0].strip()
    return False, 0, ""

def render_message(msg):
    if msg["role"] == "user":
        text = msg["text"].replace('\n', '<br>')
        st.markdown(f"""
        <div class="user-bubble">
          <div class="user-bubble-inner">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        formatted_text = format_subscripts(msg["text"]).replace('\n', '<br>')
        st.markdown(f"""
        <div class="ai-bubble">
          <div class="ai-bubble-inner">{formatted_text}</div>
        </div>
        """, unsafe_allow_html=True)

def render_quiz():
    if not st.session_state.quiz_questions:
        return
    total = len(st.session_state.quiz_questions)
    idx = st.session_state.quiz_index

    if idx >= total:
        score = st.session_state.quiz_score
        pct = int(score / total * 100)
        emoji = "🌟" if pct >= 80 else "📚" if pct >= 50 else "💪"
        msg = "Excellent!" if pct >= 80 else "Good effort! Keep studying" if pct >= 50 else "Need more practice!"
        st.markdown(f"""
        <div class="quiz-box">
        <b>Quiz Complete! 🎉</b><br><br>
        Your Score: {score}/{total} ({pct}%)<br><br>
        {emoji} {msg}
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start New Quiz"):
            st.session_state.quiz_mode = False
            st.session_state.quiz_questions = []
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.rerun()
        return

    q = st.session_state.quiz_questions[idx]
    st.markdown(f"""
    <div class="quiz-box">
    <b>Question {idx+1} of {total}</b><br><br>
    {q['question']}
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.quiz_answered:
        cols = st.columns(2)
        options = q.get("options", {})
        for i, (letter, option) in enumerate(options.items()):
            with cols[i % 2]:
                if st.button(f"{letter}) {option}", key=f"opt_{idx}_{letter}"):
                    correct = q["answer"].upper().strip()[0] if q["answer"].strip() else ""
                    given = letter.upper()
                    is_correct = given == correct
                    if is_correct:
                        st.session_state.quiz_score += 1
                    st.session_state.last_result = {
                        "correct": is_correct,
                        "given": given,
                        "correct_answer": correct,
                        "explanation": q.get("explanation", "")
                    }
                    st.session_state.quiz_answered = True
                    st.rerun()
    else:
        result = st.session_state.last_result
        if result["correct"]:
            st.markdown('<div class="quiz-correct">✅ Correct!</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="quiz-wrong">❌ Wrong! Correct answer: {result["correct_answer"]}</div>',
                unsafe_allow_html=True
            )
        if result["explanation"]:
            st.markdown(
                f'<div class="quiz-explanation">💡 {result["explanation"]}</div>',
                unsafe_allow_html=True
            )
        if st.button("Next Question ➤"):
            st.session_state.quiz_index += 1
            st.session_state.quiz_answered = False
            st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
if len(st.session_state.messages) == 0 and not st.session_state.quiz_mode:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-icon">⚙️</span>
      <div class="empty-title">Ask anything about IC Engines</div>
      <div class="empty-subtitle">
        Answers sourced directly from your course documents.
        Say "ask me 5 questions on SI engine" to start a quiz!
      </div>
    </div>
    <div class="sugg-title">SUGGESTED QUESTIONS</div>
    """, unsafe_allow_html=True)

    suggested_questions = [
        "What is compression ratio?",
        "Explain the 4-stroke diesel cycle",
        "How does turbocharging work?",
        "Difference between SI and CI engines",
        "What is volumetric efficiency?",
        "Ask me 5 questions on SI engine",
    ]

    cols = st.columns(2)
    for i, q in enumerate(suggested_questions):
        with cols[i % 2]:
            st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.suggested = q
            st.markdown('</div>', unsafe_allow_html=True)

else:
    for msg in st.session_state.messages:
        render_message(msg)

    if st.session_state.quiz_mode:
        render_quiz()

    _, _, col3 = st.columns([4, 1, 1])
    with col3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.quiz_mode = False
            st.session_state.quiz_questions = []
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.rerun()

# ── Handle suggested click ────────────────────────────────────────────────────
if st.session_state.suggested:
    question = st.session_state.suggested
    st.session_state.suggested = None

    is_quiz, num_q, topic = is_quiz_request(question)
    if is_quiz:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.spinner(f"Generating {num_q} questions on {topic}..."):
            data, err = call_quiz_api(topic, num_q)
        if err:
            st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
            st.session_state.messages.pop()
        else:
            st.session_state.quiz_questions = data["questions"]
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.session_state.quiz_mode = True
            st.session_state.messages.append({
                "role": "ai",
                "text": f"Starting quiz on '{topic}'! {len(data['questions'])} questions ready. Good luck! 🎯",
                "sources": []
            })
        st.rerun()
    else:
        history = build_history()
        st.session_state.messages.append({"role": "user", "text": question})
        res, err = call_api(question, history=history)
        if err:
            st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
            st.session_state.messages.pop()
        else:
            placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_answer += chunk
                        display_text = format_subscripts(full_answer).replace('\n', '<br>')
                        placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
            except Exception:
                pass
            finally:
                if full_answer:
                    display_text = format_subscripts(full_answer).replace('\n', '<br>')
                    placeholder.markdown(display_text, unsafe_allow_html=True)
                else:
                    placeholder.markdown("Connection dropped. Please try again.", unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "ai",
                "text": full_answer,
                "sources": []
            })
        st.rerun()

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    if "input_area" not in st.session_state:
        st.session_state.input_area = ""

    user_input = st.text_area(
    label="question",
    placeholder="Ask a question or say 'ask me 5 questions on SI engine'...",
    label_visibility="collapsed",
    height=80,
    key=f"input_area_{st.session_state.get('input_counter', 0)}"
    )
with col2:
    st.markdown("<div style='padding-top:20px'>", unsafe_allow_html=True)
    submitted = st.button("SEND ➤", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-size:10px;color:#4b5563;
font-family:'IBM Plex Mono',monospace;letter-spacing:0.05em;margin-top:4px;">
CLICK SEND TO SUBMIT
</div>
""", unsafe_allow_html=True)

# ── Process question ──────────────────────────────────────────────────────────
if submitted and user_input.strip():
    st.session_state.input_counter += 1
    question = user_input.strip()
    if len(question) < 5:
        st.markdown('<div class="error-box">⚠ Question is too short</div>', unsafe_allow_html=True)
    elif len(question) > 1000:
        st.markdown('<div class="error-box">⚠ Question is too long — maximum 1000 characters</div>', unsafe_allow_html=True)
    else:
        is_quiz, num_q, topic = is_quiz_request(question)

        if is_quiz:
            st.session_state.messages.append({"role": "user", "text": question})
            with st.spinner(f"Generating {num_q} questions on {topic}..."):
                data, err = call_quiz_api(topic, num_q)
            if err:
                st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
                st.session_state.messages.pop()
            else:
                st.session_state.quiz_questions = data["questions"]
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answered = False
                st.session_state.quiz_mode = True
                st.session_state.messages.append({
                    "role": "ai",
                    "text": f"Starting quiz on '{topic}'! {len(data['questions'])} questions ready. Good luck! 🎯",
                    "sources": []
                })
            st.rerun()
        else:
            history = build_history()
            st.session_state.messages.append({"role": "user", "text": question})
            res, err = call_api(question, history=history)
            if err:
                st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
                st.session_state.messages.pop()
            else:
                placeholder = st.empty()
                full_answer = ""
                try:
                    for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_answer += chunk
                            display_text = format_subscripts(full_answer).replace('\n', '<br>')
                            placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
                except Exception:
                    pass
                finally:
                    if full_answer:
                        display_text = format_subscripts(full_answer).replace('\n', '<br>')
                        placeholder.markdown(display_text, unsafe_allow_html=True)
                    else:
                        placeholder.markdown("Connection dropped. Please try again.", unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "ai",
                    "text": full_answer,
                    "sources": []
                })
            st.rerun()
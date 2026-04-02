import streamlit as st
import requests

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
    padding: 18px 24px;
    background: rgba(10,14,10,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(74,222,128,0.12);
    margin-bottom: 24px;
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

.stTextInput > div > div > input {
    background: rgba(14,18,14,0.85) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 12px !important; color: #e8e6df !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important; padding: 12px 15px !important;
}
.stTextInput > div > div > input:focus {
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

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested" not in st.session_state:
    st.session_state.suggested = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-icon">⚙️</div>
  <div>
    <div class="header-title">IC Engine Assistant</div>
    <div class="header-sub">POWERED BY COURSE MATERIAL</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii").replace("?", " ")

def build_history():
    """Build history list from current messages"""
    history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        history.append({"role": role, "content": msg["text"]})
    return history

def call_api(question: str, history: list = []):
    try:
        res = requests.post(
            f"{API_URL}/ask-stream",
            json={
                "question": question,
                "top_k": 10,
                "history": history
            },
            stream=True,
            timeout=120
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

def render_message(msg):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
          <div class="user-bubble-inner">{msg["text"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-bubble">
          <div class="ai-bubble-inner">{msg["text"]}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Main chat area ────────────────────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-icon">⚙️</span>
      <div class="empty-title">Ask anything about IC Engines</div>
      <div class="empty-subtitle">
        Answers sourced directly from your course documents
        with exact page citations
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
        "Explain knocking in petrol engines",
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

    _, _, col3 = st.columns([4, 1, 1])
    with col3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Handle suggested click ────────────────────────────────────────────────────
if st.session_state.suggested:
    question = st.session_state.suggested
    st.session_state.suggested = None
    history = build_history()
    st.session_state.messages.append({"role": "user", "text": question})
    res, err = call_api(question, history=history)
    if err:
        st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
        st.session_state.messages.pop()
    else:
        placeholder = st.empty()
        full_answer = ""
        for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_answer += chunk
                placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)
        st.session_state.messages.append({
            "role": "ai",
            "text": full_answer,
            "sources": []
        })
    st.rerun()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="question",
            placeholder="Ask about combustion, cycles, components...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("SEND ➤", use_container_width=True)

st.markdown("""
<div style="text-align:center;font-size:10px;color:#4b5563;
font-family:'IBM Plex Mono',monospace;letter-spacing:0.05em;margin-top:4px;">
PRESS ENTER TO SEND
</div>
""", unsafe_allow_html=True)

# ── Process question ──────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()
    if len(question) < 10:
        st.markdown('<div class="error-box">⚠ Question is too short</div>', unsafe_allow_html=True)
    elif len(question) > 500:
        st.markdown('<div class="error-box">⚠ Question is too long</div>', unsafe_allow_html=True)
    else:
        history = build_history()
        st.session_state.messages.append({"role": "user", "text": question})
        res, err = call_api(question, history=history)
        if err:
            st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
            st.session_state.messaFges.pop()
        else:
            placeholder = st.empty()
            full_answer = ""
            for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    full_answer += chunk
                    placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)
            st.session_state.messages.append({
                "role": "ai",
                "text": full_answer,
                "sources": []
            })
        st.rerun()
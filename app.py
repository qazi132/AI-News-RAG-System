import streamlit as st
import json
import os
from dotenv import load_dotenv
from scraper import scrape_techcrunch
from rag import build_vector_store, query_rag, get_summary

# Load .env at startup 
load_dotenv()
HF_TOKEN     = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(
    page_title="AI News RAG",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e2e2e2; }

.main-header {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0d2e 50%, #0d1a1a 100%);
    border: 1px solid #00ff9d33; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top:0;left:0;right:0;bottom:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,#00ff9d05 2px,#00ff9d05 4px);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Space Mono', monospace; font-size: 2.1rem;
    color: #00ff9d; margin: 0; text-shadow: 0 0 30px #00ff9d66; letter-spacing: -1px;
}
.main-header p { color: #7a8a9a; margin: 0.5rem 0 0; font-size: 0.9rem; font-family: 'Space Mono', monospace; }

.free-badge {
    display: inline-block; background: linear-gradient(135deg,#00ff9d22,#00ff9d44);
    border: 1px solid #00ff9d66; border-radius: 20px; padding: 0.15rem 0.7rem;
    font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #00ff9d;
    margin-left: 0.7rem; vertical-align: middle;
}

/* env key status row */
.env-row {
    display: flex; align-items: center; gap: 0.5rem;
    background: #0e0e18; border: 1px solid #1e2030; border-radius: 8px;
    padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
}
.env-row.ok   { border-left: 3px solid #00ff9d; }
.env-row.fail { border-left: 3px solid #ff4d4d; }
.env-dot-ok   { color: #00ff9d; font-size: 0.9rem; }
.env-dot-fail { color: #ff4d4d; font-size: 0.9rem; }
.env-label    { color: #9a9ab0; flex: 1; }
.env-status-ok   { color: #00ff9d; font-size: 0.7rem; }
.env-status-fail { color: #ff4d4d; font-size: 0.7rem; }

.env-hint {
    background: #110d0d; border: 1px solid #ff4d4d22; border-radius: 6px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.8rem;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #9a6060;
    line-height: 1.6;
}
.env-hint code { color: #fb923c; }

.news-card {
    background: #111118; border: 1px solid #1e2030;
    border-left: 3px solid #00ff9d; border-radius: 8px;
    padding: 1.2rem 1.5rem; margin-bottom: 1rem; transition: border-color 0.2s;
}
.news-card:hover { border-left-color: #7c3aed; background: #13131f; }
.news-card .card-title  { font-size: 1rem; font-weight: 600; color: #e2e2ff; margin-bottom: 0.4rem; line-height: 1.4; }
.news-card .card-date   { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #00ff9d; margin-bottom: 0.6rem; }
.news-card .card-summary{ font-size: 0.88rem; color: #9a9ab0; line-height: 1.6; }
.news-card .card-link a { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #7c3aed; text-decoration: none; }

.answer-box {
    background: #0d1a0d; border: 1px solid #00ff9d44; border-radius: 10px;
    padding: 1.5rem; margin-top: 1rem; font-size: 0.95rem; line-height: 1.8; color: #c8f0d8;
}
.source-chip {
    display: inline-block; background: #1a1a2e; border: 1px solid #7c3aed44;
    border-radius: 4px; padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #a78bfa;
    margin: 0.2rem 0.2rem 0 0;
}
.stat-box {
    background: #111118; border: 1px solid #1e2030;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.stat-box .stat-num   { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #00ff9d; font-weight: 700; }
.stat-box .stat-label { font-size: 0.8rem; color: #6a6a8a; margin-top: 0.2rem; }
.stack-pill {
    display: inline-block; background: #111118; border: 1px solid #1e2030;
    border-radius: 20px; padding: 0.25rem 0.7rem;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #9a9ab0; margin: 0.15rem;
}
.stack-pill.groq { border-color: #f97316aa; color: #fb923c; }
.stack-pill.hf   { border-color: #facc15aa; color: #fde047; }
.stack-pill.free { border-color: #00ff9daa; color: #00ff9d; }
div[data-testid="stSidebar"] { background: #080810; border-right: 1px solid #1e2030; }
.stButton > button {
    background: linear-gradient(135deg,#00ff9d22,#7c3aed22); border: 1px solid #00ff9d66;
    color: #00ff9d; font-family: 'Space Mono', monospace; font-size: 0.82rem;
    border-radius: 6px; transition: all 0.2s; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00ff9d44,#7c3aed44);
    border-color: #00ff9d; box-shadow: 0 0 20px #00ff9d33;
}
.stTextInput > div > div > input, .stTextArea textarea {
    background: #111118 !important; border: 1px solid #1e2030 !important;
    color: #e2e2ff !important; font-family: 'Space Mono', monospace !important; border-radius: 6px !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #00ff9d66 !important; box-shadow: 0 0 0 1px #00ff9d33 !important;
}
.stSelectbox > div > div { background: #111118 !important; border-color: #1e2030 !important; color: #e2e2ff !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00ff9d; font-family: 'Space Mono', monospace;
    font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;
}
.chat-message-user {
    background: #1a1a2e; border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.5rem 0;
    font-size: 0.9rem; color: #c4b5fd;
}
.chat-message-ai {
    background: #0d1a0d; border-left: 3px solid #00ff9d;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.5rem 0;
    font-size: 0.9rem; color: #c8f0d8;
}
.tag-ai   { color: #00ff9d; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.tag-user { color: #a78bfa; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
hr { border-color: #1e2030; }
</style>
""", unsafe_allow_html=True)


for key, val in {"articles": [], "vector_store": None, "chat_history": []}.items():
    if key not in st.session_state:
        st.session_state[key] = val

DATA_FILE = "data/articles.json"

def save_articles(articles):
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(articles, f, indent=2)

def load_articles():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []

def keys_loaded() -> bool:
    """True only if both env keys are present."""
    return bool(HF_TOKEN and GROQ_API_KEY)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-family:Space Mono,monospace;font-size:1.4rem;color:#00ff9d;text-shadow:0 0 20px #00ff9d66;'>⬡ RAG</div>
        <div style='font-size:0.75rem;color:#6a6a8a;margin-top:0.3rem;'>News Intelligence System</div>
        <div style='margin-top:0.6rem;'>
            <span class='stack-pill groq'>⚡ Groq LLM</span>
            <span class='stack-pill hf'>🤗 HF Embeddings</span>
            <span class='stack-pill free'>✦ Free</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # .env Key Status 
    st.markdown("### 🔑 ENV KEY STATUS")

    groq_ok = bool(GROQ_API_KEY)
    hf_ok   = bool(HF_TOKEN)

    st.markdown(f"""
    <div class='env-row {"ok" if groq_ok else "fail"}'>
        <span class='{"env-dot-ok" if groq_ok else "env-dot-fail"}'>{'●' if groq_ok else '○'}</span>
        <span class='env-label'>GROQ_API_KEY</span>
        <span class='{"env-status-ok" if groq_ok else "env-status-fail"}'>{'✓ loaded' if groq_ok else '✗ missing'}</span>
    </div>
    <div class='env-row {"ok" if hf_ok else "fail"}'>
        <span class='{"env-dot-ok" if hf_ok else "env-dot-fail"}'>{'●' if hf_ok else '○'}</span>
        <span class='env-label'>HF_TOKEN</span>
        <span class='{"env-status-ok" if hf_ok else "env-status-fail"}'>{'✓ loaded' if hf_ok else '✗ missing'}</span>
    </div>
    """, unsafe_allow_html=True)

    if not keys_loaded():
        st.markdown("""
        <div class='env-hint'>
        Create a <code>.env</code> file in the project root:<br><br>
        <code>HF_TOKEN="hf_..."</code><br>
        <code>GROQ_API_KEY="gsk_..."</code><br><br>
        Then restart the app.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ MODEL SETTINGS")

    llm_model = st.selectbox(
        "Groq LLM Model",
        options=[
            "llama-3.1-8b-instant",      
            "llama-3.3-70b-versatile",   
        ],
        index=0,
        help="All free on Groq. llama-3.1-8b-instant is fastest; llama-3.3-70b-versatile is smartest.",
    )

    st.markdown("""
    <div style='background:#0d1a0d;border:1px solid #00ff9d22;border-radius:6px;padding:0.6rem 0.8rem;margin:0.4rem 0;'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#00ff9d88;'>EMBEDDING MODEL (auto)</div>
        <div style='font-size:0.8rem;color:#c8f0d8;margin-top:0.2rem;'>all-MiniLM-L6-v2</div>
        <div style='font-size:0.68rem;color:#555570;font-family:Space Mono,monospace;margin-top:0.1rem;'>HuggingFace Inference API</div>
    </div>
    """, unsafe_allow_html=True)

    top_k = st.slider("Sources per answer (k)", 2, 8, 4)

    st.markdown("---")
    st.markdown("### 📡 DATA PIPELINE")

    num_articles = st.slider("Articles to scrape", 5, 30, 10)

    col1, col2 = st.columns(2)
    with col1:
        scrape_btn = st.button("🔄 Scrape", use_container_width=True)
    with col2:
        load_btn   = st.button("💾 Load Cache", use_container_width=True)

    if scrape_btn:
        if not keys_loaded():
            st.error("⚠️ Add both keys to your .env file and restart the app.")
        else:
            with st.spinner("Scraping TechCrunch..."):
                try:
                    articles = scrape_techcrunch(num_articles)
                    save_articles(articles)
                    st.session_state.articles = articles
                except Exception as e:
                    st.error(f"Scrape failed: {e}")
                    st.stop()
            with st.spinner("Building FAISS index via HuggingFace API..."):
                try:
                    st.session_state.vector_store = build_vector_store(articles)
                    st.success(f"✅ {len(articles)} articles indexed!")
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

    if load_btn:
        articles = load_articles()
        if not articles:
            st.warning("No cache found. Scrape first.")
        elif not keys_loaded():
            st.error("⚠️ Add both keys to your .env file and restart the app.")
        else:
            st.session_state.articles = articles
            with st.spinner("Rebuilding FAISS index..."):
                try:
                    st.session_state.vector_store = build_vector_store(articles)
                    st.success(f"✅ Loaded {len(articles)} articles!")
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

    st.markdown("---")
    if st.session_state.articles:
        n = len(st.session_state.articles)
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-num'>{n}</div>
            <div class='stat-label'>Articles indexed</div>
        </div>
        """, unsafe_allow_html=True)

# Main Header 
st.markdown("""
<div class='main-header'>
    <h1>🚀 AI News RAG System</h1>
    <p>Real-time TechCrunch intelligence • Groq LLM • HuggingFace Embeddings • LangChain + FAISS</p>
</div>
""", unsafe_allow_html=True)

# Tabs 
tab1, tab2, tab3 = st.tabs(["💬 Chat with News", "📰 Browse Articles", "📊 Weekly Summary"])

# TAB 1 Chat 
with tab1:
    st.markdown("#### Ask anything about recent tech news")

    if not st.session_state.articles:
        st.info("👈 Click **Scrape** or **Load Cache** in the sidebar to get started.")
    else:
        st.markdown("**Suggested questions:**")
        sugg_cols = st.columns(3)
        suggestions = [
            "What happened in AI this week?",
            "Any news about funding or startups?",
            "What are the latest product launches?",
        ]
        for i, s in enumerate(suggestions):
            if sugg_cols[i].button(s, key=f"sugg_{i}"):
                if not keys_loaded():
                    st.error("⚠️ Keys missing in .env — restart the app after adding them.")
                elif st.session_state.vector_store:
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    with st.spinner(f"Querying {llm_model}..."):
                        answer, sources = query_rag(
                            s, st.session_state.vector_store,
                            model=llm_model, k=top_k,
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                    st.rerun()

        st.markdown("---")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='tag-user'>▶ YOU</div>"
                    f"<div class='chat-message-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='tag-ai'>◆ AI  ({llm_model})</div>"
                    f"<div class='chat-message-ai'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    chips = "".join([f"<span class='source-chip'>📰 {s}</span>" for s in msg["sources"]])
                    st.markdown(f"<div style='margin-top:0.4rem;'>Sources: {chips}</div>", unsafe_allow_html=True)

        user_input = st.text_input(
            "question", placeholder="e.g. What AI startups raised money this week?",
            key="user_q", label_visibility="collapsed",
        )
        ask_btn = st.button("⬡ Ask", use_container_width=False)

        if ask_btn and user_input:
            if not keys_loaded():
                st.error("⚠️ Keys missing in .env — restart the app after adding them.")
            elif not st.session_state.vector_store:
                st.error("⚠️ Build vector store first (Scrape or Load Cache).")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner(f"Searching articles → asking {llm_model}..."):
                    try:
                        answer, sources = query_rag(
                            user_input, st.session_state.vector_store,
                            model=llm_model, k=top_k,
                        )
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()

        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# TAB 2 Browse
with tab2:
    if not st.session_state.articles:
        st.info("👈 Scrape or load articles from the sidebar first.")
    else:
        search_term = st.text_input("🔍 Filter articles", placeholder="Search by title or keyword...")
        articles = st.session_state.articles
        if search_term:
            articles = [
                a for a in articles
                if search_term.lower() in a["title"].lower()
                or search_term.lower() in a.get("summary", "").lower()
            ]
        st.markdown(f"**{len(articles)} articles**")
        for art in articles:
            st.markdown(f"""
            <div class='news-card'>
                <div class='card-date'>{art.get('date','N/A')}</div>
                <div class='card-title'>{art['title']}</div>
                <div class='card-summary'>{art.get('summary','No summary available.')[:300]}...</div>
                <div class='card-link' style='margin-top:0.6rem;'>
                    <a href='{art.get("url","#")}' target='_blank'>↗ Read full article</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# TAB 3 Summary 
with tab3:
    st.markdown("#### 📊 Auto-generated weekly digest")
    if not st.session_state.articles:
        st.info("👈 Scrape articles first.")
    else:
        if st.button("⚡ Generate Digest", use_container_width=False):
            if not GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY missing in .env — restart after adding it.")
            else:
                with st.spinner(f"Generating digest with {llm_model}..."):
                    try:
                        summary = get_summary(st.session_state.articles, model=llm_model)
                        st.markdown(f"<div class='answer-box'>{summary}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
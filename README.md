# 🚀 AI Tech News Summarizer + Q&A (RAG)

> Real-time news intelligence — scrape TechCrunch, chat with articles, and get weekly digests using Retrieval-Augmented Generation. **100% free. Zero local model downloads.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ What It Does

| Feature | Description |
|---|---|
| 🔄 **Live Scraping** | Fetches latest articles from TechCrunch in real time |
| 💾 **Article Cache** | Saves articles to `data/articles.json` for offline reuse |
| 💬 **RAG Chat** | Ask questions like *"What happened in AI this week?"* and get grounded answers with sources |
| 📰 **Browse** | Filter and read all scraped articles in-app |
| 📊 **Weekly Digest** | One-click auto-generated structured summary (themes, funding, big tech, outlook) |
| 🔑 **Env-based keys** | API keys loaded from `.env` — never hardcoded |

---

## 🏗 Project Structure

```
ai_news_rag/
├── app.py               ← Streamlit frontend (3 tabs: Chat, Browse, Summary)
├── utils/
│   ├── scraper.py       ← BeautifulSoup scraper for TechCrunch
│   └── rag.py           ← HuggingFace embeddings + Groq LLM + FAISS RAG chain
├── data/
│   └── articles.json    ← Cached articles (auto-created on first scrape)
├── .env                 ← Your API keys (never commit this)
├── .env.example         ← Template to copy
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1 — Get your 2 free API keys

**Groq** (LLM — generation):
1. Sign up at [console.groq.com](https://console.groq.com) — no credit card needed
2. Go to **API Keys → Create API Key**
3. Copy the `gsk_...` key

**HuggingFace** (Embeddings):
1. Sign up at [huggingface.co](https://huggingface.co) — free
2. Go to **Settings → Access Tokens → New Token (Read)**
3. Copy the `hf_...` token

---

### 2 — Clone & install

```bash
git clone https://github.com/your-username/ai-news-rag.git
cd ai-news-rag

python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

### 3 — Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
HF_TOKEN="hf_your_token_here"
GROQ_API_KEY="gsk_your_key_here"
```

---

### 4 — Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

In the sidebar:
1. Click **🔄 Scrape** to fetch the latest TechCrunch articles
2. Switch to the **💬 Chat** tab
3. Ask anything!

---

## 🧠 How It Works (Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                    USER OPENS APP                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              SCRAPING  (BeautifulSoup)                  │
│   TechCrunch /latest/ → title, date, content, summary  │
│               saved to data/articles.json              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            CHUNKING  (LangChain TextSplitter)           │
│         800-char chunks with 100-char overlap          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│     EMBEDDING  (HuggingFace Inference API — FREE)       │
│       model: sentence-transformers/all-MiniLM-L6-v2    │
│           Pure API call — zero local download          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           VECTOR STORE  (FAISS — in-memory)             │
│      Stores all chunk embeddings for similarity search  │
└────────────────────────┬────────────────────────────────┘
                         │
              User asks a question
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│      RETRIEVAL  (HuggingFace API embeds the query)      │
│       FAISS finds top-k most relevant article chunks   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         GENERATION  (Groq API — FREE, ultra-fast)       │
│   model: llama-3.1-8b-instant / llama-3.3-70b-versatile│
│     LangChain RetrievalQA → grounded answer + sources  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         ANSWER displayed in Streamlit Chat UI           │
│              with source article titles                 │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Scraping | BeautifulSoup4 + Requests |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (in-memory) |
| LLM | Groq API (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`) |
| RAG Chain | LangChain `RetrievalQA` |
| Config | `python-dotenv` + `.env` file |

---

## 🔧 Customisation

- **Change news source** → edit `BASE_URL` in `utils/scraper.py`
- **Switch LLM** → select in the sidebar dropdown (both models are free on Groq)
- **Adjust chunk size** → edit `chunk_size` in `utils/rag.py`
- **Persistent vector store** → swap `FAISS` for `Chroma` or `Qdrant` in `rag.py`
- **More articles** → adjust the slider in the sidebar (up to 30)

---

## ⚠️ Notes

- **No model downloads** — embeddings run on HuggingFace servers, LLM runs on Groq servers
- First HuggingFace call may be slow (~10s cold start); subsequent calls are fast
- Groq free tier: ~14,400 requests/day on `llama-3.1-8b-instant` — more than enough for personal use
- TechCrunch HTML structure may change over time; update selectors in `scraper.py` if scraping breaks
- For production, replace in-memory FAISS with a persistent store like Chroma or Qdrant

---

## 📄 License

MIT — free to use, modify, and distribute.

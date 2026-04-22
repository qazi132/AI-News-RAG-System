"""
rag.py — Build vector store, run RAG queries, and generate summaries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keys are loaded automatically from your .env file:
    HF_TOKEN      → HuggingFace Inference API (embeddings)
    GROQ_API_KEY  → Groq API (LLM generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import numpy as np
import requests
from typing import List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Load .env file at import time
load_dotenv()

HF_TOKEN     = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBED_URL   = (
    f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}/pipeline/feature-extraction"
)
DEFAULT_LLM = "llama-3.1-8b-instant"   


def _require(key_name: str, value: str | None) -> str:
    """Raise a clear error if a required env variable is missing."""
    if not value:
        raise EnvironmentError(
            f"\n\n❌  '{key_name}' not found.\n"
            f"    Add it to your .env file:\n"
            f"    {key_name}=\"your_key_here\"\n"
        )
    return value


# HuggingFace Inference API Embeddings

class HuggingFaceAPIEmbeddings(Embeddings):
    """
    Calls HuggingFace Serverless Inference API for sentence embeddings.
    Reads HF_TOKEN from environment — no local model download needed.
    """

    def __init__(self):
        token = _require("HF_TOKEN", HF_TOKEN)
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = HF_EMBED_URL

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                # Mean-pool token-level vectors if returned
                if isinstance(item[0], list):
                    results.append(np.mean(item, axis=0).tolist())
                else:
                    results.append(item)
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


# Build FAISS vector store

def build_vector_store(articles: list[dict]) -> FAISS:
    """
    Chunk articles → embed via HuggingFace API → store in FAISS.
    HF_TOKEN is read from the environment automatically.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    docs = []
    for art in articles:
        text = f"TITLE: {art['title']}\nDATE: {art['date']}\n\n{art['content']}"
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": art["title"],
                        "url":   art["url"],
                        "date":  art["date"],
                    },
                )
            )

    embeddings = HuggingFaceAPIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


# RAG Query 

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert tech journalist and analyst. Use the following news articles to answer the question accurately and concisely.

Always ground your answer in the provided context. If the context doesn't cover the question, say so honestly.
Format your answer in clear paragraphs. Mention specific companies, people, or products when relevant.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
)


def query_rag(
    question: str,
    vector_store: FAISS,
    model: str = DEFAULT_LLM,
    k: int = 4,
) -> tuple[str, list[str]]:
    """
    1. Embed question via HuggingFace API  (uses HF_TOKEN from .env)
    2. Retrieve top-k chunks from FAISS
    3. Generate answer via Groq             (uses GROQ_API_KEY from .env)
    Returns (answer_text, deduplicated_source_titles).
    """
    groq_key = _require("GROQ_API_KEY", GROQ_API_KEY)

    hf_embeddings = HuggingFaceAPIEmbeddings()
    vector_store.embedding_function = hf_embeddings.embed_query

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(
        model=model,
        groq_api_key=groq_key,
        temperature=0.2,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    seen, sources = set(), []
    for doc in result.get("source_documents", []):
        title = doc.metadata.get("title", "Unknown")
        if title not in seen:
            seen.add(title)
            sources.append(title[:60] + ("..." if len(title) > 60 else ""))

    return answer, sources


# Weekly Summary 

def get_summary(articles: list[dict], model: str = DEFAULT_LLM) -> str:
    """
    Generate a structured weekly digest using Groq.
    GROQ_API_KEY is read from the environment automatically.
    """
    groq_key = _require("GROQ_API_KEY", GROQ_API_KEY)

    llm = ChatGroq(
        model=model,
        groq_api_key=groq_key,
        temperature=0.3,
    )

    digest_input = "\n\n".join(
        [f"[{a['date']}] {a['title']}\n{a.get('summary', '')[:200]}" for a in articles]
    )

    messages = [
        SystemMessage(content=(
            "You are a senior tech journalist writing a weekly briefing for executives. "
            "Be concise, insightful, and structured. Use markdown with headers."
        )),
        HumanMessage(content=(
            "Here are this week's top tech news items. Write a structured weekly digest "
            "covering: (1) Key Themes, (2) AI & Machine Learning highlights, "
            "(3) Funding & Startups, (4) Big Tech moves, (5) What to watch next week.\n\n"
            f"NEWS ITEMS:\n{digest_input}"
        )),
    ]

    return llm.invoke(messages).content

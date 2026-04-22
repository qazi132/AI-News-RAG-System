"""
scraper.py — Scrapes latest articles from TechCrunch
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://techcrunch.com/latest/"


def scrape_techcrunch(max_articles: int = 10) -> list[dict]:
    """
    Scrape the TechCrunch /latest/ feed and return a list of article dicts.
    Each dict contains: title, url, date, summary, content
    """
    print(f"[Scraper] Fetching {BASE_URL} ...")
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    articles = []

    # TechCrunch article cards
    cards = soup.select("div.wp-block-tc23-post-picker, article, div[class*='post-block']")

    # Fallback: find <a> tags with article-like hrefs
    if not cards:
        cards = soup.select("a[href*='techcrunch.com/20']")

    seen_urls = set()
    raw_links = []

    for card in cards:
        link_tag = card.find("a", href=True) if card.name != "a" else card
        if not link_tag:
            continue
        url = link_tag.get("href", "")
        if not url or url in seen_urls:
            continue
        if "techcrunch.com/20" not in url:
            continue
        seen_urls.add(url)
        raw_links.append(url)
        if len(raw_links) >= max_articles:
            break

    # Brute-force fallback
    if not raw_links:
        for a in soup.find_all("a", href=True):
            url = a["href"]
            if "techcrunch.com/20" in url and url not in seen_urls:
                seen_urls.add(url)
                raw_links.append(url)
                if len(raw_links) >= max_articles:
                    break

    print(f"[Scraper] Found {len(raw_links)} article links. Fetching content...")

    for url in raw_links[:max_articles]:
        try:
            article_data = fetch_article(url)
            if article_data:
                articles.append(article_data)
            time.sleep(0.5)
        except Exception as e:
            print(f"[Scraper] Failed to fetch {url}: {e}")
            continue

    print(f"[Scraper] Done. {len(articles)} articles collected.")
    return articles


def fetch_article(url: str) -> dict | None:
    """Fetch and parse a single TechCrunch article."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Date
    date = "Unknown"
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime", time_tag.get_text(strip=True))
        try:
            date = datetime.fromisoformat(date[:19]).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Content
    content_div = (
        soup.find("div", class_="article-content")
        or soup.find("div", class_="entry-content")
        or soup.find("div", {"class": lambda c: c and "content" in c})
    )

    paragraphs = []
    if content_div:
        paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
    else:
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 60]

    full_content = "\n\n".join(paragraphs)
    summary = " ".join(paragraphs[:2]) if paragraphs else "No summary available."

    if not full_content:
        return None

    return {
        "title": title,
        "url": url,
        "date": date,
        "summary": summary[:500],
        "content": full_content[:4000],
    }

import os
import re
import logging
import requests
import numpy as np

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI
from threading import Thread

# =========================================================
# CONFIG
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("confluence-rag")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Confluence RAG")

# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80
TOP_K = 8

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4.1-mini"

# =========================================================
# MEMORY
# =========================================================

documents = []

job_state = {
    "running": False,
    "progress": 0,
    "message": "Idle"
}

# =========================================================
# HELPERS
# =========================================================

def clean_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed(text: str):
    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000]
    )
    return res.data[0].embedding


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# =========================================================
# INDEXING
# =========================================================

def fetch_confluence_pages():

    global documents, job_state

    new_docs = []
    start = 0
    limit = 50
    processed = 0

    job_state["running"] = True
    job_state["progress"] = 1
    job_state["message"] = "Indexing..."

    while True:

        url = (
            f"{CONFLUENCE_BASE_URL}/rest/api/content"
            f"?expand=body.storage&limit={limit}&start={start}"
        )

        r = requests.get(
            url,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
            timeout=60
        )

        if r.status_code != 200:
            break

        data = r.json()
        results = data.get("results", [])

        if not results:
            break

        for page in results:

            try:
                title = page.get("title", "Untitled")
                html = page["body"]["storage"]["value"]
                text = clean_html(html)

                if len(text) < 50:
                    continue

                page_url = f"{CONFLUENCE_BASE_URL}{page.get('_links', {}).get('webui','')}"
                chunks = chunk_text(text)

                for c in chunks:
                    emb = embed(c)
                    new_docs.append({
                        "title": title,
                        "text": c,
                        "url": page_url,
                        "embedding": emb
                    })

                processed += 1
                job_state["progress"] = min(int(processed * 2), 99)
                job_state["message"] = f"Processing {title}"

            except Exception as e:
                logger.error(e)

        start += limit

    documents = new_docs

    job_state["progress"] = 100
    job_state["running"] = False
    job_state["message"] = "Completed"

# =========================================================
# BACKGROUND
# =========================================================

def run_refresh():
    try:
        fetch_confluence_pages()
    except Exception as e:
        logger.error(e)
        job_state["running"] = False

# =========================================================
# SEARCH
# =========================================================

def search(query: str):

    if not documents:
        return []

    q_emb = embed(query)

    scored = []

    for d in documents:
        score = cosine(q_emb, d["embedding"])
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:TOP_K]

# =========================================================
# GPT (ONLY ONE BEST REFERENCE LINK)
# =========================================================

def answer(query, results):

    if not results:
        return "No data found", None

    best = results[0][1]

    context = f"{best['title']}\n{best['text']}"

    res = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Answer strictly from context."},
            {"role": "user", "content": f"Q: {query}\n\nContext:\n{context}"}
        ]
    )

    return res.choices[0].message.content, best["url"]

# =========================================================
# UI (FIXED EXACTLY AS REQUESTED)
# =========================================================

def render(content="", link=None):

    link_html = ""
    if link:
        link_html = f"""
        <div style="margin-top:20px;">
            <b>Reference:</b>
            <a href="{link}" target="_blank">Open Most Relevant Page</a>
        </div>
        """

    return f"""
    <html>
    <head>
        <title>Confluence RAG</title>

        <style>

            body {{
                font-family: Arial;
                margin: 40px;
                background: #f5f5f5;
            }}

            /* =========================
               🔴 RED PROGRESS LINE ONLY
            ========================= */
            #progress-bar {{
                position: fixed;
                top: 0;
                left: 0;
                height: 6px;
                width: 0%;
                background: red;
                z-index: 9999;
                transition: width 0.3s ease;
            }}

            /* =========================
               🔴 RED BANNER HEADER
            ========================= */
            .banner {{
                background: red;
                color: white;
                padding: 15px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                border-radius: 6px;
                margin-bottom: 20px;
            }}

            input {{
                width: 550px;
                padding: 12px;
                font-size: 16px;
                border-radius: 6px;
                border: 1px solid #ccc;
            }}

            .btn {{
                background: red;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 15px;
            }}

            /* NAVY BLUE BUTTON */
            .refresh-btn {{
                background: #001f4d;
                margin-left: 10px;
            }}

            .box {{
                background: white;
                padding: 20px;
                margin-top: 20px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}

            .footer {{
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background: #111;
                color: white;
                text-align: center;
                padding: 10px;
            }}

        </style>
    </head>

    <body>

        <div id="progress-bar"></div>

        <!-- 🔴 FULL RED BANNER RESTORED -->
        <div class="banner">
            Emirates Group Confluence Search ( OpenAI RAG Solution )
        </div>

        <div style="display:flex; align-items:center;">

            <form method="post" action="/ask">
                <input name="query" placeholder="Ask Confluence..." required />
                <button class="btn">ASK</button>
            </form>

            <form onsubmit="event.preventDefault(); startRefresh();">
                <button class="btn refresh-btn">Refresh Confluence</button>
            </form>

        </div>

        <div class="box">
            {content}
            {link_html}
        </div>

        <div class="footer">
            POC Project By Syed Abbas Rizvi
        </div>

<script>

const bar = document.getElementById("progress-bar");

async function startRefresh() {{

    await fetch("/refresh", {{ method: "POST" }});

    const interval = setInterval(async () => {{

        const res = await fetch("/status");
        const data = await res.json();

        bar.style.width = data.progress + "%";

        if (!data.running) {{
            clearInterval(interval);
            bar.style.width = "100%";

            setTimeout(() => {{
                bar.style.width = "0%";
            }}, 800);
        }}

    }}, 800);
}}

</script>

    </body>
    </html>
    """

# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return render()


@app.post("/refresh")
def refresh():
    if not job_state["running"]:
        Thread(target=run_refresh, daemon=True).start()
    return {"status": "started"}


@app.get("/status")
def status():
    return job_state


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):

    results = search(query)

    if not results:
        return render("No results found")

    ans, link = answer(query, results)

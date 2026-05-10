import os
import re
import html
import time
import threading
import logging
import requests
import numpy as np

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from openai import OpenAI

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("confluence-rag")

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY missing")

if not CONFLUENCE_BASE_URL:
    raise Exception("CONFLUENCE_BASE_URL missing")

if not CONFLUENCE_USERNAME:
    raise Exception("CONFLUENCE_USERNAME missing")

if not CONFLUENCE_API_TOKEN:
    raise Exception("CONFLUENCE_API_TOKEN missing")

# =========================================================
# OPENAI
# =========================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60
)

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Enterprise Confluence RAG"
)

# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

TOP_K = 8

MAX_CHUNKS = 5000

EMBEDDING_MODEL = "text-embedding-3-small"

GPT_MODEL = "gpt-4.1-mini"

STOPWORDS = {
    "the", "is", "was", "when", "where", "how",
    "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "at", "from",
    "that", "this", "are", "were", "be", "been",
    "has", "have", "had"
}

# =========================================================
# MEMORY STORE
# =========================================================

documents = []

index_loaded = False

refresh_running = False

# =========================================================
# HELPERS
# =========================================================

def html_escape(text):

    if not text:
        return ""

    return html.escape(text).replace("\n", "<br>")


def clean_html(raw_html: str):

    soup = BeautifulSoup(
        raw_html,
        "html.parser"
    )

    text = soup.get_text(" ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text: str):

    words = re.findall(
        r"\w+",
        text.lower()
    )

    return [
        w for w in words
        if w not in STOPWORDS
        and len(w) > 2
    ]


def chunk_text(
    text,
    size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + size

        chunk = " ".join(
            words[start:end]
        )

        if len(chunk.strip()) > 100:
            chunks.append(chunk)

        start += (size - overlap)

    return chunks


def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


def create_embedding(text):

    try:

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]
        )

        return response.data[0].embedding

    except Exception:

        logger.exception(
            "Embedding failed"
        )

        return None

# =========================================================
# CONFLUENCE INDEXER
# =========================================================

def fetch_confluence_pages():

    global documents
    global index_loaded
    global refresh_running

    if refresh_running:

        logger.info(
            "Refresh already running"
        )

        return

    refresh_running = True

    logger.info(
        "Refreshing Confluence index..."
    )

    try:

        new_documents = []

        start = 0
        limit = 50

        total_pages = 0

        while True:

            try:

                url = (
                    f"{CONFLUENCE_BASE_URL}"
                    f"/rest/api/content"
                    f"?expand=body.storage"
                    f"&limit={limit}"
                    f"&start={start}"
                )

                response = requests.get(
                    url,
                    auth=(
                        CONFLUENCE_USERNAME,
                        CONFLUENCE_API_TOKEN
                    ),
                    timeout=60
                )

                if response.status_code != 200:

                    logger.error(
                        f"Confluence fetch failed: "
                        f"{response.status_code}"
                    )

                    break

                data = response.json()

                results = data.get(
                    "results",
                    []
                )

                if not results:
                    break

                for page in results:

                    try:

                        title = page.get(
                            "title",
                            "Untitled"
                        )

                        html_content = (
                            page["body"]
                            ["storage"]
                            ["value"]
                        )

                        cleaned_text = clean_html(
                            html_content
                        )

                        if len(cleaned_text) < 50:
                            continue

                        relative_url = (
                            page.get("_links", {})
                            .get("webui", "")
                        )

                        page_url = (
                            f"{CONFLUENCE_BASE_URL}"
                            f"{relative_url}"
                        )

                        chunks = chunk_text(
                            cleaned_text
                        )

                        logger.info(
                            f"Indexing: {title}"
                        )

                        for chunk in chunks:

                            if len(new_documents) >= MAX_CHUNKS:

                                logger.warning(
                                    "MAX_CHUNKS reached"
                                )

                                break

                            embedding = create_embedding(
                                chunk
                            )

                            if not embedding:
                                continue

                            new_documents.append({

                                "title": title,

                                "text": chunk,

                                "url": page_url,

                                "embedding": embedding
                            })

                        total_pages += 1

                    except Exception:

                        logger.exception(
                            "Page processing failed"
                        )

                start += limit

                time.sleep(0.3)

            except Exception:

                logger.exception(
                    "Fetch loop failed"
                )

                break

        documents = new_documents

        index_loaded = True

        logger.info(
            f"Indexed pages: {total_pages}"
        )

        logger.info(
            f"Loaded chunks: {len(documents)}"
        )

    finally:

        refresh_running = False

# =========================================================
# SEARCH
# =========================================================

def semantic_search(query):

    if not documents:
        return []

    query_embedding = create_embedding(
        query
    )

    if not query_embedding:
        return []

    query_words = tokenize(query)

    scored_results = []

    for doc in documents:

        try:

            similarity = cosine_similarity(
                query_embedding,
                doc["embedding"]
            )

            keyword_bonus = 0

            title = doc["title"].lower()

            text = doc["text"].lower()

            for word in query_words:

                if word in title:
                    keyword_bonus += 0.40

                if word in text:
                    keyword_bonus += 0.20

            if query.lower() in text:
                keyword_bonus += 1.5

            final_score = (
                similarity +
                keyword_bonus
            )

            scored_results.append(
                (final_score, doc)
            )

        except Exception:

            logger.exception(
                "Scoring failed"
            )

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return scored_results[:TOP_K]

# =========================================================
# GPT ANSWER
# =========================================================

def generate_answer(query, results):

    try:

        context = "\n\n".join([

            f"""
TITLE:
{doc['title']}

CONTENT:
{doc['text']}
"""

            for score, doc in results
        ])

        prompt = f"""
You are an enterprise Confluence assistant.

Answer ONLY using provided context.

Rules:

- Give concise answers
- Do not hallucinate
- Combine facts if needed
- If answer unavailable say exactly:
  Information not found in Confluence.

QUESTION:
{query}

CONTEXT:
{context}
"""

        response = client.chat.completions.create(

            model=GPT_MODEL,

            temperature=0.1,

            messages=[

                {
                    "role": "system",
                    "content":
                    "You answer using enterprise knowledge only."
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:
            return (
                "Information not found "
                "in Confluence."
            )

        return answer.strip()

    except Exception:

        logger.exception(
            "GPT failed"
        )

        return (
            "Information not found "
            "in Confluence."
        )

# =========================================================
# HTML UI
# =========================================================

def render(content=""):

    return f"""
    <html>

    <head>

        <title>Confluence RAG</title>

        <meta charset="utf-8">

        <style>

            body {{
                font-family: Arial;
                margin: 40px;
                background: #f5f5f5;
                padding-top: 20px;
            }}

            .top-strip {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 6px;
                background: red;
                z-index: 9999;
            }}

            .progress-container {{
                position: fixed;
                top: 6px;
                left: 0;
                width: 100%;
                height: 4px;
                background: #d9d9d9;
                z-index: 9999;
                display: none;
            }}

            .progress-bar {{
                width: 0%;
                height: 100%;
                background: #007bff;
                transition: width 0.2s ease;
            }}

            h2 {{
                color: #111;
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

            .refresh-btn {{
                background: #111;
            }}

            .box {{
                background: white;
                padding: 20px;
                margin-top: 20px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}

            .source {{
                margin-top: 20px;
                border-top: 1px solid #ddd;
                padding-top: 15px;
            }}

            a {{
                color: blue;
                text-decoration: none;
            }}

            .footer {{
                margin-top: 40px;
                background: #111;
                color: white;
                text-align: center;
                padding: 12px;
                border-radius: 6px;
            }}

        </style>

    </head>

    <body>

        <div class="top-strip"></div>

        <div
            class="progress-container"
            id="progressContainer"
        >
            <div
                class="progress-bar"
                id="progressBar"
            ></div>
        </div>

        <h2>
            Emirates Group Confluence Search
            ( OpenAI RAG Solution )
        </h2>

        <div
            style="
                display:flex;
                align-items:center;
                gap:10px;
                flex-wrap:wrap;
            "
        >

            <form method="post" action="/ask">

                <input
                    name="query"
                    placeholder="Ask Confluence..."
                    required
                />

                <button class="btn">
                    ASK Confluence
                </button>

            </form>

            <form method="post" action="/refresh">

                <button
                    class="btn refresh-btn"
                    type="submit"
                >
                    Refresh Confluence
                </button>

            </form>

        </div>

        <div class="box">

            {content}

        </div>

        <div class="footer">

            POC Project By :
            Syed Abbas Rizvi,
            STE,
            Skywards Emirates Airlines

        </div>

        <script>

        const askForm = document.querySelector(
            'form[action="/ask"]'
        );

        const refreshForm = document.querySelector(
            'form[action="/refresh"]'
        );

        const progressContainer = document.getElementById(
            'progressContainer'
        );

        const progressBar = document.getElementById(
            'progressBar'
        );

        function startProgress() {{

            progressContainer.style.display = 'block';

            let width = 0;

            const interval = setInterval(() => {{

                if (width >= 90) {{
                    clearInterval(interval);
                    return;
                }}

                width += 5;

                progressBar.style.width =
                    width + '%';

            }}, 200);
        }}

        if (askForm) {{

            askForm.addEventListener(
                'submit',
                startProgress
            );
        }}

        if (refreshForm) {{

            refreshForm.addEventListener(
                'submit',
                startProgress
            );
        }}

        window.onload = function () {{

            progressBar.style.width = '100%';

            setTimeout(() => {{

                progressContainer.style.display =
                    'none';

                progressBar.style.width = '0%';

            }}, 400);
        }};

        </script>

    </body>

    </html>
    """

# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    status = (
        "Confluence index loaded"
        if index_loaded
        else "Confluence indexing in progress..."
    )

    return render(
        f"""
        <h3>System Ready</h3>

        <div style="margin-top:15px;">
            {status}
        </div>
        """
    )

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():

    logger.info(
        "Application started"
    )

    logger.info(
        "Background indexing started"
    )

    thread = threading.Thread(
        target=fetch_confluence_pages
    )

    thread.daemon = True

    thread.start()

# =========================================================
# MANUAL REFRESH
# =========================================================

@app.post(
    "/refresh",
    response_class=HTMLResponse
)
def refresh_confluence():

    try:

        thread = threading.Thread(
            target=fetch_confluence_pages
        )

        thread.daemon = True

        thread.start()

        return render(
            """
            <h3>Refresh Started</h3>

            <div style="margin-top:15px;">
                Confluence refresh running
                in background.
            </div>
            """
        )

    except Exception:

        logger.exception(
            "Refresh failed"
        )

        return render(
            """
            <h3>Error</h3>

            <div style="margin-top:15px;">
                Refresh failed.
            </div>
            """
        )

# =========================================================
# ASK ROUTE
# =========================================================

@app.post(
    "/ask",
    response_class=HTMLResponse
)
def ask(query: str = Form(...)):

    try:

        if not index_loaded:

            return render(
                """
                <h3>Indexing In Progress</h3>

                <div style="margin-top:15px;">
                    Confluence indexing is still running.
                    Please retry in few minutes.
                </div>
                """
            )

        results = semantic_search(query)

        if not results:

            return render(
                """
                <h3>Answer</h3>

                <div style="margin-top:15px;">
                    Information not found in Confluence.
                </div>
                """
            )

        answer = generate_answer(
            query,
            results
        )

        safe_answer = html_escape(
            answer
        )

        if (
            "Information not found"
            in answer
        ):

            return render(
                f"""
                <h3>Answer</h3>

                <div style="margin-top:15px;">
                    {safe_answer}
                </div>
                """
            )

        # =========================================
        # ONLY MOST RELEVANT SOURCE
        # =========================================

        best_score, best_doc = results[0]

        title = html_escape(
            best_doc["title"]
        )

        url = best_doc["url"]

        sources_html = f"""

        <div class="source">

            <b>{title}</b>

            <br><br>

            <a
                href="{url}"
                target="_blank"
            >
                Open Most Relevant Confluence Page
            </a>

        </div>
        """

        html_content = f"""

        <h3>Answer</h3>

        <div style="margin-top:15px;">
            {safe_answer}
        </div>

        <br>

        <h3>Most Relevant Source</h3>

        {sources_html}
        """

        return render(html_content)

    except Exception:

        logger.exception(
            "Search failed"
        )

        return render(
            """
            <h3>Error</h3>

            <div style="margin-top:15px;">
                Internal server error occurred.
            </div>
            """
        )

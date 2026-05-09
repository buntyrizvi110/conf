
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

# =========================================================
# CONFIG
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("confluence-rag")

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

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Enterprise Confluence RAG")

# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80

TOP_K = 10

EMBEDDING_MODEL = "text-embedding-3-small"

GPT_MODEL = "gpt-4.1-mini"

STOPWORDS = {
    "the", "is", "was", "when", "where", "how",
    "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "at", "from",
    "that", "this", "are", "were"
}

# =========================================================
# MEMORY STORE
# =========================================================

documents = []

# =========================================================
# HELPERS
# =========================================================

def clean_html(html: str) -> str:

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(" ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str):

    words = re.findall(r"\w+", text.lower())

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + size

        chunk = " ".join(words[start:end])

        if len(chunk.strip()) > 100:
            chunks.append(chunk)

        start += size - overlap

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


def create_embedding(text: str):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000]
    )

    return response.data[0].embedding

# =========================================================
# CONFLUENCE INDEXER
# =========================================================

def fetch_confluence_pages():

    global documents

    logger.info("Refreshing Confluence index...")

    new_documents = []

    start = 0
    limit = 50

    while True:

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

        results = data.get("results", [])

        if not results:
            break

        for page in results:

            try:

                title = page.get(
                    "title",
                    "Untitled"
                )

                html = (
                    page["body"]["storage"]["value"]
                )

                text = clean_html(html)

                if len(text) < 50:
                    continue

                relative_url = (
                    page.get("_links", {})
                    .get("webui", "")
                )

                page_url = (
                    f"{CONFLUENCE_BASE_URL}"
                    f"{relative_url}"
                )

                chunks = chunk_text(text)

                logger.info(
                    f"Indexing: {title}"
                )

                for chunk in chunks:

                    embedding = create_embedding(chunk)

                    new_documents.append({

                        "title": title,

                        "text": chunk,

                        "url": page_url,

                        "embedding": embedding
                    })

            except Exception as e:

                logger.error(
                    f"Page processing failed: {e}"
                )

        start += limit

    documents = new_documents

    logger.info(
        f"Loaded {len(documents)} chunks"
    )

# =========================================================
# SEARCH
# =========================================================

def semantic_search(query: str):

    if not documents:
        return []

    query_embedding = create_embedding(query)

    query_words = tokenize(query)

    scored_results = []

    for doc in documents:

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

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return scored_results[:TOP_K]

# =========================================================
# GPT ANSWER
# =========================================================

def generate_answer(query: str, results):

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

Answer ONLY using the provided context.

Rules:

- Give direct answers
- Combine facts if needed
- Do not hallucinate
- If answer unavailable say:
  "Information not found in Confluence."

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
                "You answer using enterprise knowledge."
            },

            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# =========================================================
# HTML UI
# =========================================================

def render(content=""):

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
                margin-left: 10px;
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

        <h2>
            Emirates Group Confluence Search ( OpenAI RAG Solution )
        </h2>

        <div style="display:flex; align-items:center;">

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

    </body>

    </html>
    """

# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return render()


@app.on_event("startup")
def startup():

    logger.info("Starting system...")

    fetch_confluence_pages()

    logger.info("System ready")


@app.post("/refresh", response_class=HTMLResponse)
def refresh_confluence():

    try:

        logger.info(
            "Manual refresh started..."
        )

        fetch_confluence_pages()

        logger.info(
            "Manual refresh completed"
        )

        return render(
            """
            <h3>Success</h3>

            <div style="margin-top:15px;">
                Confluence refreshed successfully.
            </div>
            """
        )

    except Exception as e:

        logger.error(
            f"Refresh failed: {e}"
        )

        return render(
            """
            <h3>Error</h3>

            <div style="margin-top:15px;">
                Refresh failed.
            </div>
            """
        )


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):

    try:

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

        # =====================================================
        # IF GPT SAYS INFORMATION NOT FOUND
        # DO NOT SHOW SOURCES
        # =====================================================

        if "Information not found in Confluence" in answer:

            html = f"""

            <h3>Answer</h3>

            <div style="margin-top:15px;">
                {answer}
            </div>

            """

            return render(html)

        # =====================================================
        # SHOW SOURCES ONLY WHEN VALID ANSWER EXISTS
        # =====================================================

        seen = set()

        sources_html = ""

        for score, doc in results:

            if doc["url"] in seen:
                continue

            seen.add(doc["url"])

            sources_html += f"""

            <div class="source">

                <b>{doc['title']}</b>

                <br><br>

                <a href="{doc['url']}"
                   target="_blank">

                    Open Confluence Page

                </a>

            </div>
            """

        html = f"""

        <h3>Answer</h3>

        <div style="margin-top:15px;">
            {answer}
        </div>

        <br>

        <h3>Sources</h3>

        {sources_html}
        """

        return render(html)

    except Exception as e:

        logger.error(f"Search error: {e}")

        return render(
            """
            <h3>Error</h3>

            <div style="margin-top:15px;">
                Internal server error
            </div>
            """
        )


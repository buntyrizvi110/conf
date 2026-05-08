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

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("confluence-rag")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Production Confluence RAG")

# =========================================================
# SETTINGS
# =========================================================

MAX_PAGES = 100
CHUNK_SIZE = 350
TOP_K = 5

EMBEDDING_MODEL = "text-embedding-3-small"

STOPWORDS = {
    "the", "is", "was", "when", "where", "how",
    "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "at", "from"
}

# =========================================================
# MEMORY STORE
# =========================================================

documents = []

# Each document:
# {
#   title,
#   text,
#   url,
#   embedding
# }

# =========================================================
# HELPERS
# =========================================================

def clean_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ")


def tokenize(text: str):
    text = text.lower()
    words = re.findall(r"\w+", text)

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]


def chunk_text(text: str, size=CHUNK_SIZE):

    words = text.split()

    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])

        if len(chunk.strip()) > 100:
            chunks.append(chunk)

    return chunks


def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def create_embedding(text: str):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000]
    )

    return response.data[0].embedding


# =========================================================
# FETCH CONFLUENCE
# =========================================================

def fetch_confluence_pages():

    logger.info("Fetching Confluence pages...")

    if not CONFLUENCE_BASE_URL:
        logger.error("CONFLUENCE_BASE_URL missing")
        return

    try:

        url = (
            f"{CONFLUENCE_BASE_URL}"
            f"/rest/api/content"
            f"?expand=body.storage"
            f"&limit={MAX_PAGES}"
        )

        response = requests.get(
            url,
            auth=(
                CONFLUENCE_USERNAME,
                CONFLUENCE_API_TOKEN
            ),
            timeout=30
        )

        if response.status_code != 200:

            logger.error(
                f"Confluence fetch failed: "
                f"{response.status_code}"
            )

            return

        data = response.json()

        total_chunks = 0

        for page in data.get("results", []):

            try:

                page_id = page.get("id")

                title = page.get("title", "Untitled")

                body_html = page["body"]["storage"]["value"]

                clean_text = clean_html(body_html)

                page_url = (
                    f"{CONFLUENCE_BASE_URL}"
                    f"/pages/viewpage.action?pageId={page_id}"
                )

                chunks = chunk_text(clean_text)

                logger.info(
                    f"Processing: {title} "
                    f"({len(chunks)} chunks)"
                )

                for chunk in chunks:

                    embedding = create_embedding(chunk)

                    documents.append({
                        "title": title,
                        "text": chunk,
                        "url": page_url,
                        "embedding": embedding
                    })

                    total_chunks += 1

            except Exception as page_error:

                logger.error(
                    f"Error processing page: {page_error}"
                )

        logger.info(
            f"Loaded {len(documents)} chunks "
            f"from Confluence"
        )

    except Exception as e:

        logger.error(f"Confluence fetch error: {e}")


# =========================================================
# SEARCH
# =========================================================

def semantic_search(query: str):

    if not documents:
        return []

    query_embedding = create_embedding(query)

    scored_results = []

    for doc in documents:

        similarity = cosine_similarity(
            query_embedding,
            doc["embedding"]
        )

        # keyword bonus
        keyword_bonus = 0

        query_words = tokenize(query)

        title = doc["title"].lower()
        text = doc["text"].lower()

        for word in query_words:

            if word in title:
                keyword_bonus += 0.10

            if word in text:
                keyword_bonus += 0.05

        final_score = similarity + keyword_bonus

        scored_results.append(
            (final_score, doc)
        )

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return scored_results[:TOP_K]


# =========================================================
# GPT ANSWER GENERATION
# =========================================================

def generate_answer(query: str, results):

    context = "\n\n".join([
        f"""
        Title: {doc['title']}

        Content:
        {doc['text']}
        """
        for score, doc in results
    ])

    prompt = f"""
You are a Confluence enterprise assistant.

Answer the user's question ONLY from the provided context.

If the answer is not available,
say:
"Information not found in Confluence."

QUESTION:
{query}

CONTEXT:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer using enterprise knowledge."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================================================
# UI
# =========================================================

def render(answer=""):

    return f"""
    <html>

    <head>

        <title>Confluence RAG Production</title>

        <style>

            body {{
                font-family: Arial;
                margin: 40px;
                background: #f5f5f5;
            }}

            h2 {{
                color: #111;
            }}

            input {{
                width: 450px;
                padding: 12px;
                font-size: 16px;
            }}

            .btn {{
                background: red;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }}

            .box {{
                background: white;
                padding: 20px;
                margin-top: 20px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}

            .source {{
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
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

        <h2>Confluence RAG Final (Production)</h2>

        <form method="post" action="/ask">

            <input
                name="query"
                placeholder="Ask Confluence..."
                required
            />

            <button class="btn">
                ASK
            </button>

        </form>

        <div class="box">

            <b>Answer</b>

            <div style="margin-top:15px;">
                {answer}
            </div>

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

    return render("")


@app.on_event("startup")
def startup():

    logger.info("Starting Confluence RAG")

    fetch_confluence_pages()

    logger.info("System ready")


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):

    try:

        results = semantic_search(query)

        if not results:
            return render(
                "Information not found in Confluence."
            )

        answer = generate_answer(query, results)

        sources_html = ""

        seen = set()

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

        final_output = f"""
        <div>
            {answer}
        </div>

        <br>

        <h3>Sources</h3>

        {sources_html}
        """

        return render(final_output)

    except Exception as e:

        logger.error(f"Search error: {e}")

        return render(
            "Internal server error"
        )


# =========================================================
# LOCAL RUN
# =========================================================

# uvicorn app:app --reload

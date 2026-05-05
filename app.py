import os
import logging
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

# =========================
# SAFE STARTUP CONFIG
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

load_dotenv()

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# OpenAI (safe init)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# =========================
# IN-MEMORY STORE (SAFE)
# =========================
documents = []   # {text, url, title}


# =========================
# UTILITIES
# =========================
def clean_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ")


def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# =========================
# CONFLUENCE FETCH (SAFE)
# =========================
def fetch_confluence_pages():
    """Safe fetch with failure handling"""
    global documents

    if not CONFLUENCE_BASE_URL:
        logger.warning("Confluence not configured")
        return

    try:
        url = f"{CONFLUENCE_BASE_URL}/rest/api/content?expand=body.storage&limit=50"

        res = requests.get(
            url,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
            timeout=10
        )

        if res.status_code != 200:
            logger.error("Confluence fetch failed")
            return

        data = res.json()

        for page in data.get("results", []):
            title = page.get("title", "")
            body = page["body"]["storage"]["value"]
            text = clean_html(body)

            documents.append({
                "title": title,
                "text": text,
                "url": f"{CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page.get('id')}"
            })

        logger.info(f"Loaded {len(documents)} documents")

    except Exception as e:
        logger.error(f"Confluence error: {e}")


# =========================
# SIMPLE SEARCH (PRODUCTION SAFE)
# =========================
def search(query: str):
    query = query.lower()
    results = []

    for doc in documents:
        score = 0

        # partial match (important fix for your issue)
        if query in doc["text"].lower():
            score += 5

        if any(word in doc["text"].lower() for word in query.split()):
            score += 2

        if query in doc["title"].lower():
            score += 3

        if score > 0:
            results.append((score, doc))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:5]


# =========================
# UI
# =========================
def render(answer=""):
    return f"""
    <html>
    <head>
        <title>Confluence RAG Pro</title>
        <style>
            body {{
                font-family: Arial;
                margin: 40px;
                background: #fafafa;
            }}

            .btn {{
                background: red;
                color: white;
                padding: 14px 30px;
                border: none;
                font-size: 18px;
                cursor: pointer;
                border-radius: 6px;
            }}

            input {{
                width: 320px;
                padding: 10px;
            }}

            .box {{
                background: white;
                padding: 15px;
                margin-top: 20px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}

            .footer {{
                position: fixed;
                bottom: 0;
                width: 100%;
                background: #111;
                color: white;
                text-align: center;
                padding: 12px;
                font-size: 14px;
            }}

            a {{
                color: blue;
            }}
        </style>
    </head>

    <body>

        <h2>Confluence RAG Final (Production)</h2>

        <form method="post" action="/ask">
            <input name="query" placeholder="Search Confluence..." />
            <button class="btn">ASK</button>
        </form>

        <div class="box">
            <b>Answer</b>
            <div>{answer}</div>
        </div>

        <div class="footer">
            POC Project By : Syed Abbas Rizvi , STE, Skywards Emirates Airlines
        </div>

    </body>
    </html>
    """


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return render("")


@app.on_event("startup")
def startup():
    logger.info("Loading Confluence data...")
    fetch_confluence_pages()


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):

    results = search(query)

    if not results:
        return render("Not found in Confluence")

    output = ""

    for score, doc in results:
        output += f"""
        <h4>{doc['title']}</h4>
        <p>{doc['text'][:500]}...</p>
        <a href="{doc['url']}" target="_blank">Open Page</a>
        <hr>
        """

    return render(output)

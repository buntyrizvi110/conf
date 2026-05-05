import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# ======================
# SAFE IMPORT HANDLING
# ======================

try:
    import numpy as np
except:
    np = None

try:
    import faiss
except:
    faiss = None  # prevent crash

# ======================
# LOAD ENV
# ======================
load_dotenv()

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ======================
# GLOBAL STORAGE
# ======================
chunks = []
metadata = []
vectors_store = None
index = None


# ======================
# HELPERS
# ======================
def clean(html):
    return BeautifulSoup(html, "html.parser").get_text(" ")


def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# ======================
# SIMPLE SEARCH (NO FAISS CRASH)
# ======================
def simple_search(query):
    results = []
    query_lower = query.lower()

    for i, chunk in enumerate(chunks):
        if query_lower in chunk.lower():
            results.append((chunk, metadata[i]))

    return results[:5]


# ======================
# HTML UI
# ======================
def render(answer=""):
    return f"""
    <html>
    <head>
        <title>Confluence RAG Pro</title>
        <style>
            body {{
                font-family: Arial;
                margin: 40px;
            }}
            .btn {{
                background: red;
                color: white;
                padding: 12px 25px;
                border: none;
                font-size: 16px;
                cursor: pointer;
            }}
            .box {{
                border: 1px solid #ddd;
                padding: 15px;
                margin-top: 20px;
            }}
            .footer {{
                position: fixed;
                bottom: 0;
                width: 100%;
                background: #111;
                color: white;
                text-align: center;
                padding: 10px;
            }}
        </style>
    </head>

    <body>

        <h2>Confluence RAG Pro</h2>

        <form method="post" action="/ask">
            <input name="query" style="width:300px;padding:8px" />
            <button class="btn">ASK</button>
        </form>

        <div class="box">
            <b>Answer</b><br><br>
            {answer}
        </div>

        <div class="footer">
            POC Project By: Syed Abbas Rizvi, STE, Skywards Emirates Airlines
        </div>

    </body>
    </html>
    """


# ======================
# ROUTES
# ======================
@app.get("/", response_class=HTMLResponse)
def home():
    return render()


@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...)):

    results = simple_search(query)

    if not results:
        return render("Not found in Confluence")

    output = ""

    for chunk, meta in results:
        output += f"<p><b>Result:</b> {chunk}</p>"
        if "url" in meta:
            output += f"<p><a href='{meta['url']}' target='_blank'>Page Link</a></p>"
        output += "<hr>"

    return render(output)

import os
import requests
import numpy as np
import faiss
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# ----------------------------
# GLOBAL STORE
# ----------------------------
chunks = []
metadata = []
vectors_store = None
index = None

# ----------------------------
# HELPERS
# ----------------------------
def clean(html):
    return BeautifulSoup(html, "html.parser").get_text(" ")

def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in res.data], dtype="float32")

# ----------------------------
# FETCH CONFLUENCE
# ----------------------------
def fetch_pages(space=None):
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content/search"
    cql = f'space="{space}" AND type=page' if space else "type=page"

    r = requests.get(
        url,
        params={"cql": cql, "limit": 100, "expand": "body.storage"},
        auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
    )

    return r.json().get("results", [])

# ----------------------------
# BUILD INDEX (TITLE INCLUDED)
# ----------------------------
def build_index(space=None):
    global chunks, metadata, vectors_store, index

    pages = fetch_pages(space)

    chunks = []
    metadata = []

    for p in pages:
        text = clean(p.get("body", {}).get("storage", {}).get("value", ""))
        title = p.get("title", "")
        url = f"{CONFLUENCE_BASE_URL}/pages/{p.get('id')}"

        full_text = f"{title}. {text}"

        for c in chunk_text(full_text):
            chunks.append(c)
            metadata.append({"title": title, "url": url})

    if not chunks:
        return

    vectors_store = embed(chunks)
    index = faiss.IndexFlatL2(vectors_store.shape[1])
    index.add(vectors_store)

# ----------------------------
# RETRIEVAL
# ----------------------------
def keyword_retrieve(query):
    q = query.lower()
    results = []

    for i, text in enumerate(chunks):
        if q in text.lower() or q in metadata[i]["title"].lower():
            results.append((chunks[i], metadata[i], vectors_store[i]))

    return results[:10]

def retrieve(query):
    keyword_hits = keyword_retrieve(query)

    q_vec = embed([query])
    _, ids = index.search(q_vec, 20)

    vector_hits = [
        (chunks[i], metadata[i], vectors_store[i])
        for i in ids[0] if i < len(chunks)
    ]

    seen = set()
    combined = []

    for item in keyword_hits + vector_hits:
        if item[0] not in seen:
            seen.add(item[0])
            combined.append(item)

    ctx = [c[0] for c in combined]
    refs = [c[1] for c in combined]
    vecs = [c[2] for c in combined]

    return ctx, refs, vecs

# ----------------------------
# RERANK
# ----------------------------
def keyword_score(query, text):
    return sum(1 for w in query.lower().split() if w in text.lower())

def rerank(query, contexts, refs, vecs):
    q_vec = embed([query])[0]
    q = query.lower()

    scored = []

    for i, c in enumerate(contexts):
        score = (
            0.5 * np.dot(q_vec, vecs[i]) +
            0.2 * keyword_score(query, c) +
            (2 if q in refs[i]["title"].lower() else 0)
        )
        scored.append((score, c, refs[i]))

    scored.sort(reverse=True)
    top = scored[:5]

    return [t[1] for t in top], [t[2] for t in top]

# ----------------------------
# FILTER CONTEXT (FIX)
# ----------------------------
def filter_context(query, contexts, refs):
    q = query.lower()

    filtered_ctx = []
    filtered_refs = []

    for c, r in zip(contexts, refs):
        if q in c.lower() or q in r["title"].lower():
            filtered_ctx.append(c)
            filtered_refs.append(r)

    if not filtered_ctx:
        return contexts[:3], refs[:3]

    return filtered_ctx[:3], filtered_refs[:3]

# ----------------------------
# ANSWER (FIXED PROMPT)
# ----------------------------
def generate_answer(query, context, refs):
    sources = "\n".join(
        [f"{i+1}. {r['title']} - {r['url']}" for i, r in enumerate(refs)]
    )

    prompt = f"""
You are a Confluence assistant.

Rules:
- Answer ONLY from relevant context
- Ignore irrelevant context
- DO NOT say "Not found" if partial answer exists
- Say "Not found in Confluence" ONLY if nothing is found
- Max 4 bullets
- Use [1], [2] only when clearly relevant

Context:
{context}

Sources:
{sources}

Question:
{query}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return res.choices[0].message.content

# ----------------------------
# UNIQUE REFERENCES
# ----------------------------
def unique_refs(refs):
    seen = set()
    out = []
    for r in refs:
        if r["url"] not in seen:
            seen.add(r["url"])
            out.append(r)
    return out

# ----------------------------
# UI
# ----------------------------
def render(ans="", refs=""):
    return f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial; margin: 40px; padding-bottom: 80px; }}

    .box {{
        padding: 20px;
        border: 1px solid #ddd;
        margin-top:20px;
    }}

    .ask-btn {{
        background:red;
        color:white;
        padding:12px 25px;
        font-size:16px;
        border:none;
        border-radius:5px;
        cursor:pointer;
    }}

    input {{
        padding:10px;
        width:300px;
    }}

    .footer {{
        position: fixed;
        bottom: 0;
        width: 100%;
        background: #111;
        color: #fff;
        text-align: center;
        padding: 12px;
    }}
    </style>
    </head>
    <body>

    <h2>Confluence RAG</h2>

    <form method="post" action="/ask">
    <input name="query" placeholder="Ask question">
    <button class="ask-btn">ASK</button>
    </form>

    {f"<div class='box'><b>Answer</b><br>{ans}</div>" if ans else ""}
    {f"<div class='box'><b>References</b><br>{refs}</div>" if refs else ""}

    <div class="footer">
    POC Project By : Syed Abbas Rizvi , STE, Skywards Emirates Airlines
    </div>

    </body>
    </html>
    """

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return render()

@app.post("/ask", response_class=HTMLResponse)
def ask(query: str = Form(...), space: str = None):
    global index

    if index is None:
        build_index(space)

    if index is None:
        return render("No data found.")

    ctx, refs, vecs = retrieve(query)
    ctx, refs = rerank(query, ctx, refs, vecs)

    # 🔥 NEW FILTER
    ctx, refs = filter_context(query, ctx, refs)

    answer = generate_answer(query, "\n\n".join(ctx), refs)

    if "Not found" in answer:
        return render(answer)

    refs = unique_refs(refs)[:3]

    refs_html = ""
    for i, r in enumerate(refs):
        refs_html += f"<a href='{r['url']}' target='_blank'>🔗 [{i+1}] {r['title']}</a>"

    return render(answer, refs_html)

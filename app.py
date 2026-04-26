import os
import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

app = FastAPI(title="Confluence Project Search & Summarizer")


# ---------------------------------------------------------
# Fetch ALL pages in a Confluence space
# ---------------------------------------------------------
def get_all_pages(space_key: str):
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content"
    params = {
        "spaceKey": space_key,
        "type": "page",
        "limit": 200,
        "expand": "body.storage"
    }

    resp = requests.get(url, params=params, auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN))

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch pages: {resp.status_code} - {resp.text}")

    return resp.json().get("results", [])


# ---------------------------------------------------------
# Extract plain text from Confluence HTML
# ---------------------------------------------------------
def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ---------------------------------------------------------
# Multi-word search across all pages
# ---------------------------------------------------------
def search_content(text: str, query: str) -> str | None:
    query_words = query.lower().split()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    scored = []

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for word in query_words if word in line_lower)
        if score > 0:
            scored.append((score, line))

    if not scored:
        return None

    scored.sort(reverse=True, key=lambda x: x[0])
    best_lines = [line for score, line in scored[:15]]
    return "\n".join(best_lines)


# ---------------------------------------------------------
# Simple summary (first few lines)
# ---------------------------------------------------------
def simple_summarize(text: str, max_lines: int = 5) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


# ---------------------------------------------------------
# HTML UI Template (escaped braces)
# ---------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Confluence Project Search</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        input, textarea {{ width: 100%; padding: 10px; margin-top: 10px; }}
        button {{ padding: 10px 20px; margin-top: 20px; }}
        .box {{ margin-top: 30px; padding: 20px; border: 1px solid #ccc; white-space: pre-wrap; }}
    </style>
</head>
<body>

<h2>Confluence Project Search & Summarizer</h2>

<form method="post" action="/process">
    <label>Confluence Space Key (Project)</label>
    <input type="text" name="space_key" value="{space_key}" required>

    <label>Search Query</label>
    <input type="text" name="query" value="{query}" required>

    <button type="submit">Search Project</button>
</form>

{error_block}
{matched_block}
{summary_block}

</body>
</html>
"""


def render_page(space_key="", query="", error=None, matched=None, summary=None):
    return HTML_PAGE.format(
        space_key=space_key,
        query=query,
        error_block=f'<div class="box" style="color:red;"><b>Error:</b> {error}</div>' if error else "",
        matched_block=f'<div class="box"><h3>Matched Content</h3>{matched}</div>' if matched else "",
        summary_block=f'<div class="box"><h3>Summary</h3>{summary}</div>' if summary else "",
    )


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


@app.post("/process", response_class=HTMLResponse)
def process(space_key: str = Form(...), query: str = Form(...)):
    try:
        pages = get_all_pages(space_key)

        if not pages:
            return render_page(
                space_key=space_key,
                query=query,
                error="No pages found in this space."
            )

        combined_text = ""

        for page in pages:
            html = page["body"]["storage"]["value"]
            combined_text += extract_text(html) + "\n"

        matched = search_content(combined_text, query)

        if not matched:
            return render_page(
                space_key=space_key,
                query=query,
                error="No matching content found in the entire project."
            )

        summary = simple_summarize(matched)

        return render_page(
            space_key=space_key,
            query=query,
            matched=matched,
            summary=summary
        )

    except Exception as e:
        return render_page(error=str(e))

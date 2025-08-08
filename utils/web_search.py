from duckduckgo_search import DDGS
from models.llm import get_groq_model

def perform_web_search(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        out = []
        for r in results:
            out.append({
                "title": r.get("title", "No title"),
                "snippet": r.get("body", "No description available."),
                "link": r.get("href", "#")
            })
        return out

def get_completion(prompt):
    try:
        model = get_groq_model()
        resp = model.invoke([{"role": "user", "content": prompt}])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"⚠️ Failed to summarize: {e}"


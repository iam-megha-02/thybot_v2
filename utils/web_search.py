from duckduckgo_search import DDGS
from models.llm import get_groq_model


def perform_web_search(query, max_results=5):
    """
    Perform a web search using DuckDuckGo and return top results.
    Each result includes title, snippet, and link.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        formatted_results = []
        for res in results:
            title = res.get("title", "No title")
            snippet = res.get("body", "No description available.")
            link = res.get("href", "#")
            formatted_results.append(f"{title} - {snippet} ({link})")
        return formatted_results


def get_completion(prompt):
    """
    Summarize the search results using Groq LLM.
    """
    try:
        model = get_groq_model()
        response = model.invoke([{"role": "user", "content": prompt}])
        return response.content
    except Exception as e:
        return f"⚠️ Failed to summarize: {str(e)}"

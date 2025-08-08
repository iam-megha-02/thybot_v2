from langchain_community.tools import DuckDuckGoSearchRun

def get_web_search_tool():
    """
    Initializes and returns the DuckDuckGo search tool.
    This function centralizes the tool's creation, following the project's modular structure.
    """
    return DuckDuckGoSearchRun()

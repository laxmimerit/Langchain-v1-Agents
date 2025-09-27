

@tool
def analyze_market(url: str) -> str:
    """Fetch the full content from the URL using Docling."""

    from ollama import web_fetch
    
    result = web_fetch(url)
    
    return result


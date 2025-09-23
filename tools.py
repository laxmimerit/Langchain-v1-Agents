"""
LangChain Tools Collection
Tools for various agent examples including search, calculation, analysis, etc.
"""


from langchain_core.tools import tool

# ============================================================================
# TOOLS ORDERED BY USAGE IN AGENTS.PY
# ============================================================================

@tool
def simple_search(query: str) -> str:
    """Search for information."""
    return f"Found results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web search results for '{query}': Found comprehensive information"


@tool
def analyze_data(data: str) -> str:
    """Analyze provided data."""
    return f"Analysis of '{data}': Shows interesting patterns and trends"


@tool
def research_tool(topic: str) -> str:
    """Research a complex topic."""
    return f"Detailed research on {topic}: Multiple perspectives and data points"


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def helper_tool(request: str) -> str:
    """General helper tool."""
    return f"Helping with: {request}"


@tool
def extract_contact(text: str) -> str:
    """Extract contact information from text."""
    return f"Analyzing text for contact information: {text}"


@tool
def remember_preference(key: str, value: str) -> str:
    """Remember a user preference."""
    return f"I'll remember that you prefer {key}: {value}"


@tool
def get_personalized_help(topic: str) -> str:
    """Provide help based on user preferences."""
    return f"Providing personalized help for {topic} based on your preferences"


@tool
def sensitive_info_tool(query: str) -> str:
    """Tool that might return sensitive information."""
    if "password" in query.lower():
        return "CONFIDENTIAL: The password is admin123"
    return f"Public information about: {query}"


@tool
def slow_research(topic: str) -> str:
    """Simulate a research operation that takes time."""
    import time
    time.sleep(1)  # Simulate processing time
    return f"Completed comprehensive research on: {topic}"


@tool
def analyze_market(market: str) -> str:
    """Simulate market analysis."""
    import time
    time.sleep(0.8)
    return f"Market analysis complete for {market}: Showing positive trends"


@tool
def web_search_production(query: str) -> str:
    """Search the web for current information."""
    # In production, this would call actual search APIs
    return f"🔍 Web search results for '{query}': Found recent articles, data, and expert opinions"


@tool
def calculate_advanced(expression: str) -> str:
    """Perform advanced mathematical calculations safely."""
    try:
        # Production-safe evaluation (whitelist approach)
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"🧮 Calculation: {expression} = {result}"
        else:
            return "❌ Invalid expression: only basic math operations (+, -, *, /, parentheses) allowed"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"


@tool
def analyze_text(text: str) -> str:
    """Analyze text for insights, sentiment, and key information."""
    word_count = len(text.split())
    char_count = len(text)
    
    # Basic sentiment analysis (in production, use proper NLP libraries)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
    
    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    if positive_score > negative_score:
        sentiment = "Positive"
    elif negative_score > positive_score:
        sentiment = "Negative" 
    else:
        sentiment = "Neutral"
    
    return f"📊 Text Analysis: {word_count} words, {char_count} characters. Sentiment: {sentiment} ({positive_score} positive, {negative_score} negative indicators)"
"""
Simplified LangChain Tools Collection
Only tools that are actually used in agents.py, in the correct order.
"""
import ast
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# DuckDuckGo search integration
from ddgs import DDGS


# Configuration
class ToolConfig:
    """Configuration for tools with API keys and settings."""
    def __init__(self):
        self.user_preferences = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "langchain_tools"
        self.temp_dir.mkdir(exist_ok=True)

config = ToolConfig()


# ============================================================================
# TOOLS IN ORDER OF USAGE IN AGENTS.PY
# ============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Web search using DuckDuckGo with configurable result count."""
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                max_results=num_results,
                region="us-en",
                timelimit="d"
            ))
        
        if not results:
            return f"🔍 No results found for '{query}'"
        
        formatted_results = [f"🔍 Search Results for '{query}':\n"]
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description available')
            href = result.get('href', '')
            formatted_results.append(f"{i}. **{title}**\n   {body}\n   🔗 {href}")
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"❌ Search error: {str(e)}"


@tool
def analyze_data(data: list, operation: str) -> str:
    """
    Dynamic pandas data analysis tool.
    
    Args:
        data: List of dictionaries to convert to DataFrame
        operation: Pandas operation string (e.g., 'df.describe()', 'df.groupby("col").sum()')
    
    Returns:
        String representation of analysis result or error message
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        return f"❌ Missing dependency: {e}"
    
    try:
        if not data or not isinstance(data, list):
            return "❌ Data must be a non-empty list"
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return f"❌ Failed to create DataFrame: {str(e)}"
        
        if df.empty:
            return "❌ Empty DataFrame created"
        
        # Default operation
        if not operation or not isinstance(operation, str):
            operation = "df.describe()"
        
        # Basic validation - operation should reference 'df'
        if 'df' not in operation:
            return "❌ Operation must reference 'df' (the DataFrame)"
        
        # Execute operation with restricted namespace
        namespace = {'df': df, 'pd': pd, 'np': np}
        result = eval(operation, {"__builtins__": {}}, namespace)
        
        # Format result
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_string()
        else:
            return str(result)
            
    except SyntaxError:
        return "❌ Invalid operation syntax"
    except NameError as e:
        return f"❌ Invalid reference: {e}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions using AST parsing.
    
    Supports basic arithmetic, mathematical functions, and constants.
    Uses Abstract Syntax Tree (AST) validation for security.
    
    Args:
        expression: Mathematical expression string to evaluate
    
    Supported Operations:
        - Arithmetic: +, -, *, /, %, **
        - Functions: abs, round, min, max, sum, pow, sqrt
        - Trigonometry: sin, cos, tan
        - Logarithms: log
        - Constants: pi, e
    
    Examples:
        calculate("2 + 3 * 4")           → "2 + 3 * 4 = 14"
        calculate("sqrt(16) + pow(2,3)") → "sqrt(16) + pow(2,3) = 12.0"
        calculate("sin(pi/2)")           → "sin(pi/2) = 1.0"
        calculate("max(1,2,3) * 5")      → "max(1,2,3) * 5 = 15"
    
    Returns:
        String with calculation result or error message
    """
    try:
        import ast
        import math
        
        # Parse the expression into an AST
        node = ast.parse(expression, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression, ast.Constant, ast.Num, ast.BinOp,
            ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Name,
            ast.Load, ast.Call, ast.keyword
        }
        
        # Define allowed functions
        allowed_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'sqrt': lambda x: x ** 0.5,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'pi': math.pi, 'e': math.e
        }
        
        # Check if all nodes are allowed
        for node_item in ast.walk(node):
            if type(node_item) not in allowed_nodes:
                return f"Error: Unsupported operation: {type(node_item).__name__}"
        
        # Evaluate safely
        result = eval(compile(node, '<string>', 'eval'), 
                     {"__builtins__": {}}, allowed_functions)
        
        return f"{expression} = {result}"
    
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def helper_tool(request: str) -> str:
    """General helper tool for various requests."""
    return f"I can help you with: {request}. What specific assistance do you need?"


@tool
def save_user_preference(key: str, value: str) -> str:
    """Save a user preference."""
    try:
        config.user_preferences[key] = value
        
        # Save to file for persistence
        prefs_file = config.temp_dir / "user_preferences.json"
        with open(prefs_file, 'w') as f:
            json.dump(config.user_preferences, f, indent=2)
        
        return f"✅ Saved preference: {key} = {value}"
    except Exception as e:
        return f"❌ Error saving preference: {str(e)}"


@tool
def get_user_preference(key: str) -> str:
    """Retrieve a saved user preference."""
    try:
        # Load from file if not in memory
        if not config.user_preferences:
            prefs_file = config.temp_dir / "user_preferences.json"
            if prefs_file.exists():
                with open(prefs_file, 'r') as f:
                    config.user_preferences = json.load(f)
        
        if key in config.user_preferences:
            return f"📋 {key}: {config.user_preferences[key]}"
        else:
            return f"❌ No preference found for: {key}"
    except Exception as e:
        return f"❌ Error retrieving preference: {str(e)}"


@tool
def sensitive_info_tool(query: str) -> str:
    """Tool that demonstrates security handling."""
    sensitive_keywords = ["password", "secret", "confidential", "private"]
    if any(word in query.lower() for word in sensitive_keywords):
        return "I cannot provide sensitive or confidential information. Please ask for public information only."
    return f"Public information about: {query}"

## replace with ddgs news search
@tool
def latest_news(topic: str) -> str:
    """Simulate a research operation that takes time."""
    from ddgs import DDGS
    import time
    time.sleep(1)  # Simulate processing time

    response = DDGS.news(query=topic, region='us-en', timelimit='d', max_results=5)
    if not response:
        return f"No news found for topic: {topic}"
    
    return response
@tool
def analyze_market(url: str) -> str:
    """Fetch the full content from the URL using Docling."""

    from ollama import web_fetch
    
    result = web_fetch(url)
    
    return result

@tool
def analyze_text(text: str) -> str:
    """Basic text analysis including word count and simple sentiment."""
    if not text.strip():
        return "❌ Empty text provided"
    
    # Basic statistics
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Simple sentiment analysis
    positive_words = {
        'excellent', 'amazing', 'wonderful', 'great', 'good', 'fantastic',
        'awesome', 'brilliant', 'outstanding', 'beautiful', 'happy', 'love'
    }
    
    negative_words = {
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate',
        'disappointing', 'failure', 'wrong', 'sad', 'angry'
    }
    
    text_words = set(word.lower().strip('.,!?;:"()[]') for word in words)
    positive_matches = len(text_words.intersection(positive_words))
    negative_matches = len(text_words.intersection(negative_words))
    
    sentiment_score = positive_matches - negative_matches
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return f"""📊 Text Analysis Results:
📝 Words: {word_count} | Characters: {char_count} | Sentences: {sentence_count}
📖 Avg words/sentence: {avg_words_per_sentence:.1f}
💭 Sentiment: {sentiment} (Score: {sentiment_score:+d})"""


@tool
def extract_contact(text: str) -> str:
    """Extract contact information from text."""
    contacts = {}
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contacts['emails'] = list(set(emails))
    
    # Phone pattern
    phone_patterns = [
        r'\+?1?[-.]?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',  # US format
        r'\+?\d{1,3}[-.]?\d{3,4}[-.]?\d{3,4}[-.]?\d{3,4}',  # International
    ]
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    if phones:
        contacts['phones'] = list(set(phones))
    
    # URL pattern
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    if urls:
        contacts['urls'] = list(set(urls))
    
    if not contacts:
        return "📭 No contact information found"
    
    result = ["📞 Contact Information Found:"]
    for contact_type, items in contacts.items():
        result.append(f"  {contact_type.title()}: {', '.join(items)}")
    
    return "\n".join(result)
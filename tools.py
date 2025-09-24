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
def simple_search(query: str) -> str:
    """Simple web search using DuckDuckGo."""
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                max_results=3,
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
def analyze_data(data: str) -> str:
    """Basic data analysis with summary statistics."""
    try:
        lines = [line.strip() for line in data.split('\n') if line.strip()]
        
        if not lines:
            return "❌ No data to analyze"
        
        # Basic statistics
        total_lines = len(lines)
        total_chars = len(data)
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        longest_line = max(len(line) for line in lines)
        shortest_line = min(len(line) for line in lines)
        
        return f"""📊 Data Summary:
📝 Total lines: {total_lines}
📄 Total characters: {total_chars}
🔤 Average line length: {avg_line_length:.1f} characters
📏 Longest line: {longest_line} characters
📐 Shortest line: {shortest_line} characters"""
    
    except Exception as e:
        return f"❌ Analysis error: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
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
        import math
        allowed_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'sqrt': lambda x: x ** 0.5,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'pi': math.pi, 'e': math.e
        }
        
        # Check if all nodes are allowed
        for node_item in ast.walk(node):
            if type(node_item) not in allowed_nodes:
                return f"❌ Unsupported operation: {type(node_item).__name__}"
        
        # Evaluate safely
        result = eval(compile(node, '<string>', 'eval'), 
                     {"__builtins__": {}}, allowed_functions)
        
        return f"🧮 {expression} = {result}"
    
    except SyntaxError:
        return "❌ Invalid mathematical expression"
    except ZeroDivisionError:
        return "❌ Division by zero error"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"


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


@tool
def slow_research(topic: str) -> str:
    """Simulate a research operation that takes time."""
    import time
    time.sleep(1)  # Simulate processing time
    return f"Completed comprehensive research on: {topic}. Found multiple data points and analysis perspectives."


@tool
def analyze_market(market: str) -> str:
    """Simulate market analysis with processing time."""
    import time
    time.sleep(0.8)  # Simulate analysis time
    return f"Market analysis complete for {market}: Current trends show positive growth indicators and strong market fundamentals."


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
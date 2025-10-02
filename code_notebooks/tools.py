"""
Simplified LangChain Tools Collection
Only tools that are actually used in agents.py, in the correct order.
"""
from dotenv import load_dotenv
load_dotenv('../.env')  # Load environment variables from .env file

import json
import re
import tempfile
from pathlib import Path

from langchain_core.tools import tool

# DuckDuckGo search integration
from ddgs import DDGS

# ============================================================================
# TOOLS IN ORDER OF USAGE IN AGENTS.PY
# ============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted search results with titles, descriptions, and URLs
    """
    
    try:
        results = list(DDGS().text(query=query,
                                   max_results=num_results,
                                   region="us-en",
                                   timelimit="d",
                                   backend="google, bing, brave, yahoo, wikipedia, duckduckgo"))
        
        if not results:
            return f"No results found for '{query}'"
        
        formatted_results = [f"Search Results for '{query}':\n"]
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description available')
            href = result.get('href', '')
            formatted_results.append(f"{i}. **{title}**\n   {body}\n   {href}")
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"



@tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions using AST parsing.
    
    Args:
        expression: Mathematical expression string to evaluate
    
    Returns:
        String with calculation result or error message
        
    Supports arithmetic, math functions (sin, cos, sqrt, log), and constants (pi, e)
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
        return f"Missing dependency: {e}"
    
    try:
        if not data or not isinstance(data, list):
            return "Data must be a non-empty list"
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return f"Failed to create DataFrame: {str(e)}"
        
        if df.empty:
            return "Empty DataFrame created"
        
        # Default operation
        if not operation or not isinstance(operation, str):
            operation = "df.describe()"
        
        # Basic validation - operation should reference 'df'
        if 'df' not in operation:
            return "Operation must reference 'df' (the DataFrame)"
        
        # Execute operation with restricted namespace
        namespace = {'df': df, 'pd': pd, 'np': np}
        result = eval(operation, {"__builtins__": {}}, namespace)
        
        # Format result
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_string()
        else:
            return str(result)
            
    except SyntaxError:
        return "Invalid operation syntax"
    except NameError as e:
        return f"Invalid reference: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

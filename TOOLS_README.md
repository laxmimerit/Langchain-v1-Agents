# Production-Ready LangChain Tools

This collection contains real, functional LangChain tools that can be used in production applications.

## 🔧 Available Tools

### 1. Web Search  
- **Function**: `web_search(query: str, num_results: int = 5)`
- **Description**: Real web search using DuckDuckGo's search API via duckduckgo-search package
- **Features**: No API key required, returns structured results with titles, snippets, and links
- **Example**: `web_search("Python programming", num_results=3)`

### 1b. News Search
- **Function**: `news_search(query: str, num_results: int = 5)` 
- **Description**: Real news search using DuckDuckGo's news backend
- **Features**: Recent news articles with dates, sources, and descriptions
- **Example**: `news_search("AI technology", num_results=3)`

### 2. Safe Calculator
- **Function**: `calculate(expression: str)`
- **Description**: Safely evaluate mathematical expressions using AST parsing
- **Features**: Supports basic math operations and functions, blocks unsafe code execution
- **Example**: `calculate("2 + 3 * sqrt(16)")` → `2 + 3 * sqrt(16) = 14.0`

### 3. Text Analysis
- **Function**: `analyze_text(text: str)`
- **Description**: Comprehensive text analysis including word count, sentiment, and readability
- **Features**: Word/character count, sentiment analysis, readability assessment
- **Example**: Analyzes sentiment, readability level, and basic statistics

### 4. File Operations
- **Function**: `file_read(file_path: str)`, `file_write(file_path: str, content: str)`
- **Description**: Secure file reading and writing operations
- **Features**: Path validation, security checks, encoding support
- **Security**: Restricts access to current directory tree only

### 5. Directory Listing
- **Function**: `list_files(directory: str = ".", pattern: str = "*")`
- **Description**: List files with pattern matching and metadata
- **Features**: File size, modification dates, pattern filtering
- **Example**: `list_files(".", "*.py")` lists all Python files

### 6. Contact Information Extraction
- **Function**: `extract_contact_info(text: str)`
- **Description**: Extract emails, phone numbers, URLs, and social media handles
- **Features**: Supports multiple formats, deduplication
- **Patterns**: Emails, phone numbers, URLs, Twitter handles, LinkedIn/GitHub profiles

### 7. User Preferences
- **Functions**: `save_user_preference(key: str, value: str)`, `get_user_preference(key: str)`
- **Description**: Persistent user preference storage
- **Features**: JSON file storage, automatic loading/saving
- **Example**: Save and retrieve user settings

### 8. System Information
- **Function**: `system_info()`
- **Description**: Get comprehensive system and environment information
- **Features**: Platform details, Python version, directory info, optional hardware stats
- **Requirements**: Install `psutil` for extended hardware information

### 9. Safe Command Execution
- **Function**: `run_command(command: str, timeout: int = 30)`
- **Description**: Execute whitelisted system commands safely
- **Features**: Command whitelist, timeout protection, output capture
- **Allowed Commands**: `ls`, `dir`, `pwd`, `git`, `python`, `pip`, `npm`, etc.

### 10. Data Analysis
- **Function**: `data_analysis(data: str, analysis_type: str = "summary")`
- **Description**: Multi-type data analysis capabilities
- **Types**: 
  - `summary`: Line counts, character statistics
  - `numeric`: Extract and analyze numbers
  - `frequency`: Word frequency analysis

## 🛡️ Security Features

1. **AST-based Math Evaluation**: No unsafe `eval()` usage
2. **Path Validation**: File operations restricted to current directory
3. **Command Whitelist**: Only safe commands allowed
4. **Input Validation**: Comprehensive error handling
5. **No Hardcoded Secrets**: Removed all sensitive information

## 🚀 Usage Example

```python
from tools import web_search, calculate, analyze_text, file_read

# Search the web
result = web_search.invoke({"query": "LangChain tutorials"})
print(result)

# Safe calculations
math_result = calculate.invoke({"expression": "2**10 + sqrt(144)"})
print(math_result)  # 2**10 + sqrt(144) = 1036.0

# Text analysis
text_result = analyze_text.invoke({"text": "This is amazing! Great work!"})
print(text_result)

# File operations
file_content = file_read.invoke({"file_path": "README.md"})
print(file_content)
```

## 📦 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Optional for extended system info:
```bash
pip install psutil
```

## ⚠️ Important Notes

1. **LangChain Version**: Use `.invoke()` method instead of direct calls to avoid deprecation warnings
2. **Unicode Handling**: Some console environments may have Unicode display issues with emojis
3. **Network Access**: Web search requires internet connectivity
4. **File Security**: All file operations are restricted to the current directory tree for security

## 🔄 Migration from Mock Tools

The original mock tools have been completely replaced with production-ready implementations:

- ✅ Security vulnerabilities fixed
- ✅ Duplicate tools removed
- ✅ Real API integrations added
- ✅ Comprehensive error handling
- ✅ Input validation and sanitization
- ✅ Configuration management

These tools are now ready for use in real LangChain agent applications!
## Prerequisites

Make sure you have the required packages installed and Ollama running:

```bash
pip install --pre -U langchain langchain-community langchain-core langgraph
pip install ddgs python-dotenv
```

If you want to use UV. These are one-time setup only. 
```bash
pip install uv
uv init
uv add ddgs python-dotenv
uv add --prerelease=allow -U langchain langchain-community langchain-core langgraph

```

Local LLM Serving with Ollama
```bash
ollama pull qwen3
ollama serve
```

## Available Tools (10 Tools)

**How Tools Connect to Agents:** Each tool is decorated with `@tool` and automatically becomes available to agents. Agents can call multiple tools in sequence and combine their outputs for complex tasks.

### 1. 🔍 web_search
DuckDuckGo integration for real-time web searches with multiple backends

### 2. 📊 analyze_data
Dynamic pandas operations on DataFrames with safe evaluation

### 3. 🧮 calculate
Safe mathematical expression evaluation with AST parsing

### 4. 🆘 helper_tool
General assistance for various user requests and guidance

### 5. 💾 save_user_preference
Persistent storage of user preferences with JSON serialization

### 6. 📖 get_user_preference
Retrieval of saved user preferences for personalization

### 7. 🔒 sensitive_info_tool
Secure information handling with sensitive content filtering

### 8. 📰 latest_news
Real-time news aggregation using DuckDuckGo news API

### 9. 📝 analyze_text
Text statistics and basic sentiment analysis with word counting

### 10. 📞 extract_contact
Regex-based extraction of emails, phones, and URLs from text
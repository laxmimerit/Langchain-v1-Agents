"""
BASIC AGENT WITH MODEL STRING

This is the simplest way to create a LangChain agent. Instead of explicitly
creating a model instance, you can use a model string that follows the format
"provider:model_name". LangChain will automatically create the model instance
for you.

Key Benefits:
- Minimal code required
- Quick setup for prototyping
- Automatic model inference (e.g., "qwen3" becomes "ollama:qwen3")

Use Cases:
- Getting started with agents
- Simple proof-of-concepts
- When you don't need custom model parameters

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import tools

# Load environment variables from .env file
load_dotenv()

def example_1_basic_agent_with_model_string():
    """
    BASIC AGENT WITH MODEL STRING
    
    This is the simplest way to create a LangChain agent. Instead of explicitly
    creating a model instance, you can use a model string that follows the format
    "provider:model_name". LangChain will automatically create the model instance
    for you.
    
    Key Benefits:
    - Minimal code required
    - Quick setup for prototyping
    - Automatic model inference (e.g., "qwen3" becomes "ollama:qwen3")
    
    Use Cases:
    - Getting started with agents
    - Simple proof-of-concepts
    - When you don't need custom model parameters
    """
    print("\n=== Example 1: Basic Agent with Model String ===")
    print("📝 This example shows the simplest way to create an agent using a model string.")
    print("   The agent will use Qwen3 through Ollama with default parameters.")
    
    
    # Using Ollama provider with Qwen3 model - simplest approach
    agent = create_agent(
        "ollama:qwen3",  # Model string format: provider:model
        tools=[tools.web_search]
    )
    
    # Simplified message format as requested
    result = agent.invoke({
        "messages": "Search for Python tutorials"
    })
    
    print(f"✅ Response: {result['messages'][-1].content}")
    print("💡 Notice: LangChain automatically created ChatOllama instance from string")
    return agent

if __name__ == "__main__":
    example_1_basic_agent_with_model_string()

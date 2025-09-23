"""
AGENT WITH EXPLICIT MODEL INSTANCE

This approach gives you full control over the model configuration. You create
the ChatOllama instance yourself and configure all parameters explicitly.

Key Benefits:
- Complete control over model parameters
- Custom temperature, token limits, timeouts
- Better for production environments
- Easier to debug model-specific issues

Model Parameters Explained:
- temperature: Controls randomness (0.0 = deterministic, 1.0 = very creative)
- num_predict: Maximum tokens to generate (similar to max_tokens in OpenAI)
- timeout: How long to wait for model response

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

def example_2_agent_with_model_instance():
    """
    AGENT WITH EXPLICIT MODEL INSTANCE
    
    This approach gives you full control over the model configuration. You create
    the ChatOllama instance yourself and configure all parameters explicitly.
    
    Key Benefits:
    - Complete control over model parameters
    - Custom temperature, token limits, timeouts
    - Better for production environments
    - Easier to debug model-specific issues
    
    Model Parameters Explained:
    - temperature: Controls randomness (0.0 = deterministic, 1.0 = very creative)
    - num_predict: Maximum tokens to generate (similar to max_tokens in OpenAI)
    - timeout: How long to wait for model response
    """
    print("\n=== Example 2: Agent with Explicit Model Instance ===")
    print("🔧 This example shows how to create an agent with custom model configuration.")
    print("   You have full control over temperature, token limits, and other parameters.")
    
    
    # Create model instance with custom parameters
    model = ChatOllama(
        model="qwen3",
        temperature=0.1,      # Low temperature for consistent math results
        num_predict=1000,     # Maximum tokens to generate
        timeout=30            # 30 second timeout
    )
    
    agent = create_agent(model, tools=[tools.calculate])
    
    # Simplified message format
    result = agent.invoke({
        "messages": "What's 15 * 27 + 100?"
    })
    
    print(f"✅ Response: {result['messages'][-1].content}")
    print("💡 Notice: Low temperature ensures consistent mathematical calculations")
    return agent

if __name__ == "__main__":
    example_2_agent_with_model_instance()

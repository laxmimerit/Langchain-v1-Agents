"""
DIFFERENT APPROACHES TO TOOL CONFIGURATION

LangChain offers multiple ways to set up tools for your agents:

Method 1: List of Tools (Simple)
- Pass tools directly as a list
- LangChain creates ToolNode automatically
- Good for basic use cases

Method 2: ToolNode with Configuration (Advanced)
- Create ToolNode explicitly with custom settings
- Add error handling, custom messages
- Better control over tool execution

Tool Error Handling:
- Without error handling: Errors crash the agent
- With error handling: Errors become feedback for the model
- Model can retry or adjust approach based on error messages

Best Practices:
- Use Method 1 for prototyping
- Use Method 2 for production systems
- Always include error handling for robust agents

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, ToolNode
import tools

# Load environment variables from .env file
load_dotenv()

def example_5_tool_configurations():
    """
    DIFFERENT APPROACHES TO TOOL CONFIGURATION
    ...existing docstring...
    """
    print("\n=== Example 5: Different Tool Configuration Methods ===")
    print("🛠️  This example shows two ways to configure tools and error handling.")
    print("   Method 1: Simple list | Method 2: Advanced ToolNode with error handling")

    model = ChatOllama(model="qwen3")

    # Method 1: Pass list of tools (simple approach)
    print("\n🔧 Method 1: Simple list of tools")
    agent1 = create_agent(model, tools=[tools.search, tools.calculate])

    # Method 2: Use ToolNode with error handling (advanced approach)
    print("🔧 Method 2: ToolNode with custom error handling")
    tool_node = ToolNode(
        tools=[tools.search, tools.calculate],
        handle_tool_errors="Please check your input and try again. Error details will help you correct the issue."
    )
    agent2 = create_agent(model, tools=tool_node)

    # Test both agents with the same query
    test_query = "Search for Python tutorials and calculate 10 times 5"

    print(f"\n📝 Testing both agents with: '{test_query}'")

    result1 = agent1.invoke({"messages": test_query})
    print(f"✅ Agent 1 (simple): {result1['messages'][-1].content}")

    result2 = agent2.invoke({"messages": test_query})
    print(f"✅ Agent 2 (advanced): {result2['messages'][-1].content}")

    print("💡 Both agents work the same for valid inputs, but Agent 2 handles errors better")
    return agent1, agent2

if __name__ == "__main__":
    example_5_tool_configurations()

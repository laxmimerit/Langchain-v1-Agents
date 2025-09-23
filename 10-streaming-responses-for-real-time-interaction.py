"""
STREAMING RESPONSES FOR REAL-TIME INTERACTION

Streaming allows you to see agent responses as they're generated, rather
than waiting for the complete response. This is especially important for:

Benefits:
- Better user experience (no waiting for long responses)
- Real-time feedback on agent progress
- Ability to cancel long-running operations
- Transparent view of agent reasoning process

Stream Modes:
- "values": Get complete state at each step
- "updates": Get only changes at each step

What You Can Stream:
- Agent thoughts and reasoning
- Tool calls and their names
- Partial responses as they're generated
- Tool results and observations

Use Cases:
- Interactive chat interfaces
- Long research tasks
- Multi-step reasoning problems
- Real-time status updates

Implementation Notes:
- Use .stream() instead of .invoke()
- Process chunks as they arrive
- Handle both text content and tool calls

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
import tools

# Load environment variables from .env file
load_dotenv()

def example_10_streaming():
    """
    STREAMING RESPONSES FOR REAL-TIME INTERACTION
    
    Streaming allows you to see agent responses as they're generated, rather
    than waiting for the complete response. This is especially important for:
    
    Benefits:
    - Better user experience (no waiting for long responses)
    - Real-time feedback on agent progress
    - Ability to cancel long-running operations
    - Transparent view of agent reasoning process
    
    Stream Modes:
    - "values": Get complete state at each step
    - "updates": Get only changes at each step
    
    What You Can Stream:
    - Agent thoughts and reasoning
    - Tool calls and their names
    - Partial responses as they're generated
    - Tool results and observations
    
    Use Cases:
    - Interactive chat interfaces
    - Long research tasks
    - Multi-step reasoning problems
    - Real-time status updates
    
    Implementation Notes:
    - Use .stream() instead of .invoke()
    - Process chunks as they arrive
    - Handle both text content and tool calls
    """
    print("\n=== Example 10: Streaming Responses for Real-Time Interaction ===")
    print("\ud83d\udce1 This example demonstrates real-time streaming of agent responses.")
    print("   You can see tool calls, reasoning, and responses as they happen.")

    model = ChatOllama(model="qwen3")
    agent = create_agent(model, tools=[tools.slow_research, tools.analyze_market])

    query = "Research AI market trends and analyze the technology sector"
    print(f"\ud83d\udcdd Query: {query}")
    print("\ud83d\udd04 Streaming response:\n")

    # Stream the response in real-time
    for i, chunk in enumerate(agent.stream({
        "messages": query
    }, stream_mode="values")):
        latest_message = chunk["messages"][-1]
        # Handle different types of messages in the stream
        if hasattr(latest_message, 'content') and latest_message.content:
            print(f"  \ud83d\udcac Agent [{i}]: {latest_message.content}")
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in latest_message.tool_calls]
            print(f"  \ud83d\udd27 Tool Call [{i}]: {', '.join(tool_names)}")
        elif isinstance(latest_message, ToolMessage):
            print(f"  \ud83d\udcca Tool Result [{i}]: {latest_message.content}")

    print("\n\u2705 Streaming completed!")
    print("\ud83d\udca1 Notice how you could see each step of the agent's work in real-time")
    return agent

if __name__ == "__main__":
    example_10_streaming()

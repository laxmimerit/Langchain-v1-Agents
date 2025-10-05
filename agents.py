"""
Langchain v1 Agents - Code from Notebooks
All code extracted from the notebooks, organized by notebook number.
"""

# ============================================================================
# NOTEBOOK 01: Basic Agent with Model String
# ============================================================================

# Import required modules
from langchain.agents import create_agent
from code_notebooks import tools

# Create agent with model string - simplest approach
agent = create_agent(
    "ollama:qwen3:30b",  # Model string format: provider:model
    tools=[tools.web_search]
)

print("Agent created successfully with model string!")
print("LangChain automatically created ChatOllama instance from string")

# Test the agent with a search query
result = agent.invoke({
    "messages": "Search for python tutorials"
})

print(result)


# ============================================================================
# NOTEBOOK 02: Agent with Explicit Model Instance
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from code_notebooks import tools

# Create model instance with custom parameters
model = ChatOllama(
    model="qwen3",
    base_url="http://localhost:11434",
    temperature=0,      # Low temperature for consistent math results
)

# Create agent with explicit model instance
agent = create_agent(model, tools=[tools.calculate])

print("Agent created with explicit model instance!")
print("Full control over temperature, token limits, and other parameters")

# Test the agent with a mathematical calculation
result = agent.invoke({
    "messages": "What's 15 * 27 + 100?"
})

print(f"Response: {result['messages'][-1].content}")

result['messages'][-1].pretty_print()


# Experimenting with Different Settings

question = "Explain what 2 + 2 equals and show your reasoning"

# Configuration 1: Conservative/Deterministic - for consistent, reliable outputs
print("=== Configuration 1: Conservative/Deterministic ===")
llm1 = ChatOllama(
    model="qwen3",
    temperature=0,           # Low randomness for consistency
    top_p=1,                # Focus on most probable tokens
    repeat_penalty=1.1,       # Slight repetition penalty
    num_predict=500,          # Reasonable output length
    num_ctx=4096,            # Standard context window
    seed=42                   # Reproducible results
)
agent = create_agent(llm1, tools=[tools.calculate])
result = agent.invoke({"messages": question})
print(f"Conservative output: {result['messages'][-1].content}")
print()

print("=== Configuration 2: Balanced/Production ===")
llm = ChatOllama(
    model="qwen3",
    temperature=2,          # Moderate creativity
    top_k=2000,                # Standard token selection
    repeat_penalty=1.15,     # Moderate repetition control
    repeat_last_n=64,        # Check last 64 tokens for repetition
    num_predict=1000,         # Short responses allowed | check output
    num_ctx=8192,           # Larger context for complex tasks
    keep_alive="5m",         # Keep model loaded for performance
    num_thread=4             # Optimize for multi-core processing
)

agent = create_agent(llm, tools=[tools.calculate])
result = agent.invoke({"messages": question})
print(result['messages'][-1].content)
print()

print("=== Configuration 2: Balanced/Production ===")
llm = ChatOllama(
    model="qwen3",
    temperature=2,          # Moderate creativity
    top_k=2000,                # Standard token selection
    repeat_penalty=1.15,     # Moderate repetition control
    repeat_last_n=64,        # Check last 64 tokens for repetition
    num_predict=1000,         # Short responses allowed | check output
    num_ctx=8192,           # Larger context for complex tasks
    keep_alive="5m",         # Keep model loaded for performance
    num_thread=4,             # Optimize for multi-core processing
    reasoning=True
)

agent = create_agent(llm, tools=[tools.calculate])
result = agent.invoke({"messages": question})
print('question:', question)
print(result['messages'][-1].content)
print()


# ============================================================================
# NOTEBOOK 03: Dynamic Model Selection (Qwen3 → GPT-OSS)
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.runtime import Runtime
import tools

# Define tool list for both models
tool_list = [tools.web_search, tools.analyze_data]

def select_model(state: AgentState, runtime: Runtime) -> ChatOllama:
    """Choose between Qwen3 and GPT-OSS based on conversation length."""
    messages = state["messages"]
    message_count = len(messages)

    if message_count < 10:
        print(f"  Using Qwen3 for {message_count} messages (efficient)")
        return ChatOllama(model="qwen3", temperature=0.1).bind_tools(tool_list)
    else:
        print(f"  Switching to GPT-OSS for {message_count} messages (advanced)")
        return ChatOllama(model="gpt-oss", temperature=0.0, num_predict=2000).bind_tools(tool_list)

print("Model selection function defined!")
print("Logic: < 10 messages = Qwen3, >= 10 messages = GPT-OSS")

# Create agent with dynamic model selection
agent = create_agent(select_model, tools=tool_list)

print("Dynamic agent created successfully!")
print("This agent will automatically switch models based on conversation complexity")

print("=== Testing Short Conversation (Should Use Qwen3) ===")

result1 = agent.invoke({
    "messages": "Search for AI news"
})

print(f"\nShort conversation result: {result1['messages'][-1].content}")

print("=== Testing Long Conversation (Should Use GPT-OSS) ===")

# Simulate conversation state with many messages
long_messages = "This is message number 12 in our conversation. I need complex analysis."

# Create a new agent instance for this test
agent_with_history = create_agent(select_model, tools=tool_list)

result2 = agent_with_history.invoke({
    "messages": [f"Message {i}" for i in range(12)] + [long_messages]
})

print("\nLong conversation triggered model switch to GPT-OSS")


def demo_conversation_progression():
    """Demonstrate how the agent switches models as conversation grows."""
    conversation_messages = []

    # Simulate a growing conversation
    test_messages = [
        "Hello", "How are you?", "What's the weather?", "Tell me about AI",
        "Explain machine learning", "What about deep learning?", "Show me examples",
        "How does this work?", "Give me more details", "I need comprehensive analysis",
        "Please provide research data", "Analyze this thoroughly"
    ]

    for i, message in enumerate(test_messages, 1):
        conversation_messages.append(message)

        print(f"\n=== Message {i}: '{message}' ===")

        # Create a mock state to test model selection
        mock_state = {"messages": conversation_messages}

        # Show which model would be selected
        if len(conversation_messages) < 10:
            print(f"Would use Qwen3 ({len(conversation_messages)} messages)")
            agent = create_agent(select_model, tools=tool_list)
            agent.invoke(mock_state)
        else:
            print(f"Would use GPT-OSS ({len(conversation_messages)} messages) - SWITCHED!")
            agent = create_agent(select_model, tools=tool_list)
            agent.invoke(mock_state)
            break  # Stop demo after switch

demo_conversation_progression()


# ============================================================================
# NOTEBOOK 04: Advanced Model Selection
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.runtime import Runtime
import tools

def intelligent_model_select(state: AgentState, runtime: Runtime) -> ChatOllama:
    """Intelligent model selection based on multiple factors."""
    messages = state["messages"]
    message_count = len(messages)

    # Factor 1: Calculate total content length
    total_length = sum(
        len(str(msg.content))
        for msg in messages
        if hasattr(msg, 'content') and msg.content
    )

    # Factor 2: Check for complexity keywords
    complex_keywords = ["analysis", "research", "detailed", "comprehensive", "complex", "strategy"]
    has_complex_content = any(
        keyword in str(msg.content).lower()
        for msg in messages
        for keyword in complex_keywords
        if hasattr(msg, 'content') and msg.content
    )

    # Multi-factor decision logic
    if total_length > 3000 or has_complex_content or message_count > 8:
        print(f"GPT-OSS: {message_count} msgs, {total_length} chars, keywords: {has_complex_content}")
        return ChatOllama(model="gpt-oss", temperature=0.0, num_predict=2500).bind_tools([tools.calculate])
    else:
        print(f"Qwen3: {message_count} msgs, {total_length} chars, keywords: {has_complex_content}")
        return ChatOllama(model="qwen3", temperature=0.1, num_predict=1000)

# Create agent with intelligent model selection
agent = create_agent(intelligent_model_select, tools=[tools.web_search])
print("✓ Smart agent created!")

result1 = agent.invoke({"messages": "Hello there"})
print("✓ Simple query processed")

complex_query = "I need comprehensive analysis and detailed research on AI strategies"
result2 = agent.invoke({"messages": complex_query})
print("✓ Complex keywords triggered GPT-OSS")

# Create long message (>3000 chars)
long_message = "Please help me understand this topic. " * 80
result3 = agent.invoke({"messages": long_message})
print(f"✓ Long content ({len(long_message)} chars) triggered GPT-OSS")


# ============================================================================
# NOTEBOOK 05: Tool Configurations
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, ToolNode
import tools

print("=== Method 1: Simple List of Tools ===")

# Create model instance
model = ChatOllama(model="qwen3")

# Method 1: Pass list of tools (simple approach)
simple_tools = [tools.web_search, tools.calculate]
agent1 = create_agent(
    model,
    tools=simple_tools
)

print("✓ Agent 1 created with simple tool list")
print("  - LangChain automatically creates ToolNode")
print("  - Default error handling behavior")
print("  - Good for prototyping and simple use cases")

print("=== Method 2: Advanced ToolNode Configuration ===")

# Method 2: Use ToolNode with error handling (advanced approach)
tool_node = ToolNode(
    tools=[tools.web_search, tools.calculate],
    handle_tool_errors="Please check your input and try again. Error details will help you correct the issue."
)

agent2 = create_agent(model, tools=tool_node)

print("✓ Agent 2 created with advanced ToolNode")
print("  - Custom error handling messages")
print("  - Better control over tool execution")
print("  - Production-ready error recovery")

# Unlike tools.calculate.invoke() which directly calls the tool,
# ToolNode.invoke() requires a state dictionary with messages containing tool_calls.
# This is because ToolNode is designed to work within a LangGraph agent workflow where messages follow this structure.
from langchain_core.messages import AIMessage

state_with_invalid_args = {
    "messages": [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"input": "What is 123 * 456?"},
                    "id": "call_456",
                    "type": "tool_call"
                }
            ]
        )
    ]
}

result = tool_node.invoke(state_with_invalid_args)
print(result)

# Test query that uses multiple tools
test_query = "Search for Python tutorials and calculate 10 times 5"

print(f"Testing both agents with: '{test_query}'")
print("=" * 60)

# Test Agent 1 (Simple)
print("\n=== Agent 1 Results (Simple Tool List) ===")
try:
    result1 = agent1.invoke({"messages": test_query})
    print(f"✓ Response: {result1['messages'][-1].content}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test Agent 2 (Advanced)
print("\n=== Agent 2 Results (Advanced ToolNode) ===")
try:
    result2 = agent2.invoke({"messages": test_query})
    print(f"✓ Response: {result2['messages'][-1].content}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n💡 Both agents work the same for valid inputs, but Agent 2 handles errors better")

def test_agent(agent, name, query):
    """Test an agent with a query and display results."""
    print(f"\n=== {name} ===")
    try:
        result = agent.invoke({"messages": query})
        print(f"✓ {result['messages'][-1].content}")
    except Exception as e:
        print(f"✗ Error: {e}")

# Test 1: Valid query
valid_query = "Search for Python tutorials and calculate 10 times 5"
print(f"Test 1 - Valid Query: '{valid_query}'")
print("=" * 60)
test_agent(agent1, "Simple Agent", valid_query)
test_agent(agent2, "Advanced Agent", valid_query)

# Test 2: Query that causes tool error
error_query = "Calculate 10 divided by 0"
print(f"\n\nTest 2 - Error Query: '{error_query}'")
print("=" * 60)
test_agent(agent1, "Simple Agent (crashes)", error_query)
test_agent(agent2, "Advanced Agent (handles gracefully)", error_query)


# ============================================================================
# NOTEBOOK 06: Prompt Configurations
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from typing import TypedDict
from code_notebooks import tools

print("=== Method 1: Simple String Prompt ===")

# Create model instance
model = ChatOllama(model="qwen3")

# Method 1: String prompt (simplest)
agent1 = create_agent(
    model,
    [tools.helper_tool],
    prompt="You are a helpful assistant. Be concise and accurate in your responses."
)

print("✓ Agent 1 created with string prompt")
print("  Characteristics:")
print("  - Direct string instruction")
print("  - Simple and readable")
print("  - Good for basic use cases")
print("  - Easy to modify")

print("=== Method 2: SystemMessage Prompt ===")

# Method 2: SystemMessage prompt (structured)
agent2 = create_agent(
    model,
    [tools.helper_tool],
    prompt=SystemMessage(content="You are a research assistant. Always cite your sources and provide detailed explanations.")
)

print("✓ Agent 2 created with SystemMessage prompt")
print("  Characteristics:")
print("  - Structured message object")
print("  - Better integration with chat models")
print("  - More explicit about message type")
print("  - Professional for production use")

# Test question for all agents
test_question = "Help me understand artificial intelligence"

print(f"Testing all agents with: '{test_question}'")
print("=" * 70)

# Test Agent 1 (String prompt)
print("\n=== Agent 1 Response (String Prompt) ===")
print("Expected: Concise and accurate response")

result1 = agent1.invoke({"messages": test_question})
print(f"Response: {result1['messages'][-1].content}")

# Test Agent 2 (SystemMessage prompt)
print("\n=== Agent 2 Response (SystemMessage Prompt) ===")
print("Expected: Detailed explanation with research focus")

result2 = agent2.invoke({"messages": test_question})
print(f"Response: {result2['messages'][-1].content}")


# ============================================================================
# NOTEBOOK 07: Structured Output with Pydantic V2
# ============================================================================

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
# import tools

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


class ListofContactInfo(BaseModel):
    contacts: list[ContactInfo] = Field(..., description="List of contact information")

model = ChatOllama(model="qwen3")

agent = create_agent(
    model,
    tools=[], # if case if you don't want to use any tools, just pass an empty list
    response_format=ContactInfo
)

result = agent.invoke({
    "messages":"Extract contact info from: John Doe, john@example.com, (555) 123-4567, Jon Do, john@example.com1, (555) 123"
})

result

model = ChatOllama(model="qwen3")

agent = create_agent(
    model,
    tools=[], # if case if you don't want to use any tools, just pass an empty list
    response_format=ListofContactInfo
)

result = agent.invoke({
    "messages":"Extract contact info from: John Doe, john@example.com, (555) 123-4567, Jon Do, john@example.com1, (555) 123"
})
result

print(result['structured_response'].contacts[0])
print(result['structured_response'].contacts[1])


# ============================================================================
# NOTEBOOK 08: Custom State Management
# ============================================================================

# AgentState-> state: {'messages': [your message 1, m2, m3 and so on]}

# Import required modules
from typing_extensions import Annotated
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.graph.message import add_messages
from code_notebooks import tools

print("=== Custom State for Memory Management ===")

# Define custom state schema
class CustomAgentState(AgentState):
    """Extended agent state with comprehensive memory management."""

    # Required: conversation history (from base AgentState)
    messages: Annotated[list, add_messages]

    # Custom: user preferences and session data
    user_preferences: dict = {}
    session_data: dict = {}

print("✓ CustomAgentState defined with custom tracking:")
print("  - messages: Conversation history (required)")
print("  - user_preferences: User settings and preferences")
print("  - session_data: Temporary session information")

print("=== Creating Agent with Custom State ===")

model = ChatOllama(model="qwen3")

# Create agent with custom state schema
agent = create_agent(
    model,
    tools=[tools.save_user_preference, tools.get_user_preference],
    state_schema=CustomAgentState  # This enables custom state tracking
)

print("✓ Agent created with custom state schema")
print("  The agent can now track user preferences and session data")
print("  State persists throughout the conversation")

print("=== Testing Custom State Features - Saving Preferences ===")

# Test 1: Save multiple preferences
user_message = """Please save these user preferences for me:
- explanation_preference: technical explanations and detailed examples
- code_style: verbose with comments
- output_format: structured with examples"""

print(f"User input: {user_message}\n")

result = agent.invoke({
    "messages": user_message,
    "session_data": {
        "session_id": "demo123",
        "start_time": "2024-01-01T10:00:00"
    }
})

print(f"Agent response: {result['messages'][-1].content}\n")
print("=" * 60)

result

print("=== Testing Custom State Features - Reading Preferences ===")

# Test 2: Retrieve saved preferences
user_message = "What preferences do I have saved? Please check explanation_preference, code_style, and output_format."

print(f"User input: {user_message}\n")

result = agent.invoke({
    "messages": result["messages"] + [{"role": "user", "content": user_message}],
    "session_data": result.get("session_data", {})
})

print(f"Agent response: {result['messages'][-1].content}\n")
print("=" * 60)
print("\n✓ Preferences are now persisted to file and can be retrieved across sessions")


# ============================================================================
# NOTEBOOK 09: Streaming Responses
# ============================================================================

# Import required modules
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from code_notebooks import tools

print("=== Streaming Agent Responses ===")

model = ChatOllama(model="qwen3")
agent = create_agent(model, tools=[tools.web_search, tools.calculate])

print("✓ Streaming agent created with tools: web_search, calculate")
print("  Ready for real-time streaming responses")

print("=== Testing Streaming Responses ===")

# Test streaming with a multi-step query
query = "Search for the latest AI news, then calculate 25 * 4 + 100"

print(f"\nStreaming response for: '{query}'")
print("Watching agent work in real-time:")
print("-" * 50)

try:
    # Stream the agent's execution
    for chunk in agent.stream({"messages": query}, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            latest_message = chunk["messages"][-1]

            # Handle different message types
            if hasattr(latest_message, 'content') and latest_message.content:
                if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                    tool_names = [tc['name'] for tc in latest_message.tool_calls]
                    print(f"Agent is calling tools: {tool_names}")
                else:
                    print(f"Agent thinking: {latest_message.content[:100]}...")

            elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                tool_names = [tc['name'] for tc in latest_message.tool_calls]
                print(f"Agent calling tools: {tool_names}")

    print("-" * 50)
    print("Streaming completed")

except Exception as e:
    print(f"Streaming demo error: {e}")
    # Fallback to regular invoke
    result = agent.invoke({"messages": query})
    print(f"Fallback response: {result['messages'][-1].content}")

print("In production, you'd see each step as it happens")

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver


# ============================================================================
# NOTEBOOK 10: Short-Term Memory Management
# ============================================================================

from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from code_notebooks import tools


# Example 1: InMemoryStore for two different users
checkpointer = InMemorySaver()

agent = create_agent(
    ChatOllama(model="qwen3"),
    tools=[],
    checkpointer=checkpointer,
)

# User 1: kgptalkie
agent.invoke(
    {"messages": "Hi! My name is kgptalkie."},
    {"configurable": {"thread_id": "kgptalkie"}}
)

result = agent.invoke(
    {"messages": "What's my name?"},
    {"configurable": {"thread_id": "kgptalkie"}}
)
print("kgptalkie:", result["messages"][-1].content)

# User 2: laxmikant
agent.invoke(
    {"messages": "Hi! My name is laxmikant."},
    {"configurable": {"thread_id": "laxmikant"}}
)

result = agent.invoke(
    {"messages": "What's my name?"},
    {"configurable": {"thread_id": "laxmikant"}}
)
print("laxmikant:", result["messages"][-1].content)


# Example 2: Trim messages
def pre_model_hook(state) -> dict[str, list[BaseMessage]]:
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}


agent = create_agent(
    ChatOllama(model="qwen3"),
    tools=[],
    pre_model_hook=pre_model_hook,
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": "hi, my name is bob"},
    {"configurable": {"thread_id": "trim_example"}}
)
agent.invoke(
    {"messages": "write a short poem about cats"},
    {"configurable": {"thread_id": "trim_example"}}
)
agent.invoke(
    {"messages": "now do the same but for dogs"},
    {"configurable": {"thread_id": "trim_example"}}
)
result = agent.invoke(
    {"messages": "what's my name?"},
    {"configurable": {"thread_id": "trim_example"}}
)

print("Trim result:", result["messages"][-1].content)


# Example 3: Delete messages
def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


agent = create_agent(
    ChatOllama(model="qwen3"),
    tools=[],
    post_model_hook=delete_messages,
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": "hi! I'm alice"},
    {"configurable": {"thread_id": "delete_example"}}
)
result = agent.invoke(
    {"messages": "what's my name?"},
    {"configurable": {"thread_id": "delete_example"}}
)

print("Delete result:", result["messages"][-1].content)
print("Message count:", len(result["messages"]))


# Example 4: Summarize messages
from langmem.short_term import SummarizationNode, RunningSummary


class State(AgentState):
    context: dict[str, RunningSummary]


model = ChatOllama(model="qwen3")

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

agent = create_agent(
    model=model,
    tools=[],
    pre_model_hook=summarization_node,
    state_schema=State,
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": "hi, my name is charlie"},
    {"configurable": {"thread_id": "summary_example"}}
)
agent.invoke(
    {"messages": "write a short poem about cats"},
    {"configurable": {"thread_id": "summary_example"}}
)
agent.invoke(
    {"messages": "now do the same but for dogs"},
    {"configurable": {"thread_id": "summary_example"}}
)
result = agent.invoke(
    {"messages": "what's my name?"},
    {"configurable": {"thread_id": "summary_example"}}
)

print("Summary result:", result["messages"][-1].content)
print("Summary:", result["context"]["running_summary"].summary)

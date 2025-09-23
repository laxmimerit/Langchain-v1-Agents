"""
DYNAMIC MODEL SELECTION BASED ON CONVERSATION LENGTH

This is a cost-optimization strategy where the agent automatically switches
between models based on conversation complexity. It starts with the efficient
Qwen3 model and switches to the more capable GPT-OSS when needed.

Key Benefits:
- Cost optimization: Use cheaper model when possible
- Performance scaling: Better model for complex tasks
- Automatic decision-making: No manual switching required

Selection Logic:
- < 10 messages: Use Qwen3 (fast, efficient)
- ≥ 10 messages: Use GPT-OSS (better reasoning, longer context)

Real-World Application:
- Customer service bots (simple queries → Qwen3, complex issues → GPT-OSS)
- Research assistants (quick facts → Qwen3, analysis → GPT-OSS)

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama pull gpt-oss
ollama serve
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.runtime import Runtime
import tools

# Load environment variables from .env file
load_dotenv()

def example_3_dynamic_model_selection():
    """
    DYNAMIC MODEL SELECTION BASED ON CONVERSATION LENGTH
    
    This is a cost-optimization strategy where the agent automatically switches
    between models based on conversation complexity. It starts with the efficient
    Qwen3 model and switches to the more capable GPT-OSS when needed.
    
    Key Benefits:
    - Cost optimization: Use cheaper model when possible
    - Performance scaling: Better model for complex tasks
    - Automatic decision-making: No manual switching required
    
    Selection Logic:
    - < 10 messages: Use Qwen3 (fast, efficient)
    - ≥ 10 messages: Use GPT-OSS (better reasoning, longer context)
    
    Real-World Application:
    - Customer service bots (simple queries → Qwen3, complex issues → GPT-OSS)
    - Research assistants (quick facts → Qwen3, analysis → GPT-OSS)
    """
    print("\n=== Example 3: Dynamic Model Selection (Qwen3 → GPT-OSS) ===")
    print("🔄 This example demonstrates automatic model switching based on conversation length.")
    print("   Short conversations use Qwen3, longer ones automatically upgrade to GPT-OSS.")

    tool_list = [tools.web_search, tools.analyze_data]

    def select_model(state: AgentState, runtime: Runtime) -> ChatOllama:
        """Choose between Qwen3 and GPT-OSS based on conversation length."""
        messages = state["messages"]
        message_count = len(messages)
        
        if message_count < 10:
            print(f"  🟢 Using Qwen3 for {message_count} messages (efficient)")
            return ChatOllama(model="qwen3", temperature=0.1).bind_tools(tool_list)
        else:
            print(f"  🔵 Switching to GPT-OSS for {message_count} messages (advanced)")
            return ChatOllama(model="gpt-oss", temperature=0.0, num_predict=2000).bind_tools(tool_list)

    agent = create_agent(select_model, tools=tool_list)

    # Test 1: Short conversation (Qwen3)
    print("\n📝 Testing short conversation:")
    result1 = agent.invoke({
        "messages": "Search for AI news"
    })
    print(f"✅ Short conversation result: {result1['messages'][-1].content}")

    # Test 2: Simulate long conversation (GPT-OSS)
    print("\n📝 Testing long conversation (12+ messages):")
    long_messages = "This is message number 12 in our conversation. I need complex analysis."
    agent_with_history = create_agent(select_model, tools=tool_list)
    result2 = agent_with_history.invoke({
        "messages": [f"Message {i}" for i in range(12)] + [long_messages]
    })
    print(f"✅ Long conversation triggered model switch")
    return agent

if __name__ == "__main__":
    example_3_dynamic_model_selection()

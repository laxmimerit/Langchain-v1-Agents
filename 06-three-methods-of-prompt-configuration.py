"""
THREE WAYS TO CONFIGURE AGENT PROMPTS

Method 1: String Prompt (Simplest)
- Direct string instruction to the agent
- Good for simple, static instructions
- Easy to read and modify

Method 2: SystemMessage (Structured)
- Uses LangChain's SystemMessage class
- More explicit about message type
- Better integration with chat models

Method 3: Callable/Dynamic Prompt (Advanced)
- Function that generates prompts based on state
- Can adapt to user preferences, context, etc.
- Most flexible but more complex

Dynamic Prompt Benefits:
- Personalization (expert vs beginner responses)
- Context-aware instructions
- Adaptive behavior based on conversation history

When to Use Each:
- String: Simple, static agents
- SystemMessage: Production chat agents
- Callable: Adaptive, personalized agents

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
import tools

# Load environment variables from .env file
load_dotenv()

def example_6_prompt_configurations():
    """
    THREE WAYS TO CONFIGURE AGENT PROMPTS
    ...existing docstring...
    """
    print("\n=== Example 6: Three Methods of Prompt Configuration ===")
    print("📝 This example demonstrates different ways to set up agent prompts.")
    print("   String → SystemMessage → Dynamic callable prompts")

    model = ChatOllama(model="qwen3")

    # Method 1: String prompt (simplest)
    print("\n🔤 Method 1: Simple string prompt")
    agent1 = create_agent(
        model,
        [tools.helper_tool],
        prompt="You are a helpful assistant. Be concise and accurate in your responses."
    )

    # Method 2: SystemMessage prompt (structured)
    print("💬 Method 2: SystemMessage prompt")
    agent2 = create_agent(
        model,
        [tools.helper_tool],
        prompt=SystemMessage(content="You are a research assistant. Always cite your sources and provide detailed explanations.")
    )

    # Method 3: Callable/Dynamic prompt (most flexible)
    print("🔄 Method 3: Dynamic callable prompt")
    def dynamic_prompt(state):
        user_type = state.get("user_type", "standard")
        system_msg = SystemMessage(
            content="Provide detailed technical responses with code examples and advanced concepts."
            if user_type == "expert"
            else "Provide simple, clear explanations suitable for beginners."
        )
        return [system_msg] + state["messages"]

    agent3 = create_agent(model, [tools.helper_tool], prompt=dynamic_prompt)

    # Test all three agents
    test_question = "Help me understand artificial intelligence"

    print(f"\n📝 Testing all agents with: '{test_question}'")

    result1 = agent1.invoke({"messages": test_question})
    print(f"✅ String prompt: {result1['messages'][-1].content}")

    result2 = agent2.invoke({"messages": test_question})
    print(f"✅ System message: {result2['messages'][-1].content}")

    # Test dynamic prompt with expert mode
    result3 = agent3.invoke({
        "messages": test_question,
        "user_type": "expert"  # This triggers technical response
    })
    print(f"✅ Dynamic prompt (expert mode): {result3['messages'][-1].content}")

    print("💡 Notice how each agent responds differently based on its prompt configuration")
    return agent1, agent2, agent3

if __name__ == "__main__":
    example_6_prompt_configurations()

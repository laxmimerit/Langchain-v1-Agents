"""
SOPHISTICATED CONTENT-AWARE MODEL SELECTION

This advanced approach analyzes not just message count, but also:
- Content complexity (keyword analysis)
- Total conversation length (character count)
- Context clues about task difficulty

Selection Factors:
1. Keyword Analysis: Looks for words like "analysis", "research", "comprehensive"
2. Content Length: Long conversations often need better reasoning
3. Message Count: Many exchanges suggest complex discussion

Algorithm Logic:
- If ANY of these conditions are true → Use GPT-OSS:
  • Total characters > 3000
  • Contains complexity keywords
  • Message count > 8
- Otherwise → Use Qwen3

Real-World Benefits:
- A single complex question immediately gets the better model
- Long simple conversations don't waste expensive model usage
- Context-aware decisions improve user experience

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

def example_4_advanced_model_selection():
    """
    SOPHISTICATED CONTENT-AWARE MODEL SELECTION
    
    This advanced approach analyzes not just message count, but also:
    - Content complexity (keyword analysis)
    - Total conversation length (character count)
    - Context clues about task difficulty
    
    Selection Factors:
    1. Keyword Analysis: Looks for words like "analysis", "research", "comprehensive"
    2. Content Length: Long conversations often need better reasoning
    3. Message Count: Many exchanges suggest complex discussion
    
    Algorithm Logic:
    - If ANY of these conditions are true → Use GPT-OSS:
      • Total characters > 3000
      • Contains complexity keywords
      • Message count > 8
    - Otherwise → Use Qwen3
    
    Real-World Benefits:
    - A single complex question immediately gets the better model
    - Long simple conversations don't waste expensive model usage
    - Context-aware decisions improve user experience
    """
    print("\n=== Example 4: Sophisticated Content-Aware Model Selection ===")
    print("🧠 This example uses intelligent analysis of message content, not just count.")
    print("   It detects complexity through keywords, length, and context.")

    tool_list = [tools.research_tool]

    def intelligent_model_select(state: AgentState, runtime: Runtime) -> ChatOllama:
        """Intelligent model selection based on multiple factors."""
        messages = state["messages"]
        message_count = len(messages)
        
        # Factor 1: Calculate total content length
        total_length = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content') and msg.content)
        
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
            print(f"  🔵 GPT-OSS selected: {message_count} msgs, {total_length} chars, complex_keywords: {has_complex_content}")
            return ChatOllama(model="gpt-oss", temperature=0.0, num_predict=2500).bind_tools(tool_list)
        else:
            print(f"  🟢 Qwen3 selected: {message_count} msgs, {total_length} chars, complex_keywords: {has_complex_content}")
            return ChatOllama(model="qwen3", temperature=0.1, num_predict=1000).bind_tools(tool_list)

    agent = create_agent(intelligent_model_select, tools=tool_list)

    # Test 1: Simple query (should use Qwen3)
    print("\n📝 Testing simple query:")
    result1 = agent.invoke({
        "messages": "Hello there"
    })

    # Test 2: Complex query with keywords (should immediately use GPT-OSS)
    print("\n📝 Testing complex query with keywords:")
    result2 = agent.invoke({
        "messages": "I need a comprehensive analysis and detailed research on market strategies for AI companies"
    })

    print("💡 Notice: Complex keywords triggered GPT-OSS even for a single message")
    return agent

if __name__ == "__main__":
    example_4_advanced_model_selection()

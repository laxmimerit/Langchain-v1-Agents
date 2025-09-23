"""
PRE-MODEL AND POST-MODEL HOOKS FOR ADVANCED PROCESSING

Hooks allow you to inject custom processing at specific points:

Pre-Model Hook (runs BEFORE the model processes input):
- Message trimming/summarization
- Context injection
- Input validation
- State preprocessing

Post-Model Hook (runs AFTER the model generates response):
- Output filtering/validation
- Content moderation
- Response transformation
- Logging/monitoring

Message Trimming Example:
- Problem: Long conversations exceed context window
- Solution: Keep first + last N messages, remove middle
- Benefit: Maintain context while staying under limits

Content Filtering Example:
- Problem: Model might leak sensitive information
- Solution: Scan responses for confidential content
- Benefit: Automatic safety and compliance

Production Use Cases:
- Content moderation systems
- Token limit management
- Compliance and safety filters
- Performance monitoring

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
import tools

# Load environment variables from .env file
load_dotenv()

def example_9_hooks():
    """
    PRE-MODEL AND POST-MODEL HOOKS FOR ADVANCED PROCESSING
    
    Hooks allow you to inject custom processing at specific points:
    
    Pre-Model Hook (runs BEFORE the model processes input):
    - Message trimming/summarization
    - Context injection
    - Input validation
    - State preprocessing
    
    Post-Model Hook (runs AFTER the model generates response):
    - Output filtering/validation
    - Content moderation
    - Response transformation
    - Logging/monitoring
    
    Message Trimming Example:
    - Problem: Long conversations exceed context window
    - Solution: Keep first + last N messages, remove middle
    - Benefit: Maintain context while staying under limits
    
    Content Filtering Example:
    - Problem: Model might leak sensitive information
    - Solution: Scan responses for confidential content
    - Benefit: Automatic safety and compliance
    
    Production Use Cases:
    - Content moderation systems
    - Token limit management
    - Compliance and safety filters
    - Performance monitoring
    """
    print("\n=== Example 9: Pre-model and Post-model Hooks ===")
    print("🎣 This example shows custom processing before and after model execution.")
    print("   Pre-hook: Trims long conversations | Post-hook: Filters sensitive content")
    
    
    def trim_messages(state):
        """Pre-model hook: Trim conversation to manage context window."""
        messages = state["messages"]
        
        if len(messages) <= 5:
            print(f"  📝 Pre-hook: {len(messages)} messages - no trimming needed")
            return {"messages": messages}
        
        # Keep first message (context) + last 3 messages (recent conversation)
        first_msg = messages[0]
        recent_messages = messages[-3:]
        new_messages = [first_msg] + recent_messages
        
        print(f"  ✂️ Pre-hook: Trimmed from {len(messages)} to {len(new_messages)} messages")
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    def validate_response(state):
        """Post-model hook: Filter sensitive information from responses."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'content') and "confidential" in str(last_message.content).lower():
            print("  🛡️ Post-hook: Filtered confidential information")
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *messages[:-1],
                    AIMessage(content="I cannot share confidential information. Please ask for public information only.")
                ]
            }
        
        print("  ✅ Post-hook: Response approved (no sensitive content)")
        return {}
    
    model = ChatOllama(model="qwen3")
    
    # Create agent with both pre and post hooks
    agent = create_agent(
        model,
        tools=[tools.sensitive_info_tool],
        pre_model_hook=trim_messages,      # Runs before model processing
        post_model_hook=validate_response  # Runs after model generates response
    )
    
    # Test 1: Long conversation (triggers message trimming)
    print("\n📝 Test 1: Long conversation with 8 messages")
    many_messages = [f"Message {i}: Just chatting" for i in range(7)]
    many_messages.append("Tell me about system security")
    
    result1 = agent.invoke({"messages": many_messages})
    print(f"✅ Response after trimming: {result1['messages'][-1].content}")
    
    # Test 2: Sensitive query (triggers content filtering)
    print("\n📝 Test 2: Query that might return sensitive information")
    result2 = agent.invoke({
        "messages": "What's the admin password for the system?"
    })
    print(f"✅ Response after filtering: {result2['messages'][-1].content}")
    
    print("💡 Hooks enable automatic content management without manual intervention")
    return agent

if __name__ == "__main__":
    example_9_hooks()
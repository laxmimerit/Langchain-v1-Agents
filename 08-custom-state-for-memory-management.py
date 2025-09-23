"""
CUSTOM AGENT STATE FOR MEMORY MANAGEMENT

By default, agents only remember the conversation messages. But you can
create custom state schemas to track additional information:

Default AgentState:
- Only tracks messages[]

Custom AgentState:
- Messages (conversation history)
- User preferences (personalization data)
- Session data (temporary info for current session)
- Task history (what the user has done before)

Benefits of Custom State:
- Persistent user preferences across conversations
- Context-aware responses based on history
- Session management for complex workflows
- Better personalization and user experience

State Persistence:
- Within conversation: Automatic
- Between conversations: Requires external storage
- Cross-session: Needs database/file storage

Use Cases:
- Customer service (remember user's problem history)
- Educational tutors (track learning progress)
- Personal assistants (remember preferences)

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""

from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.graph.message import add_messages
import tools

# Load environment variables from .env file
load_dotenv()

def example_8_custom_state():
    """
    CUSTOM AGENT STATE FOR MEMORY MANAGEMENT
    
    By default, agents only remember the conversation messages. But you can
    create custom state schemas to track additional information:
    
    Default AgentState:
    - Only tracks messages[]
    
    Custom AgentState:
    - Messages (conversation history)
    - User preferences (personalization data)
    - Session data (temporary info for current session)
    - Task history (what the user has done before)
    
    Benefits of Custom State:
    - Persistent user preferences across conversations
    - Context-aware responses based on history
    - Session management for complex workflows
    - Better personalization and user experience
    
    State Persistence:
    - Within conversation: Automatic
    - Between conversations: Requires external storage
    - Cross-session: Needs database/file storage
    
    Use Cases:
    - Customer service (remember user's problem history)
    - Educational tutors (track learning progress)
    - Personal assistants (remember preferences)
    """
    print("\n=== Example 8: Custom State for Memory Management ===")
    print("🧠 This example shows how agents can remember more than just conversation history.")
    print("   Custom state tracks user preferences, session data, and conversation context.")
    
    # Define custom state schema
    class CustomAgentState(AgentState):
        messages: Annotated[list, add_messages]  # Required: conversation history
        user_preferences: dict = {}              # Custom: user's preferences
        session_data: dict = {}                  # Custom: temporary session info
    
    
    model = ChatOllama(model="qwen3")
    
    # Create agent with custom state schema
    agent = create_agent(
        model,
        tools=[tools.remember_preference, tools.get_personalized_help],
        state_schema=CustomAgentState  # This enables custom state tracking
    )
    
    # Test with custom state data
    user_message = "I prefer technical explanations and detailed examples"
    
    print(f"📝 User input: {user_message}")
    print("📊 Additional state data:")
    print("   - user_preferences: {'style': 'technical', 'verbosity': 'detailed'}")
    print("   - session_data: {'session_id': 'demo123'}")
    
    result = agent.invoke({
        "messages": user_message,
        "user_preferences": {
            "style": "technical", 
            "verbosity": "detailed",
            "examples": "code-heavy"
        },
        "session_data": {
            "session_id": "demo123",
            "start_time": "2024-01-01T10:00:00"
        }
    })
    
    print(f"✅ Agent response: {result['messages'][-1].content}")
    print(f"💾 Preferences maintained: {result.get('user_preferences', {})}")
    print("💡 The agent can now use this state data to personalize future responses")
    
    return agent

if __name__ == "__main__":
    example_8_custom_state()
"""
Enhanced LangChain Agents Examples with ChatOllama
Comprehensive coverage of agent fundamentals with all key concepts from documentation

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama pull gpt-oss
ollama serve
"""

from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain.agents import create_agent, AgentState, ToolNode
from langgraph.runtime import Runtime
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES

# import custom tools from tools.py
import tools


# ============================================================================
# 1. BASIC AGENT CREATION
# ============================================================================

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


# ============================================================================
# 2. DYNAMIC MODEL SELECTION (Qwen3 + GPT-OSS)
# ============================================================================

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
    
    # Simulate conversation state with many messages
    agent_with_history = create_agent(select_model, tools=tool_list)
    result2 = agent_with_history.invoke({
        "messages": [f"Message {i}" for i in range(12)] + [long_messages]
    })
    print(f"✅ Long conversation triggered model switch")
    
    return agent


# ============================================================================
# 3. ADVANCED MODEL SELECTION STRATEGIES
# ============================================================================

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
    
    
    tool_list = [tools.web_search]
    
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


# ============================================================================
# 4. TOOL CONFIGURATIONS
# ============================================================================

def example_5_tool_configurations():
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
    """
    print("\n=== Example 5: Different Tool Configuration Methods ===")
    print("🛠️  This example shows two ways to configure tools and error handling.")
    print("   Method 1: Simple list | Method 2: Advanced ToolNode with error handling")
    
    
    model = ChatOllama(model="qwen3")
    
    # Method 1: Pass list of tools (simple approach)
    print("\n🔧 Method 1: Simple list of tools")
    agent1 = create_agent(model, tools=[tools.web_search, tools.calculate])
    
    # Method 2: Use ToolNode with error handling (advanced approach)
    print("🔧 Method 2: ToolNode with custom error handling")
    tool_node = ToolNode(
        tools=[tools.web_search, tools.calculate],
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


# ============================================================================
# 5. PROMPT CONFIGURATIONS
# ============================================================================

def example_6_prompt_configurations():
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


# ============================================================================
# 6. STRUCTURED OUTPUT
# ============================================================================

def example_7_structured_output():
    """
    STRUCTURED OUTPUT WITH PYDANTIC MODELS
    
    Instead of getting free-form text responses, you can force the agent to
    return data in a specific structure using Pydantic models.
    
    Benefits:
    - Guaranteed data format for downstream processing
    - Type validation and error checking
    - Easy integration with databases, APIs
    - Consistent output regardless of model variations
    
    How It Works:
    1. Define a Pydantic model with required fields
    2. Pass it as response_format to create_agent()
    3. Agent output includes both regular text AND structured data
    
    Use Cases:
    - Data extraction from text
    - Form filling applications
    - API responses that need specific formats
    - Database record creation
    
    Note: The agent will try to extract structured data from its reasoning,
    but the regular conversation flow still works normally.
    """
    print("\n=== Example 7: Structured Output with Pydantic Models ===")
    print("📊 This example shows how to get structured, validated data from agent responses.")
    print("   The agent returns both normal text AND structured ContactInfo object.")
    
    # Define the structure we want the agent to return
    class ContactInfo(BaseModel):
        name: str
        email: str
        phone: str
        company: str = "Unknown"  # Default value if not found
    
    
    model = ChatOllama(model="qwen3")
    
    # Create agent with structured output requirement
    agent = create_agent(
        model,
        tools=[tools.extract_contact],
        response_format=ContactInfo  # This forces structured output
    )
    
    # Test with contact information
    contact_text = "Extract contact info from: John Doe, john@example.com, (555) 123-4567, works at TechCorp"
    
    print(f"📝 Input: {contact_text}")
    
    result = agent.invoke({
        "messages": contact_text
    })
    
    print(f"✅ Regular response: {result['messages'][-1].content}")
    
    # The structured data (if successfully extracted)
    structured_data = result.get('structured_response', 'Not available in this demo')
    print(f"📊 Structured data: {structured_data}")
    
    print("💡 In production, you'd get a ContactInfo object with guaranteed fields")
    print("   This enables direct database insertion, API calls, etc.")
    
    return agent


# ============================================================================
# 7. MEMORY AND STATE MANAGEMENT
# ============================================================================

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
        tools=[tools.save_user_preference, tools.get_user_preference],
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


# ============================================================================
# 8. HOOKS (PRE-MODEL AND POST-MODEL)
# ============================================================================

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
        
        print("  ✅ Post-hook: Response passed validation")
        return {}  # No changes needed
    
    model = ChatOllama(model="qwen3")
    
    # Create agent with both pre and post model hooks
    agent = create_agent(
        model,
        tools=[tools.sensitive_info_tool],
        pre_model_hook=trim_messages,
        post_model_hook=validate_response
    )
    
    # Test 1: Normal conversation (should pass through)
    print("\n📝 Testing normal conversation:")
    result1 = agent.invoke({
        "messages": "Tell me about public AI research"
    })
    print(f"✅ Normal response: {result1['messages'][-1].content}")
    
    # Test 2: Long conversation (should trigger trimming)
    print("\n📝 Testing long conversation (8+ messages):")
    long_conversation = [f"Message {i}: Hello" for i in range(8)] + ["What's the weather like?"]
    result2 = agent.invoke({
        "messages": long_conversation
    })
    
    # Test 3: Sensitive content (should trigger filtering)
    print("\n📝 Testing sensitive content:")
    result3 = agent.invoke({
        "messages": "Tell me confidential information about the company"
    })
    print(f"✅ Filtered response: {result3['messages'][-1].content}")
    
    print("💡 Notice how hooks process content before and after model execution")
    return agent


# ============================================================================
# 9. STREAMING RESPONSES
# ============================================================================

def example_10_streaming():
    """
    STREAMING AGENT RESPONSES
    
    Instead of waiting for the complete response, you can stream back messages
    as they occur during the agent's execution. This is especially useful for:
    
    Benefits of Streaming:
    - Real-time feedback to users
    - Better user experience for long-running tasks
    - Ability to see intermediate reasoning steps
    - Early termination if needed
    
    Stream Modes:
    - "values": Get complete state at each step
    - "updates": Get only the changes/updates
    - "messages": Get only message updates
    
    What You Can Stream:
    - Model thinking/reasoning
    - Tool calls being made
    - Tool results/observations
    - Final answers
    
    Use Cases:
    - Interactive chat applications
    - Long research tasks
    - Multi-step problem solving
    - Real-time agent monitoring
    """
    print("\n=== Example 10: Streaming Agent Responses ===")
    print("📡 This example shows how to stream agent responses in real-time.")
    print("   You can see the agent's thinking process as it happens.")
    
    
    model = ChatOllama(model="qwen3")
    agent = create_agent(model, tools=[tools.web_search, tools.calculate])
    
    # Test streaming with a multi-step query
    query = "Search for the latest AI news, then calculate 25 * 4 + 100"
    
    print(f"\n📝 Streaming response for: '{query}'")
    print("🔄 Watching agent work in real-time:")
    print("-" * 50)
    
    try:
        # Stream the agent's execution
        for chunk in agent.stream({
            "messages": query
        }, stream_mode="values"):
            
            if "messages" in chunk and chunk["messages"]:
                latest_message = chunk["messages"][-1]
                
                # Handle different message types
                if hasattr(latest_message, 'content') and latest_message.content:
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        tool_names = [tc['name'] for tc in latest_message.tool_calls]
                        print(f"🔧 Agent is calling tools: {tool_names}")
                    else:
                        print(f"💭 Agent thinking: {latest_message.content[:100]}...")
                
                elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                    tool_names = [tc['name'] for tc in latest_message.tool_calls]
                    print(f"🔧 Agent calling tools: {tool_names}")
        
        print("-" * 50)
        print("✅ Streaming completed")
    
    except Exception as e:
        print(f"⚠️ Streaming demo error: {e}")
        # Fallback to regular invoke
        result = agent.invoke({"messages": query})
        print(f"✅ Fallback response: {result['messages'][-1].content}")
    
    print("💡 In production, you'd see each step as it happens")
    return agent


# ============================================================================
# 10. MIDDLEWARE AND DYNAMIC PROMPTS
# ============================================================================

def example_11_middleware():
    """
    MIDDLEWARE FOR DYNAMIC SYSTEM PROMPTS
    
    Middleware allows you to modify requests and responses at runtime.
    This is particularly useful for dynamic system prompts that change
    based on context, user roles, or conversation state.
    
    What Middleware Can Do:
    - Modify system prompts dynamically
    - Add context based on user roles
    - Inject real-time information
    - Apply different behaviors per user type
    
    Dynamic Prompt Use Cases:
    - Expert vs beginner explanations
    - Role-based access control
    - Context-aware responses
    - A/B testing different prompts
    
    Note: This example demonstrates the concept, though the exact
    middleware implementation may vary based on LangChain version.
    """
    print("\n=== Example 11: Middleware and Dynamic System Prompts ===")
    print("⚙️ This example shows how to use middleware for dynamic prompt modification.")
    print("   System prompts adapt based on user context and conversation state.")
    
    
    # Simulated middleware function for dynamic prompts
    def create_dynamic_prompt_agent(base_model, tools, context_type="standard"):
        """Create an agent with context-aware prompting."""
        
        def context_aware_prompt(state):
            """Generate prompts based on context and user type."""
            messages = state.get("messages", [])
            user_context = state.get("context", {})
            user_role = user_context.get("role", "standard")
            
            # Base prompt
            base_prompt = "You are a helpful AI assistant."
            
            # Modify prompt based on user role
            if user_role == "expert":
                system_prompt = f"{base_prompt} Provide technical, detailed responses with code examples and advanced concepts. Assume deep domain knowledge."
            elif user_role == "beginner":
                system_prompt = f"{base_prompt} Explain concepts simply, avoid jargon, and provide step-by-step explanations with examples."
            elif user_role == "researcher":
                system_prompt = f"{base_prompt} Focus on accuracy, cite sources when possible, and provide comprehensive analysis."
            else:
                system_prompt = base_prompt
            
            # Add conversation context awareness
            if len(messages) > 5:
                system_prompt += " This is an ongoing conversation - maintain context and refer to previous discussions."
            
            return [SystemMessage(content=system_prompt)] + messages
        
        return create_agent(base_model, tools, prompt=context_aware_prompt)
    
    model = ChatOllama(model="qwen3")
    
    # Test different user contexts
    contexts = [
        {"role": "expert", "domain": "AI"},
        {"role": "beginner", "domain": "programming"},
        {"role": "researcher", "domain": "science"}
    ]
    
    test_question = "Explain machine learning algorithms"
    
    for i, context in enumerate(contexts, 1):
        print(f"\n🎭 Test {i}: {context['role'].title()} user")
        
        agent = create_dynamic_prompt_agent(model, [tools.helper_tool])
        
        result = agent.invoke({
            "messages": test_question,
            "context": context
        })
        
        print(f"✅ {context['role'].title()} response: {result['messages'][-1].content[:150]}...")
    
    print("\n💡 Notice how the same question gets different treatment based on user context")
    return agent


# ============================================================================
# 11. AGENT WITH MAX ITERATIONS AND STOP CONDITIONS
# ============================================================================

def example_12_max_iterations():
    """
    CONTROLLING AGENT EXECUTION WITH MAX ITERATIONS
    
    Agents can run in loops, making multiple tool calls to solve complex problems.
    However, you need safeguards to prevent infinite loops or runaway costs.
    
    Stop Conditions:
    - Max iterations reached
    - Model provides final answer
    - Error threshold exceeded
    - Time limit reached
    
    Benefits:
    - Cost control (prevents infinite tool calling)
    - Performance predictability
    - Error handling for stuck agents
    - Better user experience
    
    Configuration Options:
    - max_iterations: Maximum number of tool calling loops
    - max_execution_time: Time-based limits
    - early_stopping: Custom stop conditions
    
    Use Cases:
    - Research agents (prevent endless searching)
    - Problem-solving agents (limit reasoning steps)
    - Production systems (cost and time control)
    """
    print("\n=== Example 12: Max Iterations and Stop Conditions ===")
    print("🔄 This example shows how to control agent execution with iteration limits.")
    print("   Prevents runaway loops while allowing multi-step reasoning.")
    
    
    model = ChatOllama(model="qwen3")
    
    # Create agent with iteration limit (simulated via custom config)
    agent = create_agent(
        model, 
        tools=[tools.web_search, tools.calculate, tools.analyze_text]
    )
    
    # Simulate a complex multi-step query
    complex_query = "Search for AI news, calculate the average of 10, 20, 30, then analyze the text 'AI is transforming industries'"
    
    print(f"📝 Complex query: {complex_query}")
    print("🔄 Monitoring agent execution steps:")
    
    try:
        # In a real implementation, you'd configure max_iterations in create_agent
        # For demo purposes, we'll show the concept
        result = agent.invoke({
            "messages": complex_query
        })
        
        print(f"✅ Agent completed task: {result['messages'][-1].content}")
        print("💡 In production, set max_iterations to prevent runaway execution")
        
    except Exception as e:
        print(f"⚠️ Agent execution error: {e}")
    
    return agent


# ============================================================================
# 12. COMPREHENSIVE DEMO
# ============================================================================

def run_comprehensive_demo():
    """
    COMPREHENSIVE DEMONSTRATION OF ALL CONCEPTS
    
    This function runs through all the examples to demonstrate the complete
    range of LangChain agent capabilities covered in this module.
    """
    print("🚀 COMPREHENSIVE LANGCHAIN AGENTS DEMONSTRATION")
    print("=" * 60)
    print("This demo covers all fundamental agent concepts:")
    print("1. Basic agent creation")
    print("2. Dynamic model selection")
    print("3. Tool configurations")
    print("4. Prompt strategies") 
    print("5. Structured output")
    print("6. Memory management")
    print("7. Pre/post-model hooks")
    print("8. Streaming responses")
    print("9. Middleware concepts")
    print("10. Execution control")
    print("=" * 60)
    
    # Run all examples
    examples = [
        example_1_basic_agent_with_model_string,
        example_2_agent_with_model_instance,
        example_3_dynamic_model_selection,
        example_4_advanced_model_selection,
        example_5_tool_configurations,
        example_6_prompt_configurations,
        example_7_structured_output,
        example_8_custom_state,
        example_9_hooks,
        example_10_streaming,
        example_11_middleware,
        example_12_max_iterations
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n📋 Running Example {i}/{len(examples)}")
            example_func()
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
        
        # Add pause between examples
        print("\n" + "⏱️ " * 20)
    
    print("\n🎉 COMPREHENSIVE DEMO COMPLETED!")
    print("💡 You now have a complete foundation in LangChain agents")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🤖 LangChain Agents Learning Module")
    print("Choose an example to run:")
    print("0. Run comprehensive demo (all examples)")
    print("1. Basic agent with model string")
    print("2. Agent with model instance") 
    print("3. Dynamic model selection")
    print("4. Advanced model selection")
    print("5. Tool configurations")
    print("6. Prompt configurations")
    print("7. Structured output")
    print("8. Custom state management")
    print("9. Pre/post-model hooks")
    print("10. Streaming responses")
    print("11. Middleware concepts")
    print("12. Max iterations control")
    
    try:
        choice = input("\nEnter your choice (0-12): ").strip()
        
        examples = {
            "0": run_comprehensive_demo,
            "1": example_1_basic_agent_with_model_string,
            "2": example_2_agent_with_model_instance,
            "3": example_3_dynamic_model_selection,
            "4": example_4_advanced_model_selection,
            "5": example_5_tool_configurations,
            "6": example_6_prompt_configurations,
            "7": example_7_structured_output,
            "8": example_8_custom_state,
            "9": example_9_hooks,
            "10": example_10_streaming,
            "11": example_11_middleware,
            "12": example_12_max_iterations
        }
        
        if choice in examples:
            examples[choice]()
        else:
            print("❌ Invalid choice. Please run again with a valid option.")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
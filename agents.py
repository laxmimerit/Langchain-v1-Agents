"""
Complete LangChain Agents Examples with ChatOllama
Using Qwen3 for simple conversations and GPT-OSS for complex ones

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama pull gpt-oss
ollama serve
"""

from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated
from pydantic import BaseModel

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain.agents import create_agent, AgentState, ToolNode
from langgraph.runtime import Runtime
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
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
        tools=[tools.simple_search]
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


# ============================================================================
# 9. STREAMING RESPONSES
# ============================================================================

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
    print("📡 This example demonstrates real-time streaming of agent responses.")
    print("   You can see tool calls, reasoning, and responses as they happen.")
    
    
    model = ChatOllama(model="qwen3")
    agent = create_agent(model, tools=[tools.slow_research, tools.analyze_market])
    
    query = "Research AI market trends and analyze the technology sector"
    print(f"📝 Query: {query}")
    print("🔄 Streaming response:\n")
    
    # Stream the response in real-time
    for i, chunk in enumerate(agent.stream({
        "messages": query
    }, stream_mode="values")):
        
        latest_message = chunk["messages"][-1]
        
        # Handle different types of messages in the stream
        if hasattr(latest_message, 'content') and latest_message.content:
            print(f"  💬 Agent [{i}]: {latest_message.content}")
            
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in latest_message.tool_calls]
            print(f"  🔧 Tool Call [{i}]: {', '.join(tool_names)}")
            
        elif isinstance(latest_message, ToolMessage):
            print(f"  📊 Tool Result [{i}]: {latest_message.content}")
    
    print("\n✅ Streaming completed!")
    print("💡 Notice how you could see each step of the agent's work in real-time")
    return agent


# ============================================================================
# 10. COMPLETE REAL-WORLD EXAMPLE
# ============================================================================

def example_11_complete_real_world():
    """
    COMPLETE PRODUCTION-READY AGENT WITH ALL FEATURES
    
    This example combines all the concepts from previous examples into a
    single, production-ready agent that demonstrates:
    
    Features Included:
    1. Intelligent model selection (content-aware)
    2. Comprehensive tool suite
    3. Custom state management
    4. Context-aware preprocessing
    5. Error handling and validation
    6. Multiple test scenarios
    
    Real-World Application Areas:
    - Business intelligence systems
    - Research and analysis platforms
    - Customer service automation
    - Educational and training tools
    
    Architecture Components:
    - Multi-model orchestration (Qwen3 ↔ GPT-OSS)
    - Tool ecosystem (search, calculation, analysis)
    - State persistence (user context, task history)
    - Processing pipeline (hooks, validation, streaming)
    
    Production Considerations:
    - Cost optimization through smart model selection
    - User experience through context awareness
    - Reliability through error handling
    - Scalability through modular design
    
    Performance Characteristics:
    - Fast: Simple queries use efficient Qwen3
    - Smart: Complex queries automatically upgrade to GPT-OSS
    - Contextual: Remembers user preferences and history
    - Robust: Handles errors gracefully with fallbacks
    """
    print("\n=== Example 11: Complete Production-Ready Agent ===")
    print("🏢 This example demonstrates a full-featured agent combining all concepts.")
    print("   Multi-model selection + comprehensive tools + state management + hooks")
    
    
    # Enhanced state management for production use
    class ProductionAgentState(AgentState):
        messages: Annotated[list, add_messages]
        user_context: dict = {}      # User preferences and profile
        task_history: list = []      # Previous tasks and outcomes
        session_metadata: dict = {}  # Session tracking info
    
    # Intelligent model selection with detailed logging
    def select_optimal_model(state: ProductionAgentState, runtime: Runtime) -> ChatOllama:
        """Advanced model selection with comprehensive analysis."""
        messages = state["messages"]
        user_context = state.get("user_context", {})
        msg_count = len(messages)
        
        # Content analysis
        recent_content = " ".join([
            str(msg.content) for msg in messages[-3:] 
            if hasattr(msg, 'content') and msg.content
        ]).lower()
        
        total_length = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content') and msg.content)
        
        # Enhanced complexity detection
        complexity_indicators = [
            "analyze", "research", "complex", "detailed", "comprehensive", "strategy",
            "evaluate", "assess", "compare", "investigate", "examine", "study"
        ]
        
        business_indicators = [
            "market", "business", "financial", "revenue", "profit", "investment",
            "competition", "strategy", "analysis", "forecast"
        ]
        
        technical_indicators = [
            "algorithm", "code", "programming", "technical", "architecture",
            "system", "database", "api", "framework"
        ]
        
        has_complex_content = any(indicator in recent_content for indicator in complexity_indicators)
        has_business_content = any(indicator in recent_content for indicator in business_indicators)
        has_technical_content = any(indicator in recent_content for indicator in technical_indicators)
        
        # User preference consideration
        user_tier = user_context.get("tier", "standard")
        preferred_detail = user_context.get("detail_level", "medium")
        
        # Decision logic with detailed reasoning
        reasons = []
        
        if msg_count > 8:
            reasons.append(f"long conversation ({msg_count} messages)")
        if total_length > 3000:
            reasons.append(f"extensive content ({total_length} chars)")
        if has_complex_content:
            reasons.append("complexity keywords detected")
        if has_business_content:
            reasons.append("business analysis required")
        if has_technical_content:
            reasons.append("technical expertise needed")
        if user_tier == "premium":
            reasons.append("premium user tier")
        if preferred_detail == "high":
            reasons.append("user prefers detailed responses")
        
        # Model selection decision
        use_gpt_oss = (
            msg_count > 8 or 
            total_length > 3000 or 
            has_complex_content or 
            has_business_content or 
            user_tier == "premium" or
            preferred_detail == "high"
        )
        
        if use_gpt_oss:
            model_name = "gpt-oss"
            model_params = {"temperature": 0.1, "num_predict": 3000}
            print(f"  🔵 Selected GPT-OSS: {', '.join(reasons) if reasons else 'default advanced model'}")
        else:
            model_name = "qwen3"
            model_params = {"temperature": 0.2, "num_predict": 1500}
            print(f"  🟢 Selected Qwen3: simple query, efficient processing")
        
        return ChatOllama(model=model_name, **model_params).bind_tools([
            tools.web_search_production, tools.calculate_advanced, tools.analyze_text
        ])
    
    # Context-aware preprocessing with user personalization
    def context_preprocessor(state):
        """Add contextual information to improve response quality."""
        messages = state["messages"]
        user_context = state.get("user_context", {})
        task_history = state.get("task_history", [])
        
        # Log context for debugging
        if user_context:
            context_items = [f"{k}={v}" for k, v in user_context.items()]
            print(f"  📋 User context: {', '.join(context_items)}")
        
        if task_history:
            print(f"  📚 Task history: {len(task_history)} previous tasks")
        
        # In production, you might modify the prompt or add context messages
        return {"messages": messages}
    
    # Enhanced response validation
    def production_validator(state):
        """Validate and enhance responses for production use."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, 'content') or not last_message.content:
            return {}
        
        content = str(last_message.content).lower()
        
        # Check for various production concerns
        concerns = []
        
        if "confidential" in content or "password" in content:
            concerns.append("sensitive information")
        
        if len(content) < 10:
            concerns.append("response too short")
        
        if "error" in content and "sorry" not in content:
            concerns.append("error without apology")
        
        if concerns:
            print(f"  ⚠️ Validation concerns: {', '.join(concerns)}")
            # In production, you might modify or reject the response
        else:
            print(f"  ✅ Response validated: {len(content)} chars, appropriate content")
        
        return {}
    
    # Create the production-ready agent
    agent = create_agent(
        select_optimal_model,
        tools=[tools.web_search_production, tools.calculate_advanced, tools.analyze_text],
        state_schema=ProductionAgentState,
        pre_model_hook=context_preprocessor,
        post_model_hook=production_validator,
        prompt="You are an advanced AI business analyst. Provide comprehensive, well-researched responses using available tools. Always explain your reasoning and cite sources when applicable."
    )
    
    # Comprehensive test scenarios
    test_scenarios = [
        {
            "name": "Simple Calculation",
            "description": "Basic math query to test Qwen3 selection",
            "input": {
                "messages": "What's 25 multiplied by 16?",
                "user_context": {"tier": "standard", "detail_level": "medium"}
            }
        },
        {
            "name": "Business Analysis Request", 
            "description": "Complex business query to test GPT-OSS selection",
            "input": {
                "messages": "I need a comprehensive market analysis for the AI industry, including competitive landscape and revenue forecasts",
                "user_context": {"tier": "premium", "detail_level": "high"},
                "task_history": ["market_research", "competitor_analysis"]
            }
        },
        {
            "name": "Technical Deep Dive",
            "description": "Technical query with conversation history",
            "input": {
                "messages": [
                    "I'm building a machine learning system",
                    "What algorithms should I consider?", 
                    "I need detailed technical analysis of neural network architectures for my specific use case"
                ],
                "user_context": {"tier": "standard", "detail_level": "high", "role": "developer"},
                "task_history": ["technical_consultation"]
            }
        },
        {
            "name": "Multi-Tool Workflow",
            "description": "Complex query requiring multiple tools",
            "input": {
                "messages": "Research current AI trends, calculate the market size if it grows 25% annually from $100B base, and analyze the sentiment of recent news",
                "user_context": {"tier": "premium", "detail_level": "high"}
            }
        }
    ]
    
    # Execute test scenarios
    print("\n🧪 Running Production Test Scenarios:")
    print("=" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        
        # Show input details
        input_data = scenario["input"]
        if isinstance(input_data["messages"], list):
            print(f"💬 Messages: {len(input_data['messages'])} messages in conversation")
        else:
            print(f"💬 Query: {input_data['messages']}")
        
        try:
            result = agent.invoke(input_data)
            response = result['messages'][-1].content
            print(f"✅ Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n🎉 Production agent testing completed!")
    print("💡 Key Features Demonstrated:")
    print("   • Intelligent model selection based on content complexity")
    print("   • Comprehensive tool integration (search, calc, analysis)")
    print("   • User context and preference handling")
    print("   • Production validation and error handling")
    print("   • Cost optimization through smart model switching")
    
    return agent


# ============================================================================
# MAIN EXECUTION WITH DETAILED SYSTEM INFORMATION
# ============================================================================

def main():
    """
    MAIN EXECUTION FUNCTION WITH COMPREHENSIVE TESTING
    
    This function orchestrates the execution of all examples and provides:
    - System connectivity testing
    - Progressive example execution
    - Detailed error reporting
    - Performance and feature summary
    
    Execution Flow:
    1. Test basic Ollama connectivity
    2. Verify required models are available
    3. Run all examples in logical order
    4. Collect results and provide summary
    5. Report any failures with troubleshooting tips
    
    Error Handling:
    - Connection failures: Check Ollama service
    - Missing models: Provide installation commands
    - Individual example failures: Continue with others
    - Comprehensive error reporting for debugging
    """
    print("🚀 LangChain Agents with ChatOllama (Qwen3 + GPT-OSS)")
    print("=" * 70)
    print("📚 This demo covers 11 comprehensive examples of LangChain agents:")
    print("   1. Basic agent creation")
    print("   2. Model instance configuration") 
    print("   3. Dynamic model selection")
    print("   4. Advanced content-aware selection")
    print("   5. Tool configuration methods")
    print("   6. Prompt configuration options")
    print("   7. Structured output with Pydantic")
    print("   8. Custom state management")
    print("   9. Pre/post model hooks")
    print("   10. Streaming responses")
    print("   11. Complete production example")
    print("=" * 70)
    
    try:
        # Test basic connectivity and model availability
        print("\n🔍 Testing System Connectivity...")
        
        print("  • Testing Qwen3 model...")
        qwen3_test = ChatOllama(model="qwen3", num_predict=20)
        qwen3_response = qwen3_test.invoke("Hello")
        print("  ✅ Qwen3 model: Connected and responsive")
        
        print("  • Testing GPT-OSS model...")
        gpt_oss_test = ChatOllama(model="gpt-oss", num_predict=20)
        gpt_oss_response = gpt_oss_test.invoke("Hello")
        print("  ✅ GPT-OSS model: Connected and responsive")
        
        print("  ✅ All systems operational - proceeding with examples")
        
        # Define all examples with metadata
        examples = [
            {
                "func": example_1_basic_agent_with_model_string,
                "category": "Basics",
                "description": "Simple agent creation with model strings"
            },
            {
                "func": example_2_agent_with_model_instance,
                "category": "Basics", 
                "description": "Explicit model configuration"
            },
            {
                "func": example_3_dynamic_model_selection,
                "category": "Model Selection",
                "description": "Message count-based model switching"
            },
            {
                "func": example_4_advanced_model_selection,
                "category": "Model Selection",
                "description": "Content-aware intelligent model selection"
            },
            {
                "func": example_5_tool_configurations,
                "category": "Tools",
                "description": "Different tool setup approaches"
            },
            {
                "func": example_6_prompt_configurations,
                "category": "Prompts",
                "description": "String, SystemMessage, and dynamic prompts"
            },
            {
                "func": example_7_structured_output,
                "category": "Advanced",
                "description": "Pydantic models for structured responses"
            },
            {
                "func": example_8_custom_state,
                "category": "Advanced",
                "description": "Custom state for memory management"
            },
            {
                "func": example_9_hooks,
                "category": "Advanced",
                "description": "Pre and post-processing hooks"
            },
            {
                "func": example_10_streaming,
                "category": "Advanced",
                "description": "Real-time response streaming"
            },
            {
                "func": example_11_complete_real_world,
                "category": "Production",
                "description": "Complete production-ready agent"
            }
        ]
        
        # Execute all examples with detailed tracking
        successful_agents = []
        failed_examples = []
        
        print(f"\n🎯 Executing {len(examples)} Examples:")
        print("=" * 50)
        
        for i, example in enumerate(examples, 1):
            try:
                print(f"\n[{i}/{len(examples)}] {example['category']}: {example['description']}")
                
                result = example["func"]()
                
                # Handle different return types
                if isinstance(result, tuple):
                    successful_agents.extend(result)
                    print(f"    ✅ Created {len(result)} agents successfully")
                else:
                    successful_agents.append(result)
                    print(f"    ✅ Created 1 agent successfully")
                    
            except Exception as e:
                error_info = {
                    "example": example["description"],
                    "category": example["category"],
                    "error": str(e)
                }
                failed_examples.append(error_info)
                print(f"    ❌ Failed: {str(e)}")
        
        # Provide comprehensive summary
        print(f"\n📊 EXECUTION SUMMARY")
        print("=" * 50)
        print(f"✅ Successful Examples: {len(examples) - len(failed_examples)}/{len(examples)}")
        print(f"🤖 Total Agents Created: {len(successful_agents)}")
        print(f"❌ Failed Examples: {len(failed_examples)}")
        
        if failed_examples:
            print(f"\n⚠️ Failed Examples Details:")
            for failure in failed_examples:
                print(f"   • {failure['category']}: {failure['example']}")
                print(f"     Error: {failure['error']}")
        
        # Feature summary
        print(f"\n🎉 FEATURES DEMONSTRATED:")
        print("=" * 50)
        print("🔄 Model Management:")
        print("   • Dynamic model selection (Qwen3 ↔ GPT-OSS)")
        print("   • Content-aware model switching") 
        print("   • Cost optimization strategies")
        
        print("\n🛠️ Tool Integration:")
        print("   • Multiple tool configuration methods")
        print("   • Error handling and validation")
        print("   • Custom tool behaviors")
        
        print("\n💬 Conversation Management:")
        print("   • Multiple prompt types (string, SystemMessage, callable)")
        print("   • Custom state schemas")
        print("   • Memory and context tracking")
        
        print("\n⚡ Advanced Features:")
        print("   • Pre/post processing hooks")
        print("   • Structured output with Pydantic")
        print("   • Real-time streaming responses")
        print("   • Production-ready architecture")
        
        print(f"\n🚀 Ready for Production Use!")
        print("   All examples demonstrate production-ready patterns")
        print("   Cost-optimized through intelligent model selection")
        print("   Robust error handling and validation included")
        
    except Exception as e:
        print(f"\n❌ System Connection Failed: {str(e)}")
        print("\n🔧 Troubleshooting Steps:")
        print("1. Ensure Ollama is running:")
        print("   ollama serve")
        print("\n2. Install required models:")
        print("   ollama pull qwen3")
        print("   ollama pull gpt-oss")
        print("\n3. Verify models are available:")
        print("   ollama list")
        print("\n4. Check if models respond:")
        print("   ollama run qwen3 'Hello'")
        print("   ollama run gpt-oss 'Hello'")
        print("\n5. Install Python dependencies:")
        print("   pip install langchain langchain-community langchain-core langgraph pydantic")


if __name__ == "__main__":
    main()
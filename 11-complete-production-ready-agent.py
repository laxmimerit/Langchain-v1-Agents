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

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama pull gpt-oss
ollama serve
"""

from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.runtime import Runtime
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
import tools

# Load environment variables from .env file
load_dotenv()

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

if __name__ == "__main__":
    example_11_complete_real_world()
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

Prerequisites:
pip install langchain langchain-community langchain-core langgraph pydantic
ollama pull qwen3
ollama serve
"""

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import tools

# Load environment variables from .env file
load_dotenv()

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

if __name__ == "__main__":
    example_7_structured_output()
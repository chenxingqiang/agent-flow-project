"""
Chat Assistant Example using ELL
Demonstrates multi-turn conversations and tool usage
"""

import os
import ell
import asyncio
from typing import List, Dict, Any
from datetime import datetime

# Initialize ELL with API key and versioning
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

# Initialize ELL with versioning
ell2a.init(store='./ell2a_logs')

# Define tools
@ell2a.tool()
async def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@ell2a.tool()
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information."""
    # Simulate knowledge base search
    kb = {
        "python": "Python is a high-level programming language.",
        "ell2a": "ELL is a Language Model Programming Library.",
        "llm": "LLM stands for Large Language Model."
    }
    return kb.get(query.lower(), "No information found.")

# Define chat assistant
@ell2a.complex(
    model="gpt-4",
    tools=[get_current_time, search_knowledge_base],
    temperature=0.7
)
async def chat_assistant(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None
) -> List[ell2a.Message]:
    """An intelligent assistant that can engage in conversations and use tools."""
    # Initialize conversation history if None
    if conversation_history is None:
        conversation_history = []
    
    # Build messages list
    messages = [
        ell2a.system(
            "You are a helpful assistant that can engage in natural conversations "
            "and use tools to provide accurate information. "
            "Always be concise and friendly."
        )
    ]
    
    # Add conversation history
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(ell2a.user(msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ell2a.assistant(msg["content"]))
    
    # Add current user message
    messages.append(ell2a.user(user_message))
    
    # Return messages list
    return messages

async def interactive_chat():
    """Run an interactive chat session."""
    conversation_history = []
    print("Chat Assistant: Hello! How can I help you today? (type 'exit' to end)")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("\nChat Assistant: Goodbye!")
                break
            
            # Get assistant response
            messages = await chat_assistant(user_input, conversation_history)
            response = await ell2a.complete(messages)
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Process response
            if response and response.text:
                print(f"\nChat Assistant: {response.text}")
                conversation_history.append({"role": "assistant", "content": response.text})
                
                # Handle tool calls
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        if tool_name == "get_current_time":
                            result = await get_current_time()
                            print(f"[Tool] Current time: {result}")
                        elif tool_name == "search_knowledge_base":
                            query = tool_call.function.arguments.get("query", "")
                            result = await search_knowledge_base(query)
                            print(f"[Tool] Knowledge base: {result}")
            else:
                print("\nChat Assistant: I apologize, but I encountered an error processing your request.")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Chat Assistant: I apologize for the error. Please try again.")

if __name__ == "__main__":
    print("Starting chat assistant...")
    asyncio.run(interactive_chat()) 
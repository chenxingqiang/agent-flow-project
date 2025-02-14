from agentflow.ell2a.integration import ELL2AIntegration
from agentflow.ell2a.types.message import Message, MessageRole

# Get singleton instance
ell2a = ELL2AIntegration()

@ell2a.with_ell2a(mode="simple")
async def hello_from_claude() -> str:
    """Say hello using Claude."""
    # Create message
    message = Message(
        role=MessageRole.USER,
        content="Say hello to the world!!!",
        metadata={
            "type": "text",
            "format": "plain"
        }
    )
    
    # Process message
    response = await ell2a.process_message(message)
    
    # Return the response
    if isinstance(response, Message):
        return str(response.content)
    elif isinstance(response, dict):
        return str(response.get("content", ""))
    else:
        return str(response)

if __name__ == "__main__":
    # Initialize ELL2A
    ell2a.configure({
        "enabled": True,
        "tracking_enabled": True,
        "store": "./logdir",
        "verbose": True,
        "autocommit": True,
        "model": "claude-3-sonnet-20240229",
        "default_model": "claude-3-sonnet-20240229",
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    # Run the example
    import asyncio
    response = asyncio.run(hello_from_claude())
    print("\nClaude's Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)


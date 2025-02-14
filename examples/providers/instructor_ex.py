"""
OpenAI example with structured output
This example demonstrates extracting structured data from text using OpenAI.
"""
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import os
import json

class UserDetail(BaseModel):
    """Details about a user extracted from text."""
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")
    occupation: Optional[str] = Field(None, description="The user's occupation if mentioned")
    hobbies: list[str] = Field(default_factory=list, description="List of user's hobbies if mentioned")

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

try:
    # Get OpenAI API key
    api_key = get_env_var("OPENAI_API_KEY")
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- OPENAI_API_KEY")
    exit(1)

async def extract_user_details(text: str) -> str:
    """Extract user details from text using OpenAI."""
    try:
        # Create the extraction prompt with function calling
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts user details from text."
                },
                {
                    "role": "user",
                    "content": f"Extract user details from the following text:\n\n{text}"
                }
            ],
            functions=[{
                "name": "extract_user_details",
                "description": "Extract user details from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The user's name"
                        },
                        "age": {
                            "type": "integer",
                            "description": "The user's age"
                        },
                        "occupation": {
                            "type": "string",
                            "description": "The user's occupation if mentioned"
                        },
                        "hobbies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of user's hobbies if mentioned"
                        }
                    },
                    "required": ["name", "age"]
                }
            }],
            function_call={"name": "extract_user_details"}
        )
        
        # Extract the function call arguments
        if response.choices and response.choices[0].message.function_call:
            args = json.loads(response.choices[0].message.function_call.arguments)
            user = UserDetail(**args)
            return json.dumps(user.model_dump(), indent=2)
        
        return "No details extracted."
    except Exception as e:
        print(f"Error extracting details: {e}")
        return "Error extracting user details."

async def main():
    # Example texts
    examples = [
        "John is a 32-year-old software engineer who loves hiking and photography.",
        "Sarah, 28, works as a teacher and enjoys reading in her free time.",
        "Mike is 45 years old and runs his own restaurant. He's passionate about cooking and gardening."
    ]
    
    print("\n=== User Detail Extraction Example ===")
    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        details = await extract_user_details(text)
        print(f"Extracted Details:\n{details}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



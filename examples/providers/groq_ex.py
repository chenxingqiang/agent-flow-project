"""
Groq example: pip install ell-ai[groq]
This example demonstrates using Groq to generate stories.
"""
import groq
import os
import asyncio
from typing import Optional

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

try:
    # Get Groq API key
    api_key = get_env_var("GROQ_API_KEY")
    
    # Create Groq client
    groq_client = groq.Groq(api_key=api_key)

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- GROQ_API_KEY")
    exit(1)

async def write_a_story(about: str) -> Optional[str]:
    """Write a story about the given topic."""
    try:
        # Create chat completion
        completion = groq_client.chat.completions.create(
            model="llama2-70b-4096",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative writer. Your task is to write engaging and imaginative stories."
                },
                {
                    "role": "user",
                    "content": f"""Write a short, engaging story about {about}. 
The story should be imaginative and entertaining, with a clear beginning, middle, and end.
Keep the story between 150-200 words."""
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Return the story
        return completion.choices[0].message.content if completion.choices else None
    except Exception as e:
        print(f"Error generating story: {e}")
        return None

async def main():
    # Example: Write a story about cats
    print("\n=== Story Generation Example ===")
    story_topic = "cats"
    print(f"Writing a story about: {story_topic}")
    story = await write_a_story(story_topic)
    print(f"\nStory:\n{story if story else 'No story generated.'}")

if __name__ == "__main__":
    asyncio.run(main())


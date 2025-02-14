from typing import Optional
import openai
import os

# Create OpenAI client for Ollama
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but not used
)

async def random_number() -> str:
    """Generate a random number."""
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="llama2",  # Using llama2 which should be available locally
            messages=[
                {
                    "role": "system",
                    "content": """You are a number generator. Your task is to generate a random number between 1 and 10.
Rules:
1. Only output the number
2. No additional text or explanations
3. Just the number, nothing else"""
                },
                {
                    "role": "user",
                    "content": "Generate a random number."
                }
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        # Return the response
        content = response.choices[0].message.content if response.choices else None
        return content if content is not None else "5"  # Default to 5 if no response
    except Exception as e:
        print(f"Error generating number: {e}")
        return "5"  # Default to 5 on error

async def write_a_poem(num: str) -> str:
    """Write a poem with the given number of lines."""
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="llama2",  # Using llama2 which should be available locally
            messages=[
                {
                    "role": "system",
                    "content": """You are a creative poet. Your task is to write poems with a specified number of lines.
Rules:
1. Write exactly the requested number of lines
2. Make it creative and engaging
3. Use vivid imagery and metaphors
4. Each line should be meaningful and contribute to the whole"""
                },
                {
                    "role": "user",
                    "content": f"Write a poem that is {num} lines long."
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Return the response
        content = response.choices[0].message.content if response.choices else None
        return content if content is not None else "Error: Could not generate poem"
    except Exception as e:
        print(f"Error generating poem: {e}")
        return "Error: Could not generate poem"

async def write_a_story(num: str) -> str:
    """Write a story with the given number of lines."""
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="llama2",  # Using llama2 which should be available locally
            messages=[
                {
                    "role": "system",
                    "content": """You are a creative storyteller. Your task is to write stories with a specified number of lines.
Rules:
1. Write exactly the requested number of lines
2. Make it engaging and memorable
3. Include a clear beginning, middle, and end
4. Use descriptive language and dialogue where appropriate"""
                },
                {
                    "role": "user",
                    "content": f"Write a story that is {num} lines long."
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Return the response
        content = response.choices[0].message.content if response.choices else None
        return content if content is not None else "Error: Could not generate story"
    except Exception as e:
        print(f"Error generating story: {e}")
        return "Error: Could not generate story"

async def choose_which_is_a_better_piece_of_writing(poem: str, story: str) -> str:
    """Choose which piece of writing is better."""
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="llama2",  # Using llama2 which should be available locally
            messages=[
                {
                    "role": "system",
                    "content": """You are a literary critic. Your task is to analyze and compare pieces of writing.
Rules:
1. Consider the following aspects:
   - Creativity and originality
   - Use of language and imagery
   - Structure and flow
   - Impact and memorability
2. Provide a clear analysis of both pieces
3. Choose which one is better and explain why"""
                },
                {
                    "role": "user",
                    "content": f"""Compare these two pieces of writing:

A (Poem):
{poem}

B (Story):
{story}

Which piece is better and why?"""
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Return the response
        content = response.choices[0].message.content if response.choices else None
        return content if content is not None else "Error: Could not analyze writing"
    except Exception as e:
        print(f"Error analyzing writing: {e}")
        return "Error: Could not analyze writing"

if __name__ == "__main__":
    # Run the example
    import asyncio
    
    async def main():
        # Get random number
        num = await random_number()
        print("\nRandom Number:", num)
        
        # Generate poem and story
        poem = await write_a_poem(num)
        story = await write_a_story(num)
        
        print("\nPoem:")
        print("-" * 50)
        print(poem)
        
        print("\nStory:")
        print("-" * 50)
        print(story)
        
        # Compare them
        result = await choose_which_is_a_better_piece_of_writing(poem, story)
        print("\nAnalysis:")
        print("-" * 50)
        print(result)
    
    asyncio.run(main())
    


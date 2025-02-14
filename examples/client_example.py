from typing import Optional
import openai
import os

# Create OpenAI client for Ollama
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but not used
)

async def number_to_words(number: int) -> Optional[str]:
    """Convert a number to its word representation."""
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="llama2",  # Using llama2 which should be available locally
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in converting numbers to their word representation in English.
Your task is to convert numbers to their word representation, following these rules:
1. Only output the word representation
2. Use hyphens for compound numbers (twenty-three, forty-five)
3. Use 'and' after hundreds (one hundred and twenty-three)
4. No additional text or explanations
5. No punctuation at the end

Examples:
123 -> one hundred and twenty-three
1000 -> one thousand
1234567 -> one million two hundred and thirty-four thousand five hundred and sixty-seven"""
                },
                {
                    "role": "user",
                    "content": str(number)
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        # Return the response
        content = response.choices[0].message.content if response.choices else None
        if content:
            # Clean up the response by removing any extra text and whitespace
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('Sure', 'Here', 'The')):
                    return line
        return None
    except Exception as e:
        print(f"Error generating word representation: {e}")
        return None

if __name__ == "__main__":
    # Run the example
    import asyncio
    response = asyncio.run(number_to_words(123456))
    print("\nNumber in Words:")
    print("-" * 50)
    print(response if response else "Error: Could not generate word representation")
    print("-" * 50)
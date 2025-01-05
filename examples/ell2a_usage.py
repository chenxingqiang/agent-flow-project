"""
Example usage of ELL (Extensible Language Layer)
"""

import os
import ell
from typing import List
import asyncio
from PIL import Image

# Initialize ELL with API key and versioning
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

# Initialize ELL with versioning
ell2a.init(store='./ell2a_logs')

# Simple prompt example
@ell2a.simple(model="gpt-4")
async def summarize(text: str) -> str:
    """You are a helpful assistant that summarizes text concisely."""
    return f"Please summarize this text in one sentence: {text}"

# Complex prompt with tools example
@ell2a.tool()
async def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulate web search
    return f"Search results for: {query}"

@ell2a.complex(model="gpt-4", tools=[search_web])
async def research_topic(topic: str) -> List[ell2a.Message]:
    """Research a given topic using web search."""
    messages = [
        ell2a.system("You are a research assistant that uses web search to find information."),
        ell2a.user(f"Research this topic and provide key findings: {topic}")
    ]
    return messages

# Multimodal example
@ell2a.complex(model="gpt-4-vision-preview")
async def analyze_image(image: Image.Image) -> List[ell2a.Message]:
    """Analyze the given image."""
    messages = [
        ell2a.system("You are a computer vision expert. Analyze the image and describe what you see."),
        ell2a.user(["Describe this image in detail:", image])
    ]
    return messages

async def main():
    try:
        # Test simple prompt
        print("\nTesting simple prompt...")
        summary = await summarize(
            "The quick brown fox jumps over the lazy dog. "
            "This pangram contains every letter of the English alphabet."
        )
        print("Summary:", summary)
        
        # Test complex prompt with tools
        print("\nTesting complex prompt with tools...")
        research = await research_topic("artificial intelligence trends 2024")
        print("Research Results:", research.text)
        if research.tool_calls:
            print("Tool Calls:", research.tool_calls)
        
        # Test multimodal (commented out as it requires an image)
        # print("\nTesting multimodal...")
        # image = Image.open("example.jpg")
        # analysis = await analyze_image(image)
        # print("Image Analysis:", analysis.text)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting ELL examples...")
    asyncio.run(main()) 
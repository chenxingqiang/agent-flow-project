import random
from agentflow.ell2a_integration import ELL2AIntegration

# Get singleton instance
ell2a = ELL2AIntegration()

# Initialize ELL2A
ell2a.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./ell2a_logs",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.1,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

def get_random_adjective():
    adjectives = ["enthusiastic", "cheerful", "warm", "friendly", "heartfelt", "sincere"]
    return random.choice(adjectives)

def get_random_punctuation():
    return random.choice(["!", "!!", "!!!"])

@ell2a.with_ell2a(mode="simple")
async def hello(name: str) -> str:
    """You are a helpful and expressive assistant."""
    adjective = get_random_adjective()
    punctuation = get_random_punctuation()
    return f"Say a {adjective} hello to {name}{punctuation}"

if __name__ == "__main__":
    import asyncio
    greeting = asyncio.run(hello("Sam Altman"))
    print(greeting)
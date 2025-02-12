"""Test DeepSeek provider integration."""

import os
import pytest
from agentflow.ell2a.providers.deepseek import DeepSeekProvider, DeepSeekConfig

# Get API key from environment
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

@pytest.fixture
def chat_provider():
    """Create a DeepSeek-V3 provider instance."""
    if not DEEPSEEK_API_KEY:
        pytest.skip("No DeepSeek API key provided")
    config = {
        "api_key": DEEPSEEK_API_KEY,
        "model": "deepseek-chat-v1",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    return DeepSeekProvider(config=config)

@pytest.fixture
def reasoner_provider():
    """Create a DeepSeek-R1 provider instance."""
    if not DEEPSEEK_API_KEY:
        pytest.skip("No DeepSeek API key provided")
    config = {
        "api_key": DEEPSEEK_API_KEY,
        "model": "deepseek-reasoner-v1",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    return DeepSeekProvider(config=config)

@pytest.mark.asyncio
async def test_provider_initialization(chat_provider):
    """Test provider initialization."""
    await chat_provider.initialize()
    assert chat_provider._session is not None
    assert "Authorization" in chat_provider.session.headers
    assert "Content-Type" in chat_provider.session.headers
    await chat_provider.cleanup()

@pytest.mark.asyncio
async def test_chat_completion_v3(chat_provider):
    """Test chat completion with DeepSeek-V3."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = await chat_provider.chat(messages)
    assert isinstance(response, str)
    assert len(response) > 0
    # The response may contain markdown formatting
    assert "paris" in response.lower().replace("*", "")

@pytest.mark.asyncio
async def test_chat_completion_r1(reasoner_provider):
    """Test chat completion with DeepSeek-R1."""
    messages = [
        {"role": "system", "content": "You are a logical reasoning assistant."},
        {"role": "user", "content": "If all cats are mammals, and all mammals are animals, what can we conclude about cats?"}
    ]
    
    response = await reasoner_provider.chat(messages)
    assert isinstance(response, str)
    assert len(response) > 0
    # Check for logical reasoning in response
    assert any(word in response.lower() for word in ["conclude", "therefore", "logic", "animal"])

@pytest.mark.asyncio
async def test_text_generation_v3(chat_provider):
    """Test text generation with DeepSeek-V3."""
    prompt = "Write a short poem about artificial intelligence."
    
    response = await chat_provider.generate(prompt, use_chat_endpoint=True)
    assert isinstance(response, str)
    assert len(response) > 0
    # Check if the response contains any AI-related keywords
    ai_keywords = ["ai", "artificial", "intelligence", "machine", "neural", "digital", "code", "learn"]
    assert any(keyword in response.lower() for keyword in ai_keywords), "Response should contain AI-related keywords"

@pytest.mark.asyncio
async def test_text_generation_r1(reasoner_provider):
    """Test text generation with DeepSeek-R1."""
    prompt = "Explain the logical steps to solve this problem: If A implies B, and B implies C, what can we say about A and C?"
    
    response = await reasoner_provider.generate(prompt, use_chat_endpoint=True)
    assert isinstance(response, str)
    assert len(response) > 0
    # Check for logical reasoning terms
    reasoning_keywords = ["therefore", "implies", "conclusion", "logic", "transitive"]
    assert any(keyword in response.lower() for keyword in reasoning_keywords), "Response should contain reasoning keywords"

@pytest.mark.asyncio
@pytest.mark.skip(reason="Embeddings API not available")
async def test_embeddings(chat_provider):
    """Test embeddings generation."""
    text = "This is a test sentence for embeddings."
    
    embeddings = await chat_provider.embed(text)
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
    assert all(isinstance(x, float) for x in embeddings)

@pytest.mark.asyncio
async def test_error_handling(chat_provider):
    """Test error handling with invalid API key."""
    # Create provider with invalid API key
    invalid_config = {
        "api_key": "invalid-key",
        "model": "deepseek-chat-v1"
    }
    invalid_provider = DeepSeekProvider(config=invalid_config)
    
    with pytest.raises(Exception):
        await invalid_provider.chat([{"role": "user", "content": "Hello"}])

@pytest.mark.asyncio
async def test_provider_cleanup(chat_provider):
    """Test provider cleanup."""
    await chat_provider.initialize()
    assert chat_provider._session is not None
    
    await chat_provider.cleanup()
    assert chat_provider._session is None 
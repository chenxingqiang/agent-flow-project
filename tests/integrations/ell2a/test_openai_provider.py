"""Test OpenAI provider module."""

import pytest
from typing import List, Dict, Any

from agentflow.ell2a.providers.openai import OpenAIProvider, _content_block_to_openai_format
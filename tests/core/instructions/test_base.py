"""Tests for base instruction functionality."""
import pytest
from agentflow.core.instructions.base import (
    BaseInstruction,
    InstructionResult,
    InstructionStatus,
    InstructionMetrics
)
from typing import Dict, Any

@pytest.fixture
def sample_context():
    """Create sample execution context."""
    return {
        "variables": {"test": "value"},
        "resources": {},
        "state": "ready"
    }

class SimpleInstruction(BaseInstruction):
    """Test instruction implementation."""
    
    def __init__(self, name: str, description: str, should_fail: bool = False):
        super().__init__(name, description)
        self.should_fail = should_fail
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.should_fail:
            raise Exception("Test error")
        return {"output": "test_output"}
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        return await super().execute(context)

@pytest.mark.asyncio
class TestBaseInstruction:
    """Test base instruction functionality."""
    
    async def test_initialization(self):
        """Test instruction initialization."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        assert instr.name == "test"
        assert instr.description == "test description"
        assert instr.status == InstructionStatus.PENDING
        assert instr.metrics is None
        
    async def test_execution(self, sample_context):
        """Test instruction execution."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        result = await instr.execute(sample_context)
        assert isinstance(result, InstructionResult)
        assert result.status == InstructionStatus.COMPLETED
        assert result.data == {"output": "test_output"}
        assert result.error is None
        assert isinstance(result.metrics, InstructionMetrics)
        
    async def test_error_handling(self, sample_context):
        """Test error handling during execution."""
        instr = SimpleInstruction(
            name="test",
            description="test description",
            should_fail=True
        )
        
        result = await instr.execute(sample_context)
        assert result.status == InstructionStatus.FAILED
        assert result.error == "Test error"
        assert isinstance(result.metrics, InstructionMetrics)
        
    async def test_hooks(self, sample_context):
        """Test pre and post execution hooks."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        pre_hook_called = False
        post_hook_called = False
        
        async def pre_hook(instruction, ctx):
            nonlocal pre_hook_called
            pre_hook_called = True
            
        async def post_hook(instruction, ctx):
            nonlocal post_hook_called
            post_hook_called = True
            
        instr.add_pre_hook(pre_hook)
        instr.add_post_hook(post_hook)
        
        await instr.execute(sample_context)
        assert pre_hook_called
        assert post_hook_called
        
    async def test_metrics(self, sample_context):
        """Test metrics tracking."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        result = await instr.execute(sample_context)
        metrics = result.metrics
        
        assert metrics is not None
        assert metrics.start_time > 0
        assert metrics.end_time > metrics.start_time
        assert metrics.execution_time > 0
        assert metrics.tokens_used > 0
        assert metrics.memory_used > 0
        assert not metrics.cache_hit
        assert not metrics.optimization_applied
        assert not metrics.parallel_execution

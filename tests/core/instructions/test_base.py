"""Tests for base instruction functionality."""
import pytest
from agentflow.core.instructions.base import (
    BaseInstruction,
    InstructionResult,
    InstructionStatus,
    InstructionMetrics
)
from typing import Dict, Any
import asyncio

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
    
    def __init__(self, name: str, description: str, should_fail: bool = False, delay: float = 0):
        super().__init__(name, description)
        self.should_fail = should_fail
        self.delay = delay
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
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
        assert len(instr._pre_hooks) == 0
        assert len(instr._post_hooks) == 0
        
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
        
        # Check status transitions
        assert instr.status == InstructionStatus.COMPLETED
        
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
        
        # Check status transitions
        assert instr.status == InstructionStatus.FAILED
        
    async def test_hooks(self, sample_context):
        """Test pre and post execution hooks."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        pre_hook_called = False
        post_hook_called = False
        hook_context = None
        
        async def pre_hook(instruction, ctx):
            nonlocal pre_hook_called, hook_context
            pre_hook_called = True
            hook_context = ctx
            assert instruction.status == InstructionStatus.PENDING
            
        async def post_hook(instruction, ctx):
            nonlocal post_hook_called
            post_hook_called = True
            assert instruction.status == InstructionStatus.RUNNING
            
        instr.add_pre_hook(pre_hook)
        instr.add_post_hook(post_hook)
        
        await instr.execute(sample_context)
        assert pre_hook_called
        assert post_hook_called
        assert hook_context == sample_context
        
    async def test_metrics_tracking(self, sample_context):
        """Test instruction metrics tracking."""
        instr = SimpleInstruction(
            name="test",
            description="test description",
            delay=0.1  # Add small delay to ensure measurable execution time
        )
        
        result = await instr.execute(sample_context)
        metrics = result.metrics
        
        assert isinstance(metrics, InstructionMetrics)
        assert metrics.start_time > 0
        assert metrics.end_time > metrics.start_time
        assert metrics.execution_time > 0
        assert metrics.tokens_used == 0  # Placeholder value
        assert metrics.memory_used == 0  # Placeholder value
        assert not metrics.cache_hit
        assert not metrics.optimization_applied
        assert not metrics.parallel_execution
        
    async def test_value_error_handling(self, sample_context):
        """Test specific handling of ValueError."""
        class ValueErrorInstruction(SimpleInstruction):
            async def _execute_impl(self, context):
                raise ValueError("Invalid value")
                
        instr = ValueErrorInstruction(
            name="test",
            description="test description"
        )
        
        result = await instr.execute(sample_context)
        assert result.status == InstructionStatus.FAILED
        assert result.error == "Invalid value"
        assert isinstance(result.metrics, InstructionMetrics)
        
    async def test_hook_error_handling(self, sample_context):
        """Test error handling in hooks."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        async def failing_pre_hook(instruction, ctx):
            raise Exception("Pre-hook error")
            
        instr.add_pre_hook(failing_pre_hook)
        
        result = await instr.execute(sample_context)
        assert result.status == InstructionStatus.FAILED
        assert "Pre-hook error" in result.error
        assert isinstance(result.metrics, InstructionMetrics)
        
    async def test_multiple_hooks(self, sample_context):
        """Test execution with multiple hooks."""
        instr = SimpleInstruction(
            name="test",
            description="test description"
        )
        
        hook_execution_order = []
        
        async def pre_hook1(instruction, ctx):
            hook_execution_order.append("pre1")
            
        async def pre_hook2(instruction, ctx):
            hook_execution_order.append("pre2")
            
        async def post_hook1(instruction, ctx):
            hook_execution_order.append("post1")
            
        async def post_hook2(instruction, ctx):
            hook_execution_order.append("post2")
            
        instr.add_pre_hook(pre_hook1)
        instr.add_pre_hook(pre_hook2)
        instr.add_post_hook(post_hook1)
        instr.add_post_hook(post_hook2)
        
        await instr.execute(sample_context)
        
        # Verify hooks executed in correct order
        assert hook_execution_order == ["pre1", "pre2", "post1", "post2"]

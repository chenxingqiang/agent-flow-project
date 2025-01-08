"""Test agent module."""

import pytest
import numpy as np
from agentflow.core.config import (
    AgentConfig, ConfigurationType, AgentMode, ModelConfig,
    WorkflowConfig, WorkflowStep, StepConfig, WorkflowStepType
)
from agentflow.core.exceptions import WorkflowExecutionError

class TestAgentFramework:
    """Test agent framework functionality."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        """Set up test environment."""
        self.workflow_config = WorkflowConfig(
            id="test-workflow-1",
            name="test_workflow",
            max_iterations=5,
            timeout=30,
            steps=[
                WorkflowStep(
                    id="step-1",
                    name="step_1",
                    type=WorkflowStepType.TRANSFORM,
                    config=StepConfig(
                        strategy="feature_engineering",
                        params={
                            "method": "standard",
                            "with_mean": True,
                            "with_std": True
                        }
                    )
                )
            ]
        )
        self.agent_config = AgentConfig(
            id="test-agent-1",
            name="test_agent",
            type=ConfigurationType.DATA_SCIENCE,
            mode=AgentMode.SIMPLE,
            version="1.0.0",
            model=ModelConfig(
                provider="openai",
                name="gpt-4",
                temperature=0.7,
                max_tokens=4096
            ),
            workflow=self.workflow_config
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        print("Workflow type:", type(self.agent_config.workflow))
        print("Workflow:", self.agent_config.workflow)
        assert self.agent_config.id == "test-agent-1"
        assert self.agent_config.name == "test_agent"
        assert self.agent_config.type == ConfigurationType.DATA_SCIENCE
        assert self.agent_config.mode == AgentMode.SIMPLE
        assert self.agent_config.version == "1.0.0"
        assert isinstance(self.agent_config.model, ModelConfig)
        assert self.agent_config.model.name == "gpt-4"
        assert isinstance(self.agent_config.workflow, WorkflowConfig)

    @pytest.mark.asyncio
    async def test_agent_workflow_execution(self):
        """Test agent workflow execution."""
        data = np.random.randn(10, 2)
        result = await self.agent_config.workflow.execute({"data": data})
        assert "step-1" in result
        assert "output" in result["step-1"]
        assert "data" in result["step-1"]["output"]
        assert isinstance(result["step-1"]["output"]["data"], np.ndarray)
        assert result["step-1"]["output"]["data"].shape == data.shape

    @pytest.mark.asyncio
    async def test_data_science_agent_advanced_transformations(self):
        """Test data science agent with advanced transformations."""
        workflow = WorkflowConfig(
            id="advanced-workflow-1",
            name="advanced_workflow",
            max_iterations=5,
            timeout=30,
            steps=[
                WorkflowStep(
                    id="step-1",
                    name="step_1",
                    type=WorkflowStepType.TRANSFORM,
                    config=StepConfig(
                        strategy="feature_engineering",
                        params={
                            "method": "standard",
                            "with_mean": True,
                            "with_std": True
                        }
                    )
                ),
                WorkflowStep(
                    id="step-2",
                    name="step_2",
                    type=WorkflowStepType.TRANSFORM,
                    config=StepConfig(
                        strategy="outlier_removal",
                        params={
                            "method": "isolation_forest",
                            "threshold": 0.1
                        }
                    )
                )
            ]
        )
        agent = AgentConfig(
            id="test-agent-2",
            name="test_agent",
            type=ConfigurationType.DATA_SCIENCE,
            mode=AgentMode.SIMPLE,
            version="1.0.0",
            model=ModelConfig(
                provider="openai",
                name="gpt-4",
                temperature=0.7,
                max_tokens=4096
            ),
            workflow=workflow
        )
        data = np.random.randn(100, 2)  # More data points for outlier detection
        result = await agent.workflow.execute({"data": data})
        assert "step-1" in result
        assert "step-2" in result
        assert "output" in result["step-2"]
        assert "data" in result["step-2"]["output"]
        assert isinstance(result["step-2"]["output"]["data"], np.ndarray)
        assert result["step-2"]["output"]["data"].shape == data.shape

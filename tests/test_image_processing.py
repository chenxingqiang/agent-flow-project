import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import ray
from agentflow.core.instructions.advanced import ImageProcessingInstruction
from transformers import ViTImageProcessor, ViTForImageClassification

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Set up Ray for testing."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture
def mock_processor():
    """Mock ViT processor"""
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return processor

@pytest.fixture
def mock_model():
    """Mock ViT model"""
    model = MagicMock()
    model.return_value = MagicMock(logits=torch.randn(1, 1000))
    return model

@pytest.fixture
def sample_image():
    """Create test image"""
    return torch.randn(3, 224, 224)  # Mock image tensor

@pytest.fixture
def large_image():
    """Create large test image"""
    return torch.randn(3, 1000, 1000)

@pytest.fixture
def batch_images():
    """Create test image batch"""
    return [torch.randn(3, 224, 224) for _ in range(5)]

class TestImageProcessingInstruction:
    
    async def async_setup(self):
        """Set up test case."""
        self.mock_processor = MagicMock()
        self.mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        self.mock_model = MagicMock()
        self.mock_model.return_value = MagicMock(logits=torch.randn(1, 1000))
        
        with patch('agentflow.core.instructions.advanced.ViTImageProcessor.from_pretrained', return_value=self.mock_processor), \
             patch('agentflow.core.instructions.advanced.ViTForImageClassification.from_pretrained', return_value=self.mock_model):
            self.instruction = ImageProcessingInstruction.remote(use_mock=True)
            self.test_image = torch.randn(3, 224, 224)  # Mock image tensor
        
    async def async_teardown(self):
        """Clean up test case."""
        if hasattr(self, 'instruction'):
            ray.kill(self.instruction)
        
    async def test_initialization(self):
        """Test instruction initialization."""
        await self.async_setup()
        try:
            name = await self.instruction.get_name.remote()
            assert name == "process_image"
            assert self.instruction is not None
            processor = await self.instruction.get_processor.remote()
            model = await self.instruction.get_model.remote()
            assert processor is not None
            assert model is not None
        finally:
            await self.async_teardown()
        
    async def test_should_resize_image(self):
        """Test image resizing check."""
        await self.async_setup()
        try:
            large_image = torch.randn(3, 1000, 1000)
            result_large = await self.instruction.should_resize_image.remote({"image": large_image})
            assert result_large

            small_image = torch.randn(3, 100, 100)
            result_small = await self.instruction.should_resize_image.remote({"image": small_image})
            assert not result_small
        finally:
            await self.async_teardown()
        
    async def test_should_batch_process(self):
        """Test batch processing check."""
        await self.async_setup()
        try:
            images = [torch.randn(3, 224, 224) for _ in range(50)]
            result_batch = await self.instruction.should_batch_process.remote({"images": images})
            assert result_batch

            single_image = [torch.randn(3, 224, 224)]
            result_single = await self.instruction.should_batch_process.remote({"images": single_image})
            assert not result_single
        finally:
            await self.async_teardown()
        
    async def test_optimize(self):
        """Test optimization."""
        await self.async_setup()
        try:
            context = {"image": self.test_image}
            optimized = await self.instruction.optimize.remote(context)
            assert optimized is not None
            assert "image" in optimized
        finally:
            await self.async_teardown()
        
    async def test_resize_image(self):
        """Test image resizing."""
        await self.async_setup()
        try:
            large_image = torch.randn(3, 1000, 1000)
            resized = await self.instruction.resize_image.remote(large_image, (224, 224))
            assert resized.shape == (3, 224, 224)
        finally:
            await self.async_teardown()
        
    async def test_single_image_processing(self):
        """Test processing a single image."""
        await self.async_setup()
        try:
            context = {"image": self.test_image}
            result = await self.instruction._execute_impl.remote(context)
            assert result is not None
            assert "results" in result
        finally:
            await self.async_teardown()
        
    async def test_batch_image_processing(self):
        """Test processing multiple images."""
        await self.async_setup()
        try:
            images = [torch.randn(3, 224, 224) for _ in range(5)]
            context = {"images": images}
            result = await self.instruction._execute_impl.remote(context)
            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 5
        finally:
            await self.async_teardown()
        
    async def test_error_handling(self):
        """Test error handling."""
        await self.async_setup()
        try:
            context = {"invalid": "data"}
            with pytest.raises(ValueError):
                await self.instruction._execute_impl.remote(context)
        finally:
            await self.async_teardown()
            
    @patch('torch.no_grad', MagicMock())
    async def test_model_prediction(self):
        """Test model prediction."""
        await self.async_setup()
        try:
            context = {"image": self.test_image}
            result = await self.instruction._execute_impl.remote(context)
            assert result is not None
            assert "results" in result
        finally:
            await self.async_teardown()

if __name__ == "__main__":
    pytest.main([__file__])

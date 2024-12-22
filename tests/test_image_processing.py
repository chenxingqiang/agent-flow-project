import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import ray
from agentflow.core.instructions.advanced import ImageProcessingInstruction
from transformers import ViTImageProcessor, ViTForImageClassification

# Initialize Ray
ray.init(ignore_reinit_error=True)

pytestmark = pytest.mark.asyncio  # Mark all tests as async

@pytest.fixture
def image_processor():
    """Create ImageProcessingInstruction instance"""
    return ImageProcessingInstruction.remote()

@pytest.fixture
def sample_image():
    """Create test image"""
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def large_image():
    """Create large test image"""
    return Image.new('RGB', (2000, 2000), color='blue')

@pytest.fixture
def batch_images():
    """Create test image batch"""
    return [
        Image.new('RGB', (100, 100), color='red'),
        Image.new('RGB', (100, 100), color='green'),
        Image.new('RGB', (100, 100), color='blue')
    ]

class TestImageProcessingInstruction:
    
    async def test_initialization(self, image_processor):
        """Test initialization"""
        name = await image_processor.name.remote()
        assert name == "process_image"

    async def test_should_resize_image(self, image_processor, sample_image, large_image):
        """Test image resize logic"""
        # Test small image
        context_small = {"image": sample_image}
        result_small = await image_processor._should_resize_image.remote(context_small)
        assert not result_small

        # Test large image
        context_large = {"image": large_image}
        result_large = await image_processor._should_resize_image.remote(context_large)
        assert result_large

    async def test_should_batch_process(self, image_processor, batch_images):
        """Test batch processing logic"""
        # Single image
        context_single = {"images": [batch_images[0]]}
        result_single = await image_processor._should_batch_process.remote(context_single)
        assert not result_single

        # Multiple images
        context_batch = {"images": batch_images}
        result_batch = await image_processor._should_batch_process.remote(context_batch)
        assert result_batch

    async def test_optimize(self, image_processor, large_image):
        """Test optimization method"""
        context = {"image": large_image}
        optimized_context = await image_processor._optimize.remote(context)
        
        # Verify image is resized
        assert max(optimized_context["image"].size) <= 1000
        assert "use_batches" not in optimized_context  # Single image should not use batches

    async def test_resize_image(self, image_processor, sample_image):
        """Test image resizing functionality"""
        target_size = (224, 224)
        resized_image = await image_processor._resize_image.remote(sample_image, target_size)
        
        assert resized_image.size == target_size

    async def test_single_image_processing(self, image_processor, sample_image):
        """Test single image processing"""
        context = {
            "image": sample_image,
            "resize": True,
            "target_size": (224, 224)
        }
        
        result = await image_processor._execute_impl.remote(context)
        
        assert "result" in result
        assert "label" in result["result"]
        assert "confidence" in result["result"]
        assert result["total_processed"] == 1

    async def test_batch_image_processing(self, image_processor, batch_images):
        """Test batch image processing"""
        context = {
            "images": batch_images,
            "use_batches": True,
            "batch_size": 2,
            "resize": True,
            "target_size": (224, 224)
        }

        result = await image_processor._execute_impl.remote(context)

        assert "results" in result
        assert len(result["results"]) == len(batch_images)
        assert result["total_processed"] == len(batch_images)
        assert "batch_count" in result
        assert result["batch_count"] == 2  # With batch_size=2 and 3 images, we need 2 batches

    async def test_error_handling(self, image_processor):
        """Test error handling"""
        # Test invalid image input
        with pytest.raises(ValueError) as exc_info:
            context = {"image": None}
            await image_processor._execute_impl.remote(context)
        assert "Image cannot be None" in str(exc_info.value)

        # Test empty batch
        with pytest.raises(ValueError) as exc_info:
            context = {"images": []}
            await image_processor._execute_impl.remote(context)
        assert "Images list cannot be empty" in str(exc_info.value)

        # Test batch with None
        with pytest.raises(ValueError) as exc_info:
            context = {"images": [None]}
            await image_processor._execute_impl.remote(context)
        assert "Images list cannot contain None values" in str(exc_info.value)

    @patch('torch.no_grad', MagicMock())
    async def test_model_prediction(self, image_processor, sample_image):
        """Test model prediction functionality"""
        context = {
            "image": sample_image,
            "resize": True,
            "target_size": (224, 224)
        }

        result = await image_processor._execute_impl.remote(context)
        assert "result" in result
        assert "label" in result["result"]
        assert "confidence" in result["result"]
        assert result["total_processed"] == 1

if __name__ == "__main__":
    pytest.main([__file__])

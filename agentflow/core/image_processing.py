"""Image processing instruction implementation."""
import ray
from PIL import Image
from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
import logging
import asyncio
import time

from .base import InstructionBase, InstructionResult, InstructionMetrics, InstructionStatus

logger = logging.getLogger(__name__)

@ray.remote
class ImageProcessingInstruction(InstructionBase):
    """Image processing instruction implementation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize image processing instruction."""
        super().__init__(config or {})
        self.max_image_size = config.get('max_image_size', 4096)
        self.default_target_size = config.get('default_target_size', (224, 224))
        self.batch_size = config.get('batch_size', 32)
        self.name = config.get('name', 'ImageProcessor')
        
    @ray.method(num_returns=1)
    def get_name(self) -> str:
        """Get actor name."""
        return self.name
        
    @ray.method(num_returns=1)
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the context."""
        try:
            # Single image validation
            if 'image' in context:
                return (
                    isinstance(context['image'], (Image.Image, str)) and
                    context.get('max_size', float('inf')) <= self.max_image_size
                )
            # Batch image validation    
            elif 'images' in context:
                return (
                    isinstance(context['images'], list) and
                    all(isinstance(img, (Image.Image, str)) for img in context['images']) and
                    all(img.size[0] <= self.max_image_size and img.size[1] <= self.max_image_size 
                        for img in context['images'] if isinstance(img, Image.Image))
                )
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
            
    @ray.method(num_returns=1)
    async def should_resize_image(self, context: Dict[str, Any]) -> bool:
        """Check if image should be resized."""
        if 'image' in context:
            img = context['image']
            if isinstance(img, Image.Image):
                return (
                    img.size[0] > self.max_image_size or 
                    img.size[1] > self.max_image_size or
                    context.get('resize', False)
                )
        return False
        
    @ray.method(num_returns=1)
    async def should_batch_process(self, context: Dict[str, Any]) -> bool:
        """Check if batch processing should be used."""
        return (
            'images' in context and
            isinstance(context['images'], list) and
            len(context['images']) > 1 and
            context.get('use_batches', True)
        )
        
    @ray.method(num_returns=1)
    async def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the processing."""
        if await self.should_resize_image(context):
            target_size = context.get('target_size', self.default_target_size)
            if 'image' in context:
                context['image'] = await self.resize_image(context['image'], target_size)
            elif 'images' in context:
                context['images'] = [
                    await self.resize_image(img, target_size) 
                    for img in context['images']
                ]
        return context
        
    @ray.method(num_returns=1)
    async def resize_image(self, image: Union[Image.Image, str], target_size: tuple) -> Image.Image:
        """Resize image to target size."""
        if isinstance(image, str):
            image = Image.open(image)
        return image.resize(target_size, Image.LANCZOS)
        
    @ray.method(num_returns=1)
    async def process_single_image(self, image: Union[Image.Image, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single image."""
        try:
            if isinstance(image, str):
                image = Image.open(image)
                
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float()
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.permute(2, 0, 1)
            
            # Normalize
            image_tensor = image_tensor / 255.0
            
            return {
                'result': image_tensor,
                'shape': image_tensor.shape,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }
            
    @ray.method(num_returns=1)
    async def process_batch(self, images: List[Union[Image.Image, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of images."""
        try:
            batch_size = context.get('batch_size', self.batch_size)
            results = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self.process_single_image(img, context)
                    for img in batch
                ])
                results.extend(batch_results)
                
            return {
                'results': results,
                'batch_count': len(results),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }
            
    @ray.method(num_returns=1)
    async def execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute the instruction implementation."""
        try:
            start_time = time.time()
            
            # Validate context
            if not self.validate_context(context):
                raise ValueError("Validation failed")
                
            # Optimize processing
            context = await self.optimize(context)
            
            # Process images
            if await self.should_batch_process(context):
                result = await self.process_batch(context['images'], context)
            else:
                result = await self.process_single_image(context['image'], context)
                
            end_time = time.time()
            
            metrics = InstructionMetrics(
                start_time=start_time,
                end_time=end_time,
                tokens_used=0,
                memory_used=0,
                cache_hit=False,
                optimization_applied=bool(await self.should_resize_image(context)),
                parallel_execution=bool(await self.should_batch_process(context))
            )
            
            return InstructionResult(
                status=InstructionStatus.SUCCESS if result['status'] == 'success' else InstructionStatus.FAILED,
                data=result,
                error=result.get('error'),
                execution_time=end_time - start_time,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return InstructionResult(
                status=InstructionStatus.FAILED,
                data={},
                error=str(e),
                metrics=InstructionMetrics(
                    start_time=time.time(),
                    end_time=time.time()
                )
            ) 
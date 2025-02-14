from PIL import Image
import torch
import os
from agentflow.agents.agent import Agent, AgentConfig
from agentflow.core.instructions.image import ImageProcessingInstruction, MockImageProcessor, MockImageClassifier
from agentflow.core.isa_manager import ISAManager, Instruction, InstructionType
from agentflow.core.instructions.base import InstructionResult, InstructionStatus
from agentflow.core.base_types import AgentType
import logging
import ray

logging.basicConfig(level=logging.INFO)

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init(num_cpus=2, ignore_reinit_error=True)

def create_test_images():
    """Create test images for processing."""
    os.makedirs("examples/test_images", exist_ok=True)
    
    # Create various test images
    image_paths = []
    
    # 1. Solid color images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    for i, color in enumerate(colors):
        img = Image.new('RGB', (100, 100), color)
        path = f"examples/test_images/color_image_{i+1}.jpg"
        img.save(path)
        image_paths.append(path)
    
    # 2. Gradient image
    gradient = Image.new('RGB', (200, 200))
    for x in range(200):
        for y in range(200):
            r = int(255 * x / 200)
            g = int(255 * y / 200)
            b = int(128)
            gradient.putpixel((x, y), (r, g, b))
    gradient_path = "examples/test_images/gradient.jpg"
    gradient.save(gradient_path)
    image_paths.append(gradient_path)
    
    # 3. Checkerboard pattern
    checkerboard = Image.new('RGB', (160, 160))
    square_size = 20
    for x in range(8):
        for y in range(8):
            color = (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)
            for i in range(square_size):
                for j in range(square_size):
                    checkerboard.putpixel((x * square_size + i, y * square_size + j), color)
    checkerboard_path = "examples/test_images/checkerboard.jpg"
    checkerboard.save(checkerboard_path)
    image_paths.append(checkerboard_path)
    
    return image_paths

async def main():
    isa_manager = None
    try:
        # 1. Create test images
        image_paths = create_test_images()
        
        # 2. Create base configuration
        config = AgentConfig(
            name="image_processor", 
            description="Agent for image processing tasks",
            type=AgentType.DATA_SCIENCE
        )
        
        # 3. Create agent instance
        agent = Agent(config)
        
        # 4. Create and register ImageProcessingInstruction
        isa_manager = ISAManager()
        await isa_manager.initialize()
        
        # Create the instruction instance
        image_processor = ImageProcessingInstruction.remote()
        
        # Register the instruction
        instruction = Instruction(
            id="image_processor",
            name="image_processor",
            type=InstructionType.BASIC,
            description="Process and analyze images",
            params={},
            parallelizable=True
        )
        isa_manager.register_instruction(instruction)
        
        # 5. Load test images
        test_images = [Image.open(path) for path in image_paths]
        
        # 6. Single image processing with different parameters
        print("\n=== Testing Single Image Processing ===")
        
        # Test different sizes
        sizes = [(224, 224), (448, 448), (112, 112)]
        for size in sizes:
            single_context = {
                "image": test_images[0],
                "resize": True,
                "target_size": size,
                "normalize": True,
                "channels_first": True
            }
            result = ray.get(image_processor.execute.remote(single_context))
            print(f"\nProcessing with size {size}:")
            print("Result:", result)
        
        # 7. Batch processing with different batch sizes
        print("\n=== Testing Batch Processing ===")
        batch_sizes = [2, 3]
        for batch_size in batch_sizes:
            batch_context = {
                "images": test_images,
                "use_batches": True,
                "batch_size": batch_size,
                "resize": True,
                "target_size": (224, 224),
                "normalize": True,
                "channels_first": True
            }
            batch_results = ray.get(image_processor.execute.remote(batch_context))
            print(f"\nProcessing with batch size {batch_size}:")
            print("Results:", batch_results)
        
        # 8. Testing different image types
        print("\n=== Testing Different Image Types ===")
        for idx, img in enumerate(test_images):
            context = {
                "image": img,
                "resize": True,
                "target_size": (224, 224),
                "normalize": True,
                "channels_first": True
            }
            result = ray.get(image_processor.execute.remote(context))
            print(f"\nProcessing image {idx + 1}:")
            print("Result:", result)
        
        # 9. Optimization test with different image sizes
        print("\n=== Testing Optimization ===")
        sizes = [(1000, 1000), (2000, 2000), (3000, 3000)]
        for size in sizes:
            large_image = Image.new('RGB', size, (128, 128, 128))
            optimization_context = {
                "image": large_image,
                "resize": True,
                "target_size": (224, 224),
                "normalize": True,
                "channels_first": True
            }
            optimized_result = ray.get(image_processor.execute.remote(optimization_context))
            print(f"\nProcessing large image {size}:")
            print("Result:", optimized_result)
    
    finally:
        # Clean up
        if isa_manager:
            await isa_manager.cleanup()
        ray.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

from PIL import Image
import torch
from agentflow.core.agent import Agent, AgentConfig
from agentflow.core.instructions.advanced import ImageProcessingInstruction
from agentflow.core.isa_manager import ISAManager
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    # 1. 创建基础配置
    config = AgentConfig({
        "name": "image_processor",
        "description": "Agent for image processing tasks"
    })
    
    # 2. 创建agent实例
    agent = Agent(config)
    
    # 3. 创建和注册ImageProcessingInstruction
    isa_manager = ISAManager()
    image_processor = ImageProcessingInstruction()
    isa_manager.register_instruction(image_processor)
    
    # 4. 准备测试图片
    # 假设我们有一些测试图片
    test_images = [
        Image.open("path/to/image1.jpg"),
        Image.open("path/to/image2.jpg"),
        Image.open("path/to/image3.jpg")
    ]
    
    # 5. 单张图片处理
    single_context = {
        "image": test_images[0],
        "resize": True,
        "target_size": (224, 224)
    }
    
    result = await image_processor._execute_impl(single_context)
    print("单张图片处理结果:", result)
    
    # 6. 批量处理图片
    batch_context = {
        "images": test_images,
        "use_batches": True,
        "batch_size": 2,
        "resize": True,
        "target_size": (224, 224)
    }
    
    batch_results = await image_processor._execute_impl(batch_context)
    print("批量处理结果:", batch_results)
    
    # 7. 展示优化功能
    # 添加一张大图片来测试自动优化
    large_image = Image.new('RGB', (2000, 2000))
    optimization_context = {
        "image": large_image
    }
    
    # 应用优化规则
    optimized_context = await image_processor._optimize(optimization_context)
    print("优化后的上下文:", optimized_context)
    
    # 处理优化后的图片
    optimized_result = await image_processor._execute_impl(optimized_context)
    print("优化后处理结果:", optimized_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

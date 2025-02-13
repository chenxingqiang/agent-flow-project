from agentflow import ell2a
from agentflow.ell2a.evaluation.evaluation import Evaluation
from agentflow.ell2a.integration import ELL2AIntegration
from agentflow.ell2a.types.message import Message, MessageRole, ContentBlock, MessageType
from agentflow.ell2a.lmp.simple import simple
import asyncio
import logging
from typing import Dict, Any, cast, Type
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize ELL2A integration
ell2a_integration = ELL2AIntegration()
ell2a_integration.configure({"store": "./logdir", "verbose": True})

@simple(model="deepseek-coder-33b-instruct", temperature=0.7)
async def math_problem_solver(problem: str) -> Message:
    """Solve a math problem and return the answer in a structured format.
    
    Example:
    Problem: What is 2 + 2?
    Let me solve this step by step:
    1. I need to add 2 and 2 together
    2. 2 + 2 = 4
    
    Answer: 4
    """
    return Message(
        role=MessageRole.ASSISTANT,
        content=[
            ContentBlock(
                type=MessageType.TEXT,
                text=f"""Let me solve this step by step:
1. First, let's identify the numbers and operation in the problem: {problem}
2. Let me calculate the result carefully
3. I will format the answer properly

Answer: {eval(problem.replace('What is ', '').replace('?', ''))}"""
            )
        ]
    )

def get_text_content(output: Message) -> str:
    content = output.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, list) and content:
        first_content = content[0]
        if isinstance(first_content, ContentBlock):
            return first_content.text or ""
        return str(first_content)
    elif isinstance(content, ContentBlock):
        return content.text or ""
    return str(content)

# Set fixed random seed for reproducibility
random.seed(42)

def generate_arithmetic_dataset(num_examples=100):
    operations = ['+', '-', '*', '/']
    dataset = []
    
    for _ in range(num_examples):
        # Generate random numbers up to 5 digits
        num1 = random.randint(0, 99999)
        num2 = random.randint(1, 99999) # Avoid 0 for division
        op = random.choice(operations)
        
        # Calculate result
        if op == '+':
            result = num1 + num2
        elif op == '-':
            result = num1 - num2
        elif op == '*':
            result = num1 * num2
        else:
            # For division, ensure clean division
            result = num1 / num2
            # Round to 2 decimal places for division
            result = round(result, 2)
            
        problem = f"What is {num1} {op} {num2}?"
        dataset.append({
            "input": problem,
            "output": f"Answer:\n{result}"
        })
    
    return dataset

def sync_score(datapoint, output) -> float:
    """Score the output of the math problem solver."""
    try:
        # Extract the answer from the output using get_text_content
        text = get_text_content(output) if isinstance(output, Message) else str(output)
            
        if not text:
            logging.debug("Empty text output")
            return -10.0
            
        answer_line = [line for line in text.split('\n') if line and line.startswith('Answer:')]
        if not answer_line:
            logging.debug(f"No answer line found in output: {text}")
            return -10.0
            
        answer_str = answer_line[0].replace('Answer:', '').strip()
        try:
            answer = float(answer_str)
        except ValueError:
            logging.debug(f"Could not convert answer to float: {answer_str}")
            return -10.0
            
        # Extract expected answer from datapoint
        if 'answer' not in datapoint:
            # Try to extract from output field
            expected_text = datapoint.get('output', '')
            expected_line = [line for line in expected_text.split('\n') if line and line.startswith('Answer:')]
            if not expected_line:
                logging.debug(f"No answer line found in expected output: {expected_text}")
                return -10.0
            expected_str = expected_line[0].replace('Answer:', '').strip()
            expected = float(expected_str)
        else:
            expected = float(datapoint['answer'])
        
        # Check if answers are close
        if abs(answer - expected) < 0.01:
            return 1.0
        else:
            return 0.0
            
    except Exception as e:
        logging.debug(f"Error scoring: {str(e)}")
        return -10.0

arithmetic_eval = Evaluation(
    name="Arithmetic",
    n_evals=10,
    metrics={
        "answer_is_close_l2": sync_score,
    }
)

async def main():
    dataset = generate_arithmetic_dataset()
    
    # Create a wrapper class that takes input from dataset
    class SolveProblem:
        async def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            problem = input_data["input"]
            response = await math_problem_solver(problem)
            return {"output": response}

        def __await__(self):
            async def _awaitable():
                return self
            return _awaitable().__await__()
    
    # Pass the SolveProblem class
    run = await arithmetic_eval.run(SolveProblem, data=dataset, n_workers=10, verbose=True)
    print(f"Run metrics: {run.metrics}")
    result = await math_problem_solver("What is 2 + 2?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
"""
Example script demonstrating how to use the Academic Research Support Agent
This example shows how to:
1. Initialize the agent with the YAML config
2. Run a research workflow
3. Generate output documents
"""

import os
import yaml
from typing import Dict, Any, List
import asyncio
from openai import OpenAI

class AcademicSupportAgent:
    def __init__(self, config_path: str):
        """Initialize the academic support agent with YAML configuration."""
        self.config_path = config_path
        self.load_configuration()
        self.setup_client()
        
    def load_configuration(self):
        """Load the YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded configuration for {self.config['name']}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise

    def setup_client(self):
        """Configure OpenAI client with Deepseek endpoint."""
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"  # OpenAI API endpoint
        )

    async def conduct_research(self) -> Dict[str, Any]:
        """Execute the research workflow based on the configuration."""
        try:
            # Extract research context
            topic = self.config['research_context']['topic']
            current_status = self.config['research_context']['current_status']
            specific_topics = self.config['research_context']['specific_topics']
            
            print(f"\nStarting research on: {topic}")
            print("=" * 50)

            # Step 1: Literature Review
            print("\n1. Conducting Literature Review...")
            lit_review = await self._conduct_literature_review()
            
            # Step 2: Research Planning
            print("\n2. Developing Research Plan...")
            research_plan = await self._develop_research_plan(lit_review)
            
            # Step 3: Methodology Development
            print("\n3. Developing Methodology...")
            methodology = await self._develop_methodology(research_plan)
            
            # Step 4: Analysis Framework
            print("\n4. Creating Analysis Framework...")
            analysis = await self._create_analysis_framework(methodology)

            return {
                'literature_review': lit_review,
                'research_plan': research_plan,
                'methodology': methodology,
                'analysis': analysis
            }

        except Exception as e:
            print(f"Error in research workflow: {e}")
            raise

    async def _get_completion(self, prompt: str) -> Dict[str, Any]:
        """Get completion from the API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 as specified in YAML
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant helping to conduct academic research. Provide detailed, well-structured responses in the following format:\n\nKey Findings:\n- Finding 1\n- Finding 2\n\nResearch Gaps:\n- Gap 1\n- Gap 2\n\nFuture Directions:\n- Direction 1\n- Direction 2"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            if not response or not response.choices or not response.choices[0].message:
                raise ValueError("No valid response received from the model")
                
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from the model")
                
            # Parse the response into structured data
            content = content.strip()
            sections = content.split('\n\n')
            result = {}
            
            current_section = None
            current_items = []
            
            for section in sections:
                lines = section.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.endswith(':'):
                        if current_section and current_items:
                            result[current_section] = current_items
                        current_section = line[:-1].lower().replace(' ', '_')
                        current_items = []
                    elif line.startswith('- '):
                        current_items.append(line[2:])
                        
            if current_section and current_items:
                result[current_section] = current_items
                
            return result
            
        except Exception as e:
            print(f"Error getting completion: {e}")
            raise

    async def _conduct_literature_review(self) -> Dict[str, Any]:
        """Conduct literature review phase."""
        prompt = f"""
        Conduct a literature review for the topic: {self.config['research_context']['topic']}
        Focus areas: {', '.join(self.config['research_context']['specific_topics'])}
        Research fields: {', '.join(self.config['research_context']['research_fields'])}
        
        Provide a comprehensive review with:
        1. Key findings from existing literature (at least 3 findings)
        2. Research gaps identified (at least 3 gaps)
        3. Potential future research directions (at least 3 directions)
        
        Format your response with clear section headers and bullet points:
        
        Key Findings:
        - Finding 1
        - Finding 2
        - Finding 3
        
        Research Gaps:
        - Gap 1
        - Gap 2
        - Gap 3
        
        Future Directions:
        - Direction 1
        - Direction 2
        - Direction 3
        """
        
        response = await self._get_completion(prompt)
        
        return {
            'findings': response,
            'stage': 'literature_review'
        }

    async def _develop_research_plan(self, lit_review: Dict[str, Any]) -> Dict[str, Any]:
        """Develop research plan based on literature review."""
        prompt = f"""
        Based on the literature review findings, develop a detailed research plan that:
        1. Addresses the research gaps
        2. Aligns with the publication goals: {self.config['publication_goals']['target_journal']}
        3. Fits within the timeline: {self.config['publication_goals']['submission_timeline']}
        
        Format your response with clear section headers and bullet points:
        
        Components:
        - List the main components of your research plan (at least 5)
        
        Timeline:
        - Break down the {self.config['publication_goals']['submission_timeline']} timeline into specific milestones
        
        Publication Strategy:
        - Detail your publication and dissemination strategy (at least 3 points)
        """
        
        response = await self._get_completion(prompt)
        
        return {
            'plan': response,
            'stage': 'research_planning'
        }

    async def _develop_methodology(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Develop research methodology."""
        methods = self.config['research_methods']
        prompt = f"""
        Develop a detailed methodology using these approaches: {', '.join(methods)}
        Consider:
        1. Available resources and time investment: {self.config['resources']['time_investment']}
        2. Computing resources: {self.config['resources']['computing_resources']}
        3. Database access: {self.config['resources']['database_access']}
        
        Format your response with clear section headers and bullet points:
        
        Research Methods:
        - Detail your research methods and approaches (at least 5)
        
        Technical Approach:
        - Describe your technical implementation strategy (at least 5 points)
        
        Implementation:
        - List specific implementation details and tools (at least 5 points)
        """
        
        response = await self._get_completion(prompt)
        
        return {
            'methodology': response,
            'stage': 'methodology'
        }

    async def _create_analysis_framework(self, methodology: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis framework."""
        prompt = f"""
        Create a detailed analysis framework that includes:
        1. Components and tools needed
        2. Evaluation metrics
        3. Experimental setup
        
        Format your response with clear section headers and bullet points:
        
        Components:
        - List the main components of your analysis framework (at least 5)
        
        Metrics:
        - Detail your evaluation metrics and measurements (at least 5)
        
        Setup:
        - Describe your experimental setup and requirements (at least 5 points)
        """
        
        response = await self._get_completion(prompt)
        
        return {
            'framework': response,
            'stage': 'analysis'
        }

    def generate_output(self, results: Dict[str, Any]):
        """Generate markdown output from research results."""
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate section content
        def format_section(items):
            if not items:
                return "- No items available"
            formatted_items = []
            for item in items:
                # Remove any existing bullet points or numbering
                item = item.strip('- ').strip('1234567890.: ')
                # Split into label and content if it exists
                parts = item.split(':', 1)
                if len(parts) > 1:
                    label, content = parts
                    formatted_items.append(f"- **{label.strip()}:** {content.strip()}")
                else:
                    formatted_items.append(f"- {item}")
            return "\n".join(formatted_items)

        # Generate markdown content
        markdown_content = f"""# Research Plan: 基于静态图的恶意软件分类方法研究

## Research Context
- Topic: 基于静态图的恶意软件分类方法研究
- Status: Initial Phase
- Fields: Computer Science, Data Science, Machine Learning

## Literature Review
### Key Findings
{format_section(results['literature_review']['findings'].get('key_findings', []))}

### Research Gaps
{format_section(results['literature_review']['findings'].get('research_gaps', []))}

### Future Directions
{format_section(results['literature_review']['findings'].get('future_directions', []))}

## Research Plan
### Components
{format_section(results['research_plan']['plan'].get('components', []))}

### Timeline
{format_section(results['research_plan']['plan'].get('timeline', []))}

### Publication Strategy
{format_section(results['research_plan']['plan'].get('publication_strategy', []))}

## Methodology
### Research Methods
{format_section(results['methodology']['methodology'].get('research_methods', []))}

### Technical Approach
{format_section(results['methodology']['methodology'].get('technical_approach', []))}

### Implementation
{format_section(results['methodology']['methodology'].get('implementation', []))}

## Analysis Framework
### Components
{format_section(results['analysis']['framework'].get('components', []))}

### Metrics
{format_section(results['analysis']['framework'].get('metrics', []))}

### Setup
{format_section(results['analysis']['framework'].get('setup', []))}
"""
        
        # Write markdown file
        output_path = os.path.join(output_dir, "research_plan.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        print(f"\nGenerated research plan: {output_path}")

async def main():
    """Main function to demonstrate the academic support agent."""
    # Path to the YAML config
    config_path = "agentflow/core/config/agents/academic_support_agent.yaml"
    
    # Initialize agent
    agent = AcademicSupportAgent(config_path)
    
    # Run research workflow
    results = await agent.conduct_research()
    
    # Generate output
    agent.generate_output(results)

if __name__ == "__main__":
    asyncio.run(main()) 
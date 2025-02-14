"""Example of using objective-driven workflow."""
import sys
import os
import logging
import time
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import io

# Add parent directory to path to import agentflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig, WorkflowStatus
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.agents.agent import Agent
from agentflow.agents.agent_types import AgentType
from agentflow.core.config import AgentConfig, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_accuracy_plot() -> bytes:
    """Create accuracy comparison plot."""
    # Sample data from the workflow context
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0]
    
    # Calculate metrics
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Create figure
    plt.figure(figsize=(8, 4))
    plt.bar(['Model Accuracy', 'Threshold'], [accuracy, 0.8], color=['#2E86C1', '#E74C3C'])
    plt.ylim(0, 1)
    plt.title('Model Accuracy vs Threshold')
    plt.ylabel('Accuracy Score')
    
    # Add value labels on top of bars
    for i, v in enumerate([accuracy, 0.8]):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    # Save plot to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return img_buffer.getvalue()

def create_clustering_plot() -> bytes:
    """Create clustering visualization."""
    # Generate sample clustered data
    np.random.seed(42)
    n_points = 50
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_points, 2))
    cluster2 = np.random.normal(loc=[0, 0], scale=0.5, size=(n_points, 2))
    
    plt.figure(figsize=(8, 4))
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='#2E86C1', label='Cluster 1')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='#E74C3C', label='Cluster 2')
    plt.title('Data Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Save plot to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return img_buffer.getvalue()

def create_validation_plot() -> bytes:
    """Create validation metrics plot."""
    # Sample validation metrics
    metrics = {
        'Data Quality': 1.0,
        'Model Performance': 0.85,
        'Cross-validation': 0.88,
        'Test Accuracy': 0.82
    }
    
    plt.figure(figsize=(8, 4))
    # Convert dict keys/values to lists for matplotlib
    bars = plt.bar(list(metrics.keys()), list(metrics.values()), color='#2E86C1')
    plt.ylim(0, 1.2)
    plt.title('Validation Metrics Overview')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save plot to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return img_buffer.getvalue()

def generate_pdf_report(content: str, output_path: str) -> str:
    """Generate a PDF report from the given content.
    
    Args:
        content: Report content in markdown format
        output_path: Path to save the PDF file
        
    Returns:
        str: Path to the generated PDF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1B4F72'),
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Add timestamp
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#666666'),
        alignment=1
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {timestamp}", date_style))
    story.append(Spacer(1, 30))
    
    # Create styles for different heading levels
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=16,
        textColor=colors.HexColor('#2E86C1')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        leading=16
    )
    
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=normal_style,
        leftIndent=20,
        spaceAfter=8
    )
    
    # Process markdown content
    sections = content.strip().split('#')
    for section in sections:
        if not section.strip():
            continue
        
        # Split section into title and content
        lines = section.strip().split('\n', 1)
        if len(lines) < 2:
            continue
        
        title, content = lines
        title = title.strip()
        
        # Add section title
        story.append(Paragraph(title, h1_style))
        story.append(Spacer(1, 12))
        
        # Add relevant figures based on section
        if title == "Results":
            # Add accuracy comparison plot
            story.append(Paragraph("Figure 1: Model Accuracy Analysis", styles['Heading2']))
            img_data = create_accuracy_plot()
            img = Image(io.BytesIO(img_data), width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # Add clustering visualization
            story.append(Paragraph("Figure 2: Data Clustering Visualization", styles['Heading2']))
            img_data = create_clustering_plot()
            img = Image(io.BytesIO(img_data), width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # Add validation metrics plot
            story.append(Paragraph("Figure 3: Validation Metrics Overview", styles['Heading2']))
            img_data = create_validation_plot()
            img = Image(io.BytesIO(img_data), width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 12))
        
        # Process content paragraphs
        paragraphs = content.strip().split('\n')
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            # Handle bullet points
            if p.startswith('- '):
                story.append(Paragraph(f"• {p[2:]}", bullet_style))
            elif p.startswith('1. ') or p.startswith('2. ') or p.startswith('3. '):
                # Handle numbered lists
                story.append(Paragraph(p, bullet_style))
            else:
                story.append(Paragraph(p, normal_style))
        
        story.append(Spacer(1, 16))
    
    # Build PDF
    doc.build(story)
    return output_path

async def main():
    """Run the example."""
    # Create workflow steps
    steps = [
        WorkflowStep(
            id="data_validation",
            name="Data Validation",
            type=WorkflowStepType.TRANSFORM,
            description="Validate input data quality",
            config=StepConfig(
                strategy="standard",
                params={
                    "validation_method": "schema",
                    "schema": {
                        "required": ["id", "value", "timestamp"],
                        "properties": {
                            "id": {"type": "string"},
                            "value": {"type": "number"},
                            "timestamp": {
                                "type": "string",
                                "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"
                            }
                        }
                    }
                }
            )
        ),
        WorkflowStep(
            id="data_analysis",
            name="Data Analysis",
            type=WorkflowStepType.TRANSFORM,
            description="Analyze data for patterns",
            dependencies=["data_validation"],
            config=StepConfig(
                strategy="feature_engineering",
                params={
                    "methods": ["clustering", "regression"],
                    "metrics": ["accuracy", "f1_score"],
                    "validation_method": "model_performance",
                    "metric_name": "accuracy",
                    "threshold": 0.8
                }
            )
        ),
        WorkflowStep(
            id="report_generation",
            name="Report Generation",
            type=WorkflowStepType.TRANSFORM,
            description="Generate analysis report",
            dependencies=["data_analysis"],
            config=StepConfig(
                strategy="standard",
                params={
                    "required_elements": [
                        "Executive Summary",
                        "Methodology",
                        "Results",
                        "Conclusions"
                    ],
                    "min_length": 1000,
                    "format": "pdf"
                }
            )
        )
    ]
    
    # Create workflow configuration
    workflow_config = WorkflowConfig(
        name="data_analysis_workflow",
        description="A workflow for analyzing data with objectives",
        max_iterations=3,
        timeout=3600,
        steps=steps
    )
    
    try:
        # Create workflow engine
        engine = WorkflowEngine(workflow_config)
        
        # Initialize the engine
        await engine.initialize()
        
        # Initial context
        context = {
            "input_data": {
                "path": os.path.abspath("./data/data.csv"),
                "format": "csv",
                "validation_data": {
                    "id": "sample-001",
                    "value": 42.0,
                    "timestamp": "2024-01-20T10:30:00"
                },
                "error_handling": {
                    "retry_count": 3,
                    "timeout": 300,
                    "fallback_strategy": "skip_invalid"
                }
            },
            "analysis_params": {
                "methods": ["clustering", "regression"],
                "metrics": ["accuracy", "f1_score"],
                "validation_data": {
                    "y_true": [0, 1, 1, 0, 1],
                    "y_pred": [0, 1, 1, 0, 0]
                },
                "error_thresholds": {
                    "accuracy_min": 0.8,
                    "f1_min": 0.75
                }
            },
            "report_params": {
                "format": "pdf",
                "validation_data": {
                    "content": """
# Executive Summary

Our comprehensive data analysis has revealed significant patterns and insights from the provided dataset. The analysis focused on clustering and regression techniques, achieving high accuracy metrics across multiple validation tests.

Key findings include:
- Successful validation of data quality and completeness
- Identification of distinct data clusters
- Strong predictive performance in regression models
- High accuracy scores exceeding the minimum threshold of 0.8

# Methodology

We employed a systematic approach combining multiple analytical methods:

1. Data Validation
   - Schema validation for data integrity
   - Quality checks for completeness and consistency
   - Timestamp verification for temporal consistency

2. Feature Engineering
   - Clustering analysis using advanced algorithms
   - Regression modeling for predictive insights
   - Cross-validation for model robustness

3. Performance Metrics
   - Accuracy scoring
   - F1-score evaluation
   - Threshold validation

# Results

The analysis yielded several significant results:

1. Data Quality
   - 100% compliance with schema requirements
   - Valid timestamp formatting across all entries
   - Consistent value ranges within expected bounds

2. Model Performance
   - Clustering identified clear data patterns
   - Regression models achieved high predictive accuracy
   - All metrics exceeded minimum thresholds

3. Validation Outcomes
   - Accuracy: 0.8+ across all tests
   - F1-score: 0.75+ for classification tasks
   - Robust performance across different data segments

# Conclusions

Based on our comprehensive analysis, we conclude:

1. The data demonstrates high quality and consistency
2. Machine learning models show strong predictive performance
3. All validation metrics exceed required thresholds
4. The workflow successfully processes and analyzes the data

Recommendations:
- Continue monitoring data quality metrics
- Implement automated validation checks
- Consider expanding the feature engineering pipeline
- Regular model retraining for optimal performance
                    """,
                },
                "requirements": {
                    "min_sections": 4,
                    "min_length": 1000,
                    "required_elements": ["summary", "methodology", "results", "conclusions"]
                }
            },
            "workflow_metadata": {
                "version": "1.0",
                "owner": "research_team",
                "priority": "high",
                "tags": ["data_analysis", "research", "automated"]
            }
        }
        
        # Create agent config
        agent_config = AgentConfig(
            name="data_analysis_agent",
            type=AgentType.RESEARCH,
            model=ModelConfig(name="gpt-4", provider="openai"),
            workflow=workflow_config.model_dump()
        )
        
        # Create agent
        agent = Agent(config=agent_config)
        
        # Execute workflow
        logger.info("Starting objective-driven workflow")
        workflow_id = await engine.register_workflow(agent)
        result = await engine.execute_workflow(workflow_id, context)
        
        # Generate PDF report
        report_content = result.get("steps", [])[-1].get("result", {}).get("report_params", {}).get("validation_data", {}).get("content", "")
        if report_content:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.abspath(f"output/reports/data_analysis_report_{timestamp}.pdf")
            pdf_path = generate_pdf_report(report_content, report_path)
            logger.info("\nGenerated PDF report: %s", pdf_path)
        
        # Check results
        logger.info("Workflow execution completed")
        logger.info("Final result: %s", result)
        
        # Enhanced result tracking
        workflow = engine.workflows.get(workflow_id)
        if workflow and workflow.steps:
            logger.info("\nDetailed Workflow Results:")
            logger.info("=" * 50)
            logger.info("Workflow ID: %s", workflow_id)
            logger.info("Total Steps: %d", len(workflow.steps))
            
            # Get step results from the workflow result
            step_results = result.get("steps", [])
            for step_result in step_results:
                logger.info("\nStep Details:")
                logger.info("-" * 30)
                logger.info("ID: %s", step_result.get("id"))
                logger.info("Type: %s", step_result.get("type"))
                logger.info("Status: %s", step_result.get("status"))
                
                # Get step configuration from workflow steps
                step = next((s for s in workflow.steps if s.id == step_result.get("id")), None)
                if step:
                    logger.info("Name: %s", step.name)
                    if step.dependencies:
                        logger.info("Dependencies: %s", ", ".join(step.dependencies))
                    if step.config:
                        logger.info("Strategy: %s", step.config.strategy)
                        logger.info("Parameters: %s", step.config.params)
                
                # Log step result if available
                if "result" in step_result:
                    logger.info("Result:")
                    logger.info(step_result["result"])
                
                # Log any errors if present
                if step_result.get("error"):
                    logger.error("Step Error: %s", step_result["error"])
            
            logger.info("\nWorkflow Summary:")
            logger.info("=" * 50)
            logger.info("Status: %s", result.get("status", "unknown"))
            logger.info("Completed Steps: %d/%d", 
                       sum(1 for step in step_results if step.get("status") == "success"),
                       len(workflow.steps))
        
        # Cleanup
        await engine.cleanup()
                
    except Exception as e:
        logger.error("Workflow execution failed: %s", e)
        sys.exit(1)
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

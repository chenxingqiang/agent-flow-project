"""Example of using objective-driven workflow."""
import sys
import os
import logging
from typing import Dict, Any

# Add parent directory to path to import agentflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.core.workflow import WorkflowEngine
from agentflow.core.objective_handler import ObjectiveType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    # Example workflow configuration
    workflow_config = {
        "name": "data_analysis_workflow",
        "description": "A workflow for analyzing data with objectives",
        "max_retries": 3,
        "timeout": 3600,
        "objectives": {
            "data_validation": {
                "priority": 1,
                "success_criteria": [
                    {
                        "type": "data_quality",
                        "description": "Data should be properly formatted",
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
                ]
            },
            "data_analysis": {
                "priority": 2,
                "dependencies": ["data_validation"],
                "success_criteria": [
                    {
                        "type": "analysis_quality",
                        "description": "Model should achieve minimum performance",
                        "validation_method": "model_performance",
                        "metric_name": "accuracy",
                        "threshold": 0.8
                    }
                ]
            },
            "report_generation": {
                "priority": 3,
                "dependencies": ["data_analysis"],
                "success_criteria": [
                    {
                        "type": "report_quality",
                        "description": "Report should be comprehensive",
                        "validation_method": "content",
                        "required_elements": [
                            "Executive Summary",
                            "Methodology",
                            "Results",
                            "Conclusions"
                        ],
                        "min_length": 1000
                    }
                ]
            }
        }
    }
    
    try:
        # Create workflow engine
        engine = WorkflowEngine(workflow_config)
        
        # Add objectives with validation data
        engine.add_objective(
            objective_id="data_validation",
            objective_type=ObjectiveType.ANALYSIS,
            description="Validate input data quality",
            success_criteria=workflow_config["objectives"]["data_validation"]["success_criteria"],
            priority=workflow_config["objectives"]["data_validation"]["priority"],
            metadata={
                "validation_data": {
                    "id": "sample-001",
                    "value": 42.0,
                    "timestamp": "2024-01-20T10:30:00"
                }
            }
        )
        
        engine.add_objective(
            objective_id="data_analysis",
            objective_type=ObjectiveType.ANALYSIS,
            description="Analyze data for patterns",
            success_criteria=workflow_config["objectives"]["data_analysis"]["success_criteria"],
            priority=workflow_config["objectives"]["data_analysis"]["priority"],
            dependencies=["data_validation"],
            metadata={
                "validation_data": {
                    "y_true": [0, 1, 1, 0, 1],
                    "y_pred": [0, 1, 1, 0, 0]
                }
            }
        )
        
        engine.add_objective(
            objective_id="report_generation",
            objective_type=ObjectiveType.CREATION,
            description="Generate analysis report",
            success_criteria=workflow_config["objectives"]["report_generation"]["success_criteria"],
            priority=workflow_config["objectives"]["report_generation"]["priority"],
            dependencies=["data_analysis"],
            metadata={
                "validation_data": {
                    "content": """
                    # Executive Summary
                    This report presents the findings of our data analysis...
                    
                    # Methodology
                    We employed state-of-the-art machine learning techniques...
                    
                    # Results
                    The analysis revealed several key patterns...
                    
                    # Conclusions
                    Based on our findings, we recommend...
                    """
                }
            }
        )
        
        # Initial context
        context = {
            "input_data": {
                "path": "/path/to/data.csv",
                "format": "csv"
            },
            "analysis_params": {
                "methods": ["clustering", "regression"],
                "metrics": ["accuracy", "f1_score"]
            },
            "output_format": "pdf"
        }
        
        # Execute workflow
        logger.info("Starting objective-driven workflow")
        result_context = engine.execute_objective_workflow(context)
        
        # Check results
        logger.info("Workflow execution completed")
        logger.info("Final context: %s", result_context)
        
        # Check individual objective statuses and validation results
        for obj_id in ["data_validation", "data_analysis", "report_generation"]:
            status = engine.get_objective_status(obj_id)
            logger.info("Objective %s status: %s", obj_id, status)
            
            # Get validation results
            validation_results = engine.objective_workflow.objective_handler.validate_objective(obj_id)
            logger.info("Validation results for %s:", obj_id)
            for result in validation_results["criteria_results"]:
                logger.info("  - %s: %s", result["type"], "✓" if result["validated"] else "✗")
                if "message" in result:
                    logger.info("    %s", result["message"])
                if "score" in result:
                    logger.info("    Score: %.2f", result["score"])
                
    except Exception as e:
        logger.error("Workflow execution failed: %s", e)
        sys.exit(1)
        
if __name__ == "__main__":
    main()

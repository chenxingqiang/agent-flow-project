"""Tests for the objective handler module."""
import unittest
from datetime import datetime
from agentflow.core.objective_handler import (
    ObjectiveHandler,
    ObjectiveType,
    ObjectiveStatus,
    Success_Criteria,
    Objective
)

class TestObjectiveHandler(unittest.TestCase):
    """Test cases for ObjectiveHandler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = ObjectiveHandler(config={})
        self.test_objective_id = "test_objective"
        self.test_success_criteria = [
            {
                "type": "accuracy",
                "description": "Model accuracy should be above threshold",
                "threshold": 0.95,
                "validation_method": "metric_validation"
            }
        ]
        
    def test_create_objective(self):
        """Test objective creation"""
        objective = self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        self.assertEqual(objective.objective_id, self.test_objective_id)
        self.assertEqual(objective.type, ObjectiveType.TASK)
        self.assertEqual(len(objective.success_criteria), 1)
        self.assertEqual(objective.status, ObjectiveStatus.PENDING)
        
    def test_create_objective_with_string_type(self):
        """Test objective creation with string type"""
        objective = self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type="task",
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        self.assertEqual(objective.type, ObjectiveType.TASK)
        
    def test_get_objective(self):
        """Test getting objective by ID"""
        self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        objective = self.handler.get_objective(self.test_objective_id)
        self.assertIsNotNone(objective)
        self.assertEqual(objective.objective_id, self.test_objective_id)
        
    def test_update_objective_status(self):
        """Test updating objective status"""
        self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        objective = self.handler.update_objective_status(
            self.test_objective_id,
            ObjectiveStatus.IN_PROGRESS
        )
        
        self.assertEqual(objective.status, ObjectiveStatus.IN_PROGRESS)
        
    def test_set_current_objective(self):
        """Test setting current objective"""
        self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        objective = self.handler.set_current_objective(self.test_objective_id)
        self.assertIsNotNone(objective)
        self.assertEqual(
            self.handler.get_current_objective().objective_id,
            self.test_objective_id
        )
        
    def test_get_objectives_by_status(self):
        """Test getting objectives by status"""
        self.handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        pending_objectives = self.handler.get_objectives_by_status(ObjectiveStatus.PENDING)
        self.assertEqual(len(pending_objectives), 1)
        self.assertEqual(pending_objectives[0].objective_id, self.test_objective_id)
        
    def test_serialization(self):
        """Test serialization and deserialization"""
        original_handler = ObjectiveHandler(config={})
        original_handler.create_objective(
            objective_id=self.test_objective_id,
            objective_type=ObjectiveType.TASK,
            description="Test objective",
            success_criteria=self.test_success_criteria
        )
        
        # Convert to dict
        handler_dict = original_handler.to_dict()
        
        # Create new handler from dict
        new_handler = ObjectiveHandler.from_dict(handler_dict)
        
        # Verify objectives are equal
        original_obj = original_handler.get_objective(self.test_objective_id)
        new_obj = new_handler.get_objective(self.test_objective_id)
        
        self.assertEqual(original_obj.objective_id, new_obj.objective_id)
        self.assertEqual(original_obj.type, new_obj.type)
        self.assertEqual(original_obj.description, new_obj.description)
        self.assertEqual(len(original_obj.success_criteria), len(new_obj.success_criteria))
        
if __name__ == '__main__':
    unittest.main()

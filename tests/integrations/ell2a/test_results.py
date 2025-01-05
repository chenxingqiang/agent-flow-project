"""Test results module."""

import pytest
from typing import List, Dict, Any

from agentflow.ell2a.evaluation.results import _ResultDatapoint, EvaluationResults, Label
from agentflow.ell2a.stores.models.evaluations import EvaluationLabelerType
import numpy as np

def test_evaluation_results_from_rowar_results():
    # Test that from_rowar_results correctly converts rowar_results to EvaluationResults
    rowar_results = [
        _ResultDatapoint(
            output=("output1", "id1"),
            labels=[Label.PASS],
            metadata={"metric1": 0.95, "annotation1": "anno1"}
        ),
        _ResultDatapoint(
            output=("output2", "id2"),
            labels=[Label.FAIL],
            metadata={"metric1": 0.85, "annotation1": "anno2"}
        ),
    ]
    
    results = EvaluationResults()
    for result in rowar_results:
        for label in result.labels:
            results.add_result(label, result)
            
    assert results.total == 2
    assert results.passed == 1
    assert results.failed == 1
    assert results.skipped == 0
    
    # Check invocation_ids
    assert results.invocation_ids is not None
    assert results.invocation_ids.outputs == ["id1", "id2"]
    assert (results.invocation_ids.metrics["metric1"] == np.array(["id1", "id2"])).all()
    assert (results.invocation_ids.annotations["annotation1"] == np.array(["id1", "id2"])).all()
    assert (results.invocation_ids.criterion == np.array(["id1", "id2"])).all()

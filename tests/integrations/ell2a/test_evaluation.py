"""Test evaluation module."""

import pytest

from agentflow.ell2a.lmp import function
from datetime import datetime
from agentflow.ell2a.evaluation.evaluation import Evaluation, EvaluationRun

# Mock classes and functions
@function
def MockLMP(param=None, api_params=None):
    return "mock_output"

@function
def paramless(api_params=None):
    return "mock_output"

@pytest.fixture
def mock_evaluation():
    return Evaluation(
        name="test_evaluation",
        n_evals=10,
        samples_per_datapoint=2,
        metrics={"mock_metric": lambda x, y: 1.0},
        criterion=lambda x, y: True
    )

def test_evaluation_initialization(mock_evaluation):
    assert mock_evaluation.name == "test_evaluation"
    assert mock_evaluation.n_evals == 10
    assert mock_evaluation.samples_per_datapoint == 2
    assert "mock_metric" in mock_evaluation.metrics

def test_evaluation_run_process_single(mock_evaluation):
    data_point = {"input": {"param": "test_input"}}
    lmp = MockLMP
    required_params = False

    results = mock_evaluation._process_single(data_point, lmp, {}, required_params)
    assert len(results) == 1
    assert results[0] == "mock_output"

def test_evaluation_run_with_different_inputs(mock_evaluation):
    data_point = {"input": {"param": "test_input"}}
    lmp = MockLMP
    required_params = False

    results = mock_evaluation._process_single(data_point, lmp, {}, required_params)
    assert len(results) == 1
    assert results[0] == "mock_output"

def test_evaluation_run_with_missing_params(mock_evaluation):
    data_point = {"input": {"param": "test_input"}}
    lmp = MockLMP
    lmp_params = {}  # Missing required params
    required_params = False

    results = mock_evaluation._process_single(data_point, lmp, lmp_params, required_params)
    assert len(results) == 1
    assert results[0] == "mock_output"

def test_evaluation_run_with_criterion(mock_evaluation):
    data_point = {"input": {"param": "test_input"}}
    lmp = MockLMP
    required_params = False

    results = mock_evaluation._process_single(data_point, lmp, {}, required_params)
    assert len(results) == 1
    assert results[0] == "mock_output"

def test_evaluation_run(mock_evaluation):
    lmp = paramless

    evaluation_run = mock_evaluation.run(lmp, n_workers=1, verbose=False)
    assert evaluation_run.n_evals == 10
    assert evaluation_run.samples_per_datapoint == 2

def test_evaluation_run_with_invalid_input(mock_evaluation):
    data_point = {"input": 123}  # Invalid input type
    lmp = MockLMP
    required_params = True

    with pytest.raises(ValueError, match="Invalid input type: <class 'int'>"):
        mock_evaluation._process_single(data_point, lmp, {}, required_params)




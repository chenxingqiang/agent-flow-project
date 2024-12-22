"""Advanced testing system with automated test generation and performance testing."""
import time
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ..utils.metrics import MetricsCollector

class TestGenerator:
    """Automated test case generator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
        
    def generate_test_cases(self, 
                          spec: Dict[str, Any], 
                          num_cases: int = 10) -> List[Dict[str, Any]]:
        """Generate test cases based on specification."""
        test_cases = []
        for _ in range(num_cases):
            test_case = self._generate_single_case(spec)
            test_cases.append(test_case)
        return test_cases
    
    def _generate_single_case(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single test case."""
        test_case = {}
        for key, value_spec in spec.items():
            test_case[key] = self._generate_value(value_spec)
        return test_case
    
    def _generate_value(self, spec: Dict[str, Any]) -> Any:
        """Generate a value based on specification."""
        type_name = spec.get("type", "string")
        if type_name == "string":
            return self._generate_string(spec)
        elif type_name == "number":
            return self._generate_number(spec)
        elif type_name == "boolean":
            return np.random.choice([True, False])
        return None
    
    def _generate_string(self, spec: Dict[str, Any]) -> str:
        """Generate a string value."""
        length = spec.get("length", 10)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(np.random.choice(list(chars)) for _ in range(length))
    
    def _generate_number(self, spec: Dict[str, Any]) -> float:
        """Generate a numeric value."""
        min_val = spec.get("min", 0)
        max_val = spec.get("max", 100)
        return np.random.uniform(min_val, max_val)

class PerformanceTest:
    """Performance and stress testing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
        
    def run_load_test(self, 
                     target_func: callable, 
                     test_cases: List[Dict[str, Any]],
                     concurrency: int = 10,
                     duration: int = 60) -> Dict[str, Any]:
        """Run load test with specified concurrency."""
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while time.time() - start_time < duration:
                futures = []
                for test_case in test_cases:
                    future = executor.submit(self._run_single_test, 
                                          target_func, 
                                          test_case)
                    futures.append(future)
                
                for future in futures:
                    result = future.result()
                    results.append(result)
        
        return self._analyze_results(results)
    
    def _run_single_test(self, 
                        target_func: callable, 
                        test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and collect metrics."""
        start_time = time.time()
        try:
            result = target_func(**test_case)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        
        duration = time.time() - start_time
        return {
            "success": success,
            "duration": duration,
            "result": result
        }
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and generate metrics."""
        durations = [r["duration"] for r in results]
        success_rate = sum(r["success"] for r in results) / len(results)
        
        return {
            "total_requests": len(results),
            "success_rate": success_rate,
            "avg_duration": np.mean(durations),
            "p50_duration": np.percentile(durations, 50),
            "p95_duration": np.percentile(durations, 95),
            "p99_duration": np.percentile(durations, 99),
            "min_duration": min(durations),
            "max_duration": max(durations)
        }

class RegressionTest:
    """Performance regression testing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
        self.baseline = None
        
    def set_baseline(self, metrics: Dict[str, Any]):
        """Set baseline metrics for comparison."""
        self.baseline = metrics
    
    def check_regression(self, 
                        current_metrics: Dict[str, Any],
                        threshold: float = 0.1) -> Dict[str, Any]:
        """Check for performance regression against baseline."""
        if not self.baseline:
            return {"status": "no_baseline"}
        
        regressions = {}
        for key in self.baseline:
            if key in current_metrics:
                change = (current_metrics[key] - self.baseline[key]) / self.baseline[key]
                if abs(change) > threshold:
                    regressions[key] = {
                        "baseline": self.baseline[key],
                        "current": current_metrics[key],
                        "change": change
                    }
        
        return {
            "status": "regression" if regressions else "ok",
            "regressions": regressions
        }

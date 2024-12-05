import pytest
import sys
import os
import json
from datetime import datetime

def run_performance_tests():
    """
    Run performance tests with detailed reporting
    """
    # Ensure the project root is in the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

    # Performance test configuration
    performance_test_args = [
        '-v',  # Verbose output
        '--durations=10',  # Show 10 slowest test durations
        '--tb=short',  # Shorter traceback format
        os.path.join(os.path.dirname(__file__), 'test_agent_performance.py')
    ]

    # Run tests
    result = pytest.main(performance_test_args)

    # Generate performance report
    generate_performance_report(result)

    return result

def generate_performance_report(test_result):
    """
    Generate a detailed performance test report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "test_result": {
            "exit_code": test_result,
            "result_string": {
                0: "All tests passed",
                1: "Some tests failed",
                2: "Test collection failed",
                3: "No tests ran"
            }.get(test_result, "Unknown result")
        },
        "performance_metrics": {
            "total_workflows": 4,
            "concurrent_workflows": 4,
            "max_iterations": 3,
            "distributed_mode": False,  # Using thread pool
            "expected_max_duration": 0.5  # seconds
        }
    }

    # Write report to file
    report_path = os.path.join(os.path.dirname(__file__), 'performance_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Performance report generated: {report_path}")

if __name__ == "__main__":
    sys.exit(run_performance_tests())

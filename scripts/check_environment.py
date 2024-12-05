import sys
import os
import importlib

def check_environment():
    """Check Python environment and required dependencies"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # List of required packages
    required_packages = [
        'pytest', 
        'ray', 
        'pydantic', 
        'fastapi', 
        'openai'
    ]
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"{package}: ✓ Installed")
        except ImportError:
            print(f"{package}: ✗ Not found")

if __name__ == '__main__':
    check_environment()

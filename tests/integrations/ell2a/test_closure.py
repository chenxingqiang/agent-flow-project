"""Test closure module."""

import pytest
from typing import Optional, List, Dict, Any

from agentflow.ell2a.util.closure import (
    closure,
    get_closure_for_function,
    get_closure_for_class,
    get_closure_for_module
)

# Test class moved to a separate module
class _TestClass:
    """Test class for closure testing."""
    def __init__(self):
        self.x = 1
        
    def method1(self):
        y = 2
        def inner():
            return self.x + y
        return inner
    
    def method2(self):
        return self.x

def test_get_closure_for_function():
    """Test getting closure variables for a function."""
    x = 1
    y = "test"
    
    def outer(z: int):
        def inner():
            return f"{x}{y}{z}"
        return inner
    
    func = outer(2)
    closure_vars = get_closure_for_function(func)
    
    assert "x" in closure_vars
    assert "y" in closure_vars
    assert "z" in closure_vars
    assert closure_vars["x"] == 1
    assert closure_vars["y"] == "test"
    assert closure_vars["z"] == 2

def test_closure_decorator():
    """Test closure decorator."""
    x = 1
    
    @closure
    def test_func():
        return x + 1
    
    result = test_func()
    assert result == 2

def test_get_closure_for_class():
    """Test getting closure variables for a class."""
    test_obj = _TestClass()
    inner_func = test_obj.method1()
    
    class_closure = get_closure_for_class(_TestClass)
    method_closure = get_closure_for_function(inner_func)
    
    assert "self" in method_closure
    assert method_closure["self"].x == 1
    assert "y" in method_closure
    assert method_closure["y"] == 2

def test_get_closure_for_module():
    """Test getting closure variables for a module."""
    module_closure = get_closure_for_module("agentflow.ell2a.util.closure")
    
    # The module should have access to the closure functions
    assert "closure" in module_closure
    assert "get_closure_for_function" in module_closure
    assert "get_closure_for_class" in module_closure
    assert "get_closure_for_module" in module_closure

def test_nested_closure():
    """Test nested closure handling."""
    x = 1
    
    def level1():
        y = 2
        def level2():
            z = 3
            def level3():
                return x + y + z
            return level3
        return level2()
    
    func = level1()
    closure_vars = get_closure_for_function(func)
    
    assert "x" in closure_vars
    assert "y" in closure_vars
    assert "z" in closure_vars
    assert closure_vars["x"] == 1
    assert closure_vars["y"] == 2
    assert closure_vars["z"] == 3

def test_closure_with_defaults():
    """Test closure with default arguments."""
    x = 1
    
    def outer(y: int = 2):
        def inner(z: int = 3):
            return x + y + z
        return inner
    
    func = outer()
    closure_vars = get_closure_for_function(func)
    
    assert "x" in closure_vars
    assert "y" in closure_vars
    assert closure_vars["x"] == 1
    assert closure_vars["y"] == 2

def test_closure_with_globals():
    """Test closure with global variables."""
    global_var = 42
    
    def test_func():
        return global_var
    
    closure_vars = get_closure_for_function(test_func)
    
    assert "global_var" in closure_vars
    assert closure_vars["global_var"] == 42
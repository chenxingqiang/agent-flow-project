"""Utility functions for agentflow"""
import importlib


def import_class(class_path: str):
    """Import a class from a string path.
    
    Args:
        class_path: String path to the class (e.g. 'module.submodule.ClassName')
        
    Returns:
        The class object
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_path}: {str(e)}")

"""Setup configuration for AgentFlow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-flow-project",
    version="0.1.1",
    author="Xingqiang Chen",
    author_email="chenxingqiang@hotmail.com",
    description="A flexible and extensible framework for building and managing AI agents and workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenxingqiang/agent-flow-project",
    packages=find_packages(exclude=["tests*", "examples*", "docs*", "studio*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "pydantic>=2.0.0",
        "scikit-learn>=0.24.0",
        "ray>=2.0.0",
        "requests>=2.25.0",
        "jsonschema>=3.2.0",
        "pytest>=6.0.0",
        "pytest-asyncio>=0.14.0",
        "pytest-cov>=2.10.0",
        "aiohttp>=3.7.0",
        "fastapi>=0.65.0",
        "uvicorn>=0.13.0",
        "deepseek-ai>=0.1.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "test": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentflow=agentflow.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/chenxingqiang/agent-flow-project/issues",
        "Source": "https://github.com/chenxingqiang/agent-flow-project",
        "Documentation": "https://github.com/chenxingqiang/agent-flow-project/docs",
    },
) 
"""Setup configuration for AgentFlow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentflow",
    version="0.1.0",
    author="AgentFlow Team",
    author_email="team@agentflow.ai",
    description="A flexible framework for building and managing AI agents and workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentflow/agentflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
        "sqlalchemy>=1.4.0",
        "pyyaml>=6.0.0",
        "typing-extensions>=4.0.0",
        "asyncio>=3.4.3",
        "dataclasses>=0.6",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "pylint",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentflow=agentflow.cli:main",
        ],
    },
) 
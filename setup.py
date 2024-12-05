from setuptools import setup, find_packages

setup(
    name="agentflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jinja2>=3.0.0",
        "markdown>=3.3.0",
        "python-docx>=0.8.11",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
        "ray>=2.5.0",
        "backoff>=2.2.1",
        "tenacity>=8.2.3",
        "openai>=1.0.0"
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.20.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=0.950'
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.20.0',
            'pytest-mock>=3.10.0'
        ]
    },
    author="Chen Xingqiang",
    author_email="chenxingqiang@gmail.com",
    description="A flexible framework for AI agent workflows",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
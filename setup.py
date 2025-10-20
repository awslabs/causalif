# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="causalif",
    version="0.1.0",
    author="Subhro Bose",
    author_email="bossubhr@amazon.co.uk",
    description="LLM assisted causal reasoning with JAX and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bossubhr/Causalif-private",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "jupyter": [
            "nest-asyncio>=1.5.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "causalif=causalif.tools:causalif_tool",
        ],
    },
    keywords="causal reasoning, machine learning, nlp, rag, jax, networkx, causal inference, genai, llm",
    project_urls={
        "Bug Reports": "https://github.com/bossubhr/Causalif-private/issues",
        "Source": "https://github.com/bossubhr/Causalif-private",
        # "Documentation": "https://causalif.readthedocs.io/",
        "Documentation": "https://github.com/bossubhr/Causalif-private/blob/feature-Subhro/README.md",
    },
)

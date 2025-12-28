"""Setup script for conformal-rag-guardrails."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="conformal-rag-guardrails",
    version="0.1.0",
    author="Debu Sinha",
    author_email="debusinha2009@gmail.com",
    description="Certified limits of embedding-based hallucination detection in RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debu-sinha/conformal-rag-guardrails",
    project_urls={
        "Paper": "https://arxiv.org/abs/2512.15068",
        "Bug Tracker": "https://github.com/debu-sinha/conformal-rag-guardrails/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "datasets>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.270",
            "isort>=5.12.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
    },
)

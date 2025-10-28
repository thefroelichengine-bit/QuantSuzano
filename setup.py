"""
Setup script for eda-suzano package.
This bypasses pyproject.toml build backend issues.
"""

from setuptools import setup, find_packages

setup(
    name="eda-suzano",
    version="0.1.0",
    description="EDA pipeline for Suzano/Celulose market analysis",
    author="QuantSuzano Team",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "statsmodels>=0.14.0",
        "yfinance>=0.2.0",
        "typer>=0.9.0",
        "python-bcb>=0.1.6",
        "requests>=2.31.0",
        "pyarrow>=12.0.0",
        "openpyxl>=3.1.0",
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "schedule>=1.2.0",
        "arch>=5.0.0",
        "scipy>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "eda=eda.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)


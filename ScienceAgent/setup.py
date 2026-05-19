from setuptools import setup, find_packages

setup(
    name="scienceagent",
    version="0.0.1",
    description="LLM agent for discovering physics laws in simulated worlds",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "numpy",
        "jax",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)

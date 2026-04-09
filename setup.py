from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="readytofit",
    version="1.0.1",
    author="Marco Vandone",
    author_email="marco.vandone@gmail.com",
    description="A flexible and modular Python toolkit for multi-peak curve fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ReadyToFit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)

"""
PyPI build file
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

docs_extras = ["sphinx", "docutils"]

setuptools.setup(
    name="replaybuffer",
    version="0.1.0",
    author="Matthew Beveridge",
    author_email="mattjbeveridge21@gmail.com",
    description="A simple replay buffer for temporal recall.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattbev/replaybuffer",
    keywords=[],
    install_requires=[
        "numpy"
    ],
    extras_require={"docs": docs_extras},
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
